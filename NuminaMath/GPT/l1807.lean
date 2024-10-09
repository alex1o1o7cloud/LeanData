import Mathlib

namespace mean_of_all_students_is_79_l1807_180756

def mean_score_all_students (F S : ℕ) (f s : ℕ) (hf : f = 2/5 * s) : ℕ :=
  (36 * s + 75 * s) / ((2/5 * s) + s)

theorem mean_of_all_students_is_79 (F S : ℕ) (f s : ℕ) (hf : f = 2/5 * s) (hF : F = 90) (hS : S = 75) : 
  mean_score_all_students F S f s hf = 79 := by
  sorry

end mean_of_all_students_is_79_l1807_180756


namespace percentage_reduction_is_20_l1807_180747

def original_employees : ℝ := 243.75
def reduced_employees : ℝ := 195

theorem percentage_reduction_is_20 :
  (original_employees - reduced_employees) / original_employees * 100 = 20 := 
  sorry

end percentage_reduction_is_20_l1807_180747


namespace series_fraction_simplify_l1807_180799

theorem series_fraction_simplify :
  (3 + 6 - 12 + 24 + 48 - 96 + 192) / (6 + 12 - 24 + 48 + 96 - 192 + 384) = 1 / 2 :=
by 
  sorry

end series_fraction_simplify_l1807_180799


namespace find_X_eq_A_l1807_180701

variable {α : Type*}
variable (A X : Set α)

theorem find_X_eq_A (h : X ∩ A = X ∪ A) : X = A := by
  sorry

end find_X_eq_A_l1807_180701


namespace stock_price_is_500_l1807_180742

-- Conditions
def income : ℝ := 1000
def dividend_rate : ℝ := 0.50
def investment : ℝ := 10000
def face_value : ℝ := 100

-- Theorem Statement
theorem stock_price_is_500 : 
  (dividend_rate * face_value / (investment / 1000)) = 500 := by
  sorry

end stock_price_is_500_l1807_180742


namespace right_triangle_k_value_l1807_180717

theorem right_triangle_k_value (x : ℝ) (k : ℝ) (s : ℝ) 
(h_triangle : 3*x + 4*x + 5*x = k * (1/2 * 3*x * 4*x)) 
(h_square : s = 10) (h_eq_apothems : 4*x = s/2) : 
k = 8 / 5 :=
by {
  sorry
}

end right_triangle_k_value_l1807_180717


namespace scientific_notation_of_604800_l1807_180755

theorem scientific_notation_of_604800 : 604800 = 6.048 * 10^5 := 
sorry

end scientific_notation_of_604800_l1807_180755


namespace parallel_lines_perpendicular_lines_l1807_180739

-- Define the lines
def l₁ (a : ℝ) (x y : ℝ) := (a - 1) * x + 2 * y + 1 = 0
def l₂ (a : ℝ) (x y : ℝ) := x + a * y + 3 = 0

-- The first proof statement: lines l₁ and l₂ are parallel
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, l₁ a x y → l₂ a x y → (a * (a - 1) - 2 = 0)) → (a = 2 ∨ a = -1) :=
by
  sorry

-- The second proof statement: lines l₁ and l₂ are perpendicular
theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, l₁ a x y → l₂ a x y → ((a - 1) * 1 + 2 * a = 0)) → (a = -1 / 3) :=
by
  sorry

end parallel_lines_perpendicular_lines_l1807_180739


namespace quadratic_roots_correct_l1807_180720

def quadratic (b c : ℝ) (x : ℝ) : ℝ := x^2 + b * x + c

theorem quadratic_roots_correct (b c : ℝ) 
  (h₀ : quadratic b c (-2) = 5)
  (h₁ : quadratic b c (-1) = 0)
  (h₂ : quadratic b c 0 = -3)
  (h₃ : quadratic b c 1 = -4)
  (h₄ : quadratic b c 2 = -3)
  (h₅ : quadratic b c 4 = 5)
  : (quadratic b c (-1) = 0) ∧ (quadratic b c 3 = 0) :=
sorry

end quadratic_roots_correct_l1807_180720


namespace express_2_175_billion_in_scientific_notation_l1807_180783

-- Definition of scientific notation
def scientific_notation (a : ℝ) (n : ℤ) (value : ℝ) : Prop :=
  value = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10

-- Theorem stating the problem
theorem express_2_175_billion_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), scientific_notation a n 2.175e9 ∧ a = 2.175 ∧ n = 9 :=
by
  sorry

end express_2_175_billion_in_scientific_notation_l1807_180783


namespace arithmetic_sequence_general_formula_bn_sequence_sum_l1807_180719

/-- 
  In an arithmetic sequence {a_n}, a_2 = 5 and a_6 = 21. 
  Prove the general formula for the nth term a_n and the sum of the first n terms S_n. 
-/
theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : a 2 = 5) (h2 : a 6 = 21) : 
  (∀ n, a n = 4 * n - 3) ∧ (∀ n, S n = n * (2 * n - 1)) := 
sorry

/--
  Given b_n = 2 / (S_n + 5 * n), prove the sum of the first n terms T_n for the sequence {b_n}.
-/
theorem bn_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (b : ℕ → ℚ) (T : ℕ → ℚ) 
  (h1 : a 2 = 5) (h2 : a 6 = 21) 
  (ha : ∀ n, a n = 4 * n - 3) (hS : ∀ n, S n = n * (2 * n - 1)) 
  (hb : ∀ n, b n = 2 / (S n + 5 * n)) : 
  (∀ n, T n = 3 / 4 - 1 / (2 * (n + 1)) - 1 / (2 * (n + 2))) :=
sorry

end arithmetic_sequence_general_formula_bn_sequence_sum_l1807_180719


namespace positions_after_317_moves_l1807_180732

-- Define positions for the cat and dog
inductive ArchPosition
| North | East | South | West
deriving DecidableEq

inductive PathPosition
| North | Northeast | East | Southeast | South | Southwest
deriving DecidableEq

-- Define the movement function for cat and dog
def cat_position (n : Nat) : ArchPosition :=
  match n % 4 with
  | 0 => ArchPosition.North
  | 1 => ArchPosition.East
  | 2 => ArchPosition.South
  | _ => ArchPosition.West

def dog_position (n : Nat) : PathPosition :=
  match n % 6 with
  | 0 => PathPosition.North
  | 1 => PathPosition.Northeast
  | 2 => PathPosition.East
  | 3 => PathPosition.Southeast
  | 4 => PathPosition.South
  | _ => PathPosition.Southwest

-- Theorem statement to prove the positions after 317 moves
theorem positions_after_317_moves :
  cat_position 317 = ArchPosition.North ∧
  dog_position 317 = PathPosition.South :=
by
  sorry

end positions_after_317_moves_l1807_180732


namespace brown_shoes_count_l1807_180761

-- Definitions based on given conditions
def total_shoes := 66
def black_shoe_ratio := 2

theorem brown_shoes_count (B : ℕ) (H1 : black_shoe_ratio * B + B = total_shoes) : B = 22 :=
by
  -- Proof here is replaced with sorry for the purpose of this exercise
  sorry

end brown_shoes_count_l1807_180761


namespace necessary_not_sufficient_l1807_180796

theorem necessary_not_sufficient (a b c d : ℝ) : 
  (a + c > b + d) → (a > b ∧ c > d) :=
sorry

end necessary_not_sufficient_l1807_180796


namespace area_difference_l1807_180728

theorem area_difference (A B a b : ℝ) : (A * b) - (a * B) = A * b - a * B :=
by {
  -- proof goes here
  sorry
}

end area_difference_l1807_180728


namespace number_of_three_digit_numbers_divisible_by_17_l1807_180779

theorem number_of_three_digit_numbers_divisible_by_17 : 
  let k_min := Nat.ceil (100 / 17)
  let k_max := Nat.floor (999 / 17)
  ∃ n, 
    (n = k_max - k_min + 1) ∧ 
    (n = 53) := 
by
    sorry

end number_of_three_digit_numbers_divisible_by_17_l1807_180779


namespace volume_of_rectangular_prism_l1807_180762

-- Define the conditions
def side_of_square : ℕ := 35
def area_of_square : ℕ := 1225
def radius_of_sphere : ℕ := side_of_square
def length_of_prism : ℕ := (2 * radius_of_sphere) / 5
def width_of_prism : ℕ := 10
variable (h : ℕ) -- height of the prism

-- The theorem to prove
theorem volume_of_rectangular_prism :
  area_of_square = side_of_square * side_of_square →
  length_of_prism = (2 * radius_of_sphere) / 5 →
  radius_of_sphere = side_of_square →
  volume_of_prism = (length_of_prism * width_of_prism * h)
  → volume_of_prism = 140 * h :=
by sorry

end volume_of_rectangular_prism_l1807_180762


namespace equalized_distance_l1807_180776

noncomputable def wall_width : ℝ := 320 -- wall width in centimeters
noncomputable def poster_count : ℕ := 6 -- number of posters
noncomputable def poster_width : ℝ := 30 -- width of each poster in centimeters
noncomputable def equal_distance : ℝ := 20 -- equal distance in centimeters to be proven

theorem equalized_distance :
  let total_posters_width := poster_count * poster_width
  let remaining_space := wall_width - total_posters_width
  let number_of_spaces := poster_count + 1
  remaining_space / number_of_spaces = equal_distance :=
by {
  sorry
}

end equalized_distance_l1807_180776


namespace karl_total_income_is_53_l1807_180709

noncomputable def compute_income (tshirt_price pant_price skirt_price sold_tshirts sold_pants sold_skirts sold_refurbished_tshirts: ℕ) : ℝ :=
  let tshirt_income := 2 * tshirt_price
  let pant_income := sold_pants * pant_price
  let skirt_income := sold_skirts * skirt_price
  let refurbished_tshirt_price := (tshirt_price : ℝ) / 2
  let refurbished_tshirt_income := sold_refurbished_tshirts * refurbished_tshirt_price
  tshirt_income + pant_income + skirt_income + refurbished_tshirt_income

theorem karl_total_income_is_53 : compute_income 5 4 6 2 1 4 6 = 53 := by
  sorry

end karl_total_income_is_53_l1807_180709


namespace pencils_in_drawer_after_operations_l1807_180794

def initial_pencils : ℝ := 2
def pencils_added : ℝ := 3.5
def pencils_removed : ℝ := 1.2

theorem pencils_in_drawer_after_operations : ⌊initial_pencils + pencils_added - pencils_removed⌋ = 4 := by
  sorry

end pencils_in_drawer_after_operations_l1807_180794


namespace rationalized_expression_correct_A_B_C_D_E_sum_correct_l1807_180730

noncomputable def A : ℤ := -18
noncomputable def B : ℤ := 2
noncomputable def C : ℤ := 30
noncomputable def D : ℤ := 5
noncomputable def E : ℤ := 428
noncomputable def expression := 3 / (2 * Real.sqrt 18 + 5 * Real.sqrt 20)
noncomputable def rationalized_form := (A * Real.sqrt B + C * Real.sqrt D) / E

theorem rationalized_expression_correct :
  rationalized_form = (18 * Real.sqrt 2 - 30 * Real.sqrt 5) / -428 :=
by
  sorry

theorem A_B_C_D_E_sum_correct :
  A + B + C + D + E = 447 :=
by
  sorry

end rationalized_expression_correct_A_B_C_D_E_sum_correct_l1807_180730


namespace price_difference_is_99_cents_l1807_180704

-- Definitions for the conditions
def list_price : ℚ := 3996 / 100
def discount_super_savers : ℚ := 9
def discount_penny_wise : ℚ := 25 / 100 * list_price

-- Sale prices calculated based on the given conditions
def sale_price_super_savers : ℚ := list_price - discount_super_savers
def sale_price_penny_wise : ℚ := list_price - discount_penny_wise

-- Difference in prices
def price_difference : ℚ := sale_price_super_savers - sale_price_penny_wise

-- Prove that the price difference in cents is 99
theorem price_difference_is_99_cents : price_difference = 99 / 100 := 
by
  sorry

end price_difference_is_99_cents_l1807_180704


namespace mean_weight_of_soccer_team_l1807_180777

-- Define the weights as per the conditions
def weights : List ℕ := [64, 68, 71, 73, 76, 76, 77, 78, 80, 82, 85, 87, 89, 89]

-- Define the total weight
def total_weight : ℕ := 64 + 68 + 71 + 73 + 76 + 76 + 77 + 78 + 80 + 82 + 85 + 87 + 89 + 89

-- Define the number of players
def number_of_players : ℕ := 14

-- Calculate the mean weight
noncomputable def mean_weight : ℚ := total_weight / number_of_players

-- The proof problem statement
theorem mean_weight_of_soccer_team : mean_weight = 75.357 := by
  -- This is where the proof would go.
  sorry

end mean_weight_of_soccer_team_l1807_180777


namespace tg_sum_equal_l1807_180795

variable {a b c : ℝ}
variable {φA φB φC : ℝ}

-- The sides of the triangle are labeled such that a >= b >= c.
axiom sides_ineq : a ≥ b ∧ b ≥ c

-- The angles between the median and the altitude from vertices A, B, and C.
axiom angles_def : true -- This axiom is a placeholder. In actual use, we would define φA, φB, φC properly using the given geometric setup.

theorem tg_sum_equal : Real.tan φA + Real.tan φC = Real.tan φB := 
by 
  sorry

end tg_sum_equal_l1807_180795


namespace evaluate_expression_l1807_180726

variable (b : ℝ) -- assuming b is a real number, (if b should be of different type, modify accordingly)

theorem evaluate_expression (y : ℝ) (h : y = b + 9) : y - b + 5 = 14 :=
by
  sorry

end evaluate_expression_l1807_180726


namespace greatest_possible_radius_of_circle_l1807_180789

theorem greatest_possible_radius_of_circle
  (π : Real)
  (r : Real)
  (h : π * r^2 < 100 * π) :
  ∃ (n : ℕ), n = 9 ∧ (r : ℝ) ≤ 10 ∧ (r : ℝ) ≥ 9 :=
by
  sorry

end greatest_possible_radius_of_circle_l1807_180789


namespace sum_of_reciprocals_of_roots_l1807_180712

theorem sum_of_reciprocals_of_roots :
  ∀ (c d : ℝ),
  (6 * c^2 + 5 * c + 7 = 0) → 
  (6 * d^2 + 5 * d + 7 = 0) → 
  (c + d = -5 / 6) → 
  (c * d = 7 / 6) → 
  (1 / c + 1 / d = -5 / 7) :=
by
  intros c d h₁ h₂ h₃ h₄
  sorry

end sum_of_reciprocals_of_roots_l1807_180712


namespace gcd_of_36_and_60_is_12_l1807_180743

theorem gcd_of_36_and_60_is_12 :
  Nat.gcd 36 60 = 12 :=
sorry

end gcd_of_36_and_60_is_12_l1807_180743


namespace sequence_geometric_condition_l1807_180729

theorem sequence_geometric_condition
  (a : ℕ → ℤ)
  (p q : ℤ)
  (h1 : a 1 = -1)
  (h2 : ∀ n, a (n + 1) = 2 * (a n - n + 3))
  (h3 : ∀ n, (a (n + 1) - p * (n + 1) + q) = 2 * (a n - p * n + q)) :
  a (Int.natAbs (p + q)) = 40 :=
sorry

end sequence_geometric_condition_l1807_180729


namespace pastries_selection_l1807_180785

/--
Clara wants to purchase six pastries from an ample supply of five types: muffins, eclairs, croissants, scones, and turnovers. 
Prove that there are 210 possible selections using the stars and bars theorem.
-/
theorem pastries_selection : ∃ (selections : ℕ), selections = (Nat.choose (6 + 5 - 1) (5 - 1)) ∧ selections = 210 := by
  sorry

end pastries_selection_l1807_180785


namespace algebraic_expression_value_l1807_180723

theorem algebraic_expression_value (x y : ℝ) (h : 2 * x - y = 2) : 6 * x - 3 * y + 1 = 7 := 
by
  sorry

end algebraic_expression_value_l1807_180723


namespace Kyle_throws_farther_l1807_180713

theorem Kyle_throws_farther (Parker_distance : ℕ) (Grant_ratio : ℚ) (Kyle_ratio : ℚ) (Grant_distance : ℚ) (Kyle_distance : ℚ) :
  Parker_distance = 16 → 
  Grant_ratio = 0.25 → 
  Kyle_ratio = 2 → 
  Grant_distance = Parker_distance + Parker_distance * Grant_ratio → 
  Kyle_distance = Kyle_ratio * Grant_distance → 
  Kyle_distance - Parker_distance = 24 :=
by
  intros hp hg hk hg_dist hk_dist
  subst hp
  subst hg
  subst hk
  subst hg_dist
  subst hk_dist
  -- The proof steps are omitted
  sorry

end Kyle_throws_farther_l1807_180713


namespace min_dot_product_value_l1807_180749

noncomputable def dot_product_minimum (x : ℝ) : ℝ :=
  8 * x^2 + 4 * x

theorem min_dot_product_value :
  (∀ x, dot_product_minimum x ≥ -1 / 2) ∧ (∃ x, dot_product_minimum x = -1 / 2) :=
by
  sorry

end min_dot_product_value_l1807_180749


namespace ratio_boys_girls_l1807_180718

variable (S G : ℕ)

theorem ratio_boys_girls (h : (2 / 3 : ℚ) * G = (1 / 5 : ℚ) * S) :
  (S - G) * 3 = 7 * G := by
  -- Proof goes here
  sorry

end ratio_boys_girls_l1807_180718


namespace sum_not_divisible_by_10_iff_l1807_180748

theorem sum_not_divisible_by_10_iff (n : ℕ) :
  ¬ (1981^n + 1982^n + 1983^n + 1984^n) % 10 = 0 ↔ n % 4 = 0 :=
sorry

end sum_not_divisible_by_10_iff_l1807_180748


namespace evaluate_expression_l1807_180715

def f (x : ℕ) : ℕ := 4 * x + 2
def g (x : ℕ) : ℕ := 3 * x + 4

theorem evaluate_expression : f (g (f 3)) = 186 := 
by 
  sorry

end evaluate_expression_l1807_180715


namespace tom_rope_stories_l1807_180706

/-- Define the conditions given in the problem. --/
def story_length : ℝ := 10
def rope_length : ℝ := 20
def loss_percentage : ℝ := 0.25
def pieces_of_rope : ℕ := 4

/-- Theorem to prove the number of stories Tom can lower the rope down. --/
theorem tom_rope_stories (story_length rope_length loss_percentage : ℝ) (pieces_of_rope : ℕ) : 
    story_length = 10 → 
    rope_length = 20 →
    loss_percentage = 0.25 →
    pieces_of_rope = 4 →
    pieces_of_rope * rope_length * (1 - loss_percentage) / story_length = 6 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end tom_rope_stories_l1807_180706


namespace factorize_expression_l1807_180733

variable {X M N : ℕ}

theorem factorize_expression (x m n : ℕ) : x * m - x * n = x * (m - n) :=
sorry

end factorize_expression_l1807_180733


namespace geometric_sequence_second_term_l1807_180745

theorem geometric_sequence_second_term
  (first_term : ℕ) (fourth_term : ℕ) (r : ℕ)
  (h1 : first_term = 6)
  (h2 : first_term * r^3 = fourth_term)
  (h3 : fourth_term = 768) :
  first_term * r = 24 := by
  sorry

end geometric_sequence_second_term_l1807_180745


namespace green_balls_count_l1807_180771

theorem green_balls_count (b g : ℕ) (h1 : b = 15) (h2 : 5 * g = 3 * b) : g = 9 :=
by
  sorry

end green_balls_count_l1807_180771


namespace sum_of_products_l1807_180735

def is_positive (x : ℝ) := 0 < x

theorem sum_of_products 
  (x y z : ℝ) 
  (hx : is_positive x)
  (hy : is_positive y)
  (hz : is_positive z)
  (h1 : x^2 + x * y + y^2 = 27)
  (h2 : y^2 + y * z + z^2 = 25)
  (h3 : z^2 + z * x + x^2 = 52) :
  x * y + y * z + z * x = 30 :=
  sorry

end sum_of_products_l1807_180735


namespace square_properties_l1807_180752

theorem square_properties (perimeter : ℝ) (h1 : perimeter = 40) :
  ∃ (side length area diagonal : ℝ), side = 10 ∧ length = 10 ∧ area = 100 ∧ diagonal = 10 * Real.sqrt 2 :=
by
  sorry

end square_properties_l1807_180752


namespace first_sculpture_weight_is_five_l1807_180765

variable (w x y z : ℝ)

def hourly_wage_exterminator := 70
def daily_hours := 20
def price_per_pound := 20
def second_sculpture_weight := 7
def total_income := 1640

def income_exterminator := daily_hours * hourly_wage_exterminator
def income_sculptures := total_income - income_exterminator
def income_second_sculpture := second_sculpture_weight * price_per_pound
def income_first_sculpture := income_sculptures - income_second_sculpture

def weight_first_sculpture := income_first_sculpture / price_per_pound

theorem first_sculpture_weight_is_five :
  weight_first_sculpture = 5 := sorry

end first_sculpture_weight_is_five_l1807_180765


namespace chord_length_intercepted_by_curve_l1807_180707

theorem chord_length_intercepted_by_curve
(param_eqns : ∀ θ : ℝ, (x = 2 * Real.cos θ ∧ y = 1 + 2 * Real.sin θ))
(line_eqn : 3 * x - 4 * y - 1 = 0) :
  ∃ (chord_length : ℝ), chord_length = 2 * Real.sqrt 3 := 
sorry

end chord_length_intercepted_by_curve_l1807_180707


namespace TeresaTotalMarks_l1807_180738

/-- Teresa's scores in various subjects as given conditions -/
def ScienceScore := 70
def MusicScore := 80
def SocialStudiesScore := 85
def PhysicsScore := 1 / 2 * MusicScore

/-- Total marks Teresa scored in all the subjects -/
def TotalMarks := ScienceScore + MusicScore + SocialStudiesScore + PhysicsScore

/-- Proof statement: The total marks scored by Teresa in all subjects is 275. -/
theorem TeresaTotalMarks : TotalMarks = 275 := by
  sorry

end TeresaTotalMarks_l1807_180738


namespace office_expense_reduction_l1807_180798

theorem office_expense_reduction (x : ℝ) (h : 0 ≤ x) (h' : x ≤ 1) : 
  2500 * (1 - x) ^ 2 = 1600 :=
sorry

end office_expense_reduction_l1807_180798


namespace total_oranges_l1807_180774

theorem total_oranges (a b c : ℕ) (h1 : a = 80) (h2 : b = 60) (h3 : c = 120) : a + b + c = 260 :=
by
  sorry

end total_oranges_l1807_180774


namespace min_value_y_l1807_180740

theorem min_value_y : ∀ (x : ℝ), ∃ y_min : ℝ, y_min = (x^2 + 16 * x + 10) ∧ ∀ (x' : ℝ), (x'^2 + 16 * x' + 10) ≥ y_min := 
by 
  sorry

end min_value_y_l1807_180740


namespace expected_messages_xiaoli_l1807_180741

noncomputable def expected_greeting_messages (probs : List ℝ) (counts : List ℕ) : ℝ :=
  List.sum (List.zipWith (λ p c => p * c) probs counts)

theorem expected_messages_xiaoli :
  expected_greeting_messages [1, 0.8, 0.5, 0] [8, 15, 14, 3] = 27 :=
by
  -- The proof will use the expected value formula
  sorry

end expected_messages_xiaoli_l1807_180741


namespace cottonwood_fiber_diameter_in_scientific_notation_l1807_180725

theorem cottonwood_fiber_diameter_in_scientific_notation:
  (∃ (a : ℝ) (n : ℤ), 0.0000108 = a * 10 ^ n ∧ 1 ≤ a ∧ a < 10) → (0.0000108 = 1.08 * 10 ^ (-5)) :=
by
  sorry

end cottonwood_fiber_diameter_in_scientific_notation_l1807_180725


namespace domain_f₁_range_f₂_l1807_180793

noncomputable def f₁ (x : ℝ) : ℝ := (x - 2)^0 / Real.sqrt (x + 1)
noncomputable def f₂ (x : ℝ) : ℝ := 2 * x - Real.sqrt (x - 1)

theorem domain_f₁ : ∀ x : ℝ, x > -1 ∧ x ≠ 2 → ∃ y : ℝ, y = f₁ x :=
by
  sorry

theorem range_f₂ : ∀ y : ℝ, y ≥ 15 / 8 → ∃ x : ℝ, y = f₂ x :=
by
  sorry

end domain_f₁_range_f₂_l1807_180793


namespace swap_instruments_readings_change_l1807_180746

def U0 : ℝ := 45
def R : ℝ := 50
def r : ℝ := 20

theorem swap_instruments_readings_change :
  let I_total := U0 / (R / 2 + r)
  let U1 := I_total * r
  let I1 := I_total / 2
  let I2 := U0 / R
  let I := U0 / (R + r)
  let U2 := I * r
  let ΔI := I2 - I1
  let ΔU := U1 - U2
  ΔI = 0.4 ∧ ΔU = 7.14 :=
by
  sorry

end swap_instruments_readings_change_l1807_180746


namespace football_goals_in_fifth_match_l1807_180714

theorem football_goals_in_fifth_match (G : ℕ) (h1 : (4 / 5 : ℝ) = (4 - G) / 4 + 0.3) : G = 2 :=
by
  sorry

end football_goals_in_fifth_match_l1807_180714


namespace old_toilet_water_per_flush_correct_l1807_180711

noncomputable def old_toilet_water_per_flush (water_saved : ℕ) (flushes_per_day : ℕ) (days_in_june : ℕ) (reduction_percentage : ℚ) : ℚ :=
  let total_flushes := flushes_per_day * days_in_june
  let water_saved_per_flush := water_saved / total_flushes
  let reduction_factor := reduction_percentage
  let original_water_per_flush := water_saved_per_flush / (1 - reduction_factor)
  original_water_per_flush

theorem old_toilet_water_per_flush_correct :
  old_toilet_water_per_flush 1800 15 30 (80 / 100) = 5 := by
  sorry

end old_toilet_water_per_flush_correct_l1807_180711


namespace humans_can_live_l1807_180784

variable (earth_surface : ℝ)
variable (water_fraction : ℝ := 3 / 5)
variable (inhabitable_land_fraction : ℝ := 2 / 3)

def inhabitable_fraction : ℝ := (1 - water_fraction) * inhabitable_land_fraction

theorem humans_can_live :
  inhabitable_fraction = 4 / 15 :=
by
  sorry

end humans_can_live_l1807_180784


namespace problem_part1_problem_part2_l1807_180790

section DecreasingNumber

def is_decreasing_number (a b c d : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
  10 * a + b - (10 * b + c) = 10 * c + d

theorem problem_part1 (a : ℕ) :
  is_decreasing_number a 3 1 2 → a = 4 :=
by
  intro h
  -- Proof steps
  sorry

theorem problem_part2 (a b c d : ℕ) :
  is_decreasing_number a b c d →
  (100 * a + 10 * b + c + 100 * b + 10 * c + d) % 9 = 0 →
  8165 = max_value :=
by
  intro h1 h2
  -- Proof steps
  sorry

end DecreasingNumber

end problem_part1_problem_part2_l1807_180790


namespace dingding_minimum_correct_answers_l1807_180760

theorem dingding_minimum_correct_answers (x : ℕ) :
  (5 * x - (30 - x) > 100) → x ≥ 22 :=
by
  sorry

end dingding_minimum_correct_answers_l1807_180760


namespace a1_b1_sum_l1807_180758

-- Definitions from the conditions:
def strict_inc_seq (s : ℕ → ℕ) : Prop := ∀ n, s n < s (n + 1)

def positive_int_seq (s : ℕ → ℕ) : Prop := ∀ n, s n > 0

def a : ℕ → ℕ := sorry -- Define the sequence 'a' (details skipped).

def b : ℕ → ℕ := sorry -- Define the sequence 'b' (details skipped).

-- Conditions given:
axiom cond_a_inc : strict_inc_seq a

axiom cond_b_inc : strict_inc_seq b

axiom cond_a_pos : positive_int_seq a

axiom cond_b_pos : positive_int_seq b

axiom cond_a10_b10_lt_2017 : a 10 = b 10 ∧ a 10 < 2017

axiom cond_a_rec : ∀ n, a (n + 2) = a (n + 1) + a n

axiom cond_b_rec : ∀ n, b (n + 1) = 2 * b n

-- The theorem to prove:
theorem a1_b1_sum : a 1 + b 1 = 5 :=
sorry

end a1_b1_sum_l1807_180758


namespace total_cats_in_training_center_l1807_180703

-- Definitions corresponding to the given conditions
def cats_can_jump : ℕ := 60
def cats_can_fetch : ℕ := 35
def cats_can_meow : ℕ := 40
def cats_jump_fetch : ℕ := 20
def cats_fetch_meow : ℕ := 15
def cats_jump_meow : ℕ := 25
def cats_all_three : ℕ := 11
def cats_none : ℕ := 10

-- Theorem statement corresponding to proving question == answer given conditions
theorem total_cats_in_training_center
    (cjump : ℕ := cats_can_jump)
    (cfetch : ℕ := cats_can_fetch)
    (cmeow : ℕ := cats_can_meow)
    (cjf : ℕ := cats_jump_fetch)
    (cfm : ℕ := cats_fetch_meow)
    (cjm : ℕ := cats_jump_meow)
    (cat : ℕ := cats_all_three)
    (cno : ℕ := cats_none) :
    cjump
    + cfetch
    + cmeow
    - cjf
    - cfm
    - cjm
    + cat
    + cno
    = 96 := sorry

end total_cats_in_training_center_l1807_180703


namespace original_faculty_members_l1807_180700

theorem original_faculty_members (reduced_faculty : ℕ) (percentage : ℝ) : 
  reduced_faculty = 195 → percentage = 0.80 → 
  (∃ (original_faculty : ℕ), (original_faculty : ℝ) = reduced_faculty / percentage ∧ original_faculty = 244) :=
by
  sorry

end original_faculty_members_l1807_180700


namespace q_simplification_l1807_180744

noncomputable def q (x a b c D : ℝ) : ℝ :=
  (x + a)^2 / ((a - b) * (a - c)) + 
  (x + b)^2 / ((b - a) * (b - c)) + 
  (x + c)^2 / ((c - a) * (c - b))

theorem q_simplification (a b c D x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  q x a b c D = a + b + c + 2 * x + 3 * D / (a + b + c) :=
by
  sorry

end q_simplification_l1807_180744


namespace smallest_n_with_units_digit_and_reorder_l1807_180797

theorem smallest_n_with_units_digit_and_reorder :
  ∃ n : ℕ, (∃ a : ℕ, n = 10 * a + 6) ∧ (∃ m : ℕ, 6 * 10^m + a = 4 * n) ∧ n = 153846 :=
by
  sorry

end smallest_n_with_units_digit_and_reorder_l1807_180797


namespace number_of_people_in_group_l1807_180772

theorem number_of_people_in_group 
    (N : ℕ)
    (old_person_weight : ℕ) (new_person_weight : ℕ)
    (average_weight_increase : ℕ) :
    old_person_weight = 70 →
    new_person_weight = 94 →
    average_weight_increase = 3 →
    N * average_weight_increase = new_person_weight - old_person_weight →
    N = 8 :=
by
  sorry

end number_of_people_in_group_l1807_180772


namespace evaluate_expression_l1807_180787

noncomputable def expr : ℚ := (3 ^ 512 + 7 ^ 513) ^ 2 - (3 ^ 512 - 7 ^ 513) ^ 2
noncomputable def k : ℚ := 28 * 2.1 ^ 512

theorem evaluate_expression : expr = k * 10 ^ 513 :=
by
  sorry

end evaluate_expression_l1807_180787


namespace dragons_total_games_played_l1807_180791

theorem dragons_total_games_played (y x : ℕ)
  (h1 : x = 55 * y / 100)
  (h2 : x + 8 = 60 * (y + 12) / 100) :
  y + 12 = 28 :=
by
  sorry

end dragons_total_games_played_l1807_180791


namespace sum_of_cubes_l1807_180727

theorem sum_of_cubes (a b t : ℝ) (h : a + b = t^2) : 2 * (a^3 + b^3) = (a * t)^2 + (b * t)^2 + (a * t - b * t)^2 :=
by
  sorry

end sum_of_cubes_l1807_180727


namespace find_monthly_salary_l1807_180768

-- Definitions based on the conditions
def initial_saving_rate : ℝ := 0.25
def initial_expense_rate : ℝ := 1 - initial_saving_rate
def expense_increase_rate : ℝ := 1.25
def final_saving : ℝ := 300

-- Theorem: Prove the man's monthly salary
theorem find_monthly_salary (S : ℝ) (h1 : initial_saving_rate = 0.25)
  (h2 : initial_expense_rate = 0.75) (h3 : expense_increase_rate = 1.25)
  (h4 : final_saving = 300) : S = 4800 :=
by
  sorry

end find_monthly_salary_l1807_180768


namespace maggi_initial_packages_l1807_180763

theorem maggi_initial_packages (P : ℕ) (h1 : 4 * P - 5 = 12) : P = 4 :=
sorry

end maggi_initial_packages_l1807_180763


namespace minimum_value_ineq_l1807_180753

open Real

theorem minimum_value_ineq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
    (x + 1 / (y * y)) * (x + 1 / (y * y) - 500) + (y + 1 / (x * x)) * (y + 1 / (x * x) - 500) ≥ -125000 :=
by 
  sorry

end minimum_value_ineq_l1807_180753


namespace farmer_harvest_correct_l1807_180702

def estimated_harvest : ℕ := 48097
def additional_harvest : ℕ := 684
def total_harvest : ℕ := 48781

theorem farmer_harvest_correct : estimated_harvest + additional_harvest = total_harvest :=
by
  sorry

end farmer_harvest_correct_l1807_180702


namespace solve_for_x_l1807_180766

theorem solve_for_x : ∀ x : ℝ, (x - 5) ^ 3 = (1 / 27)⁻¹ → x = 8 := by
  intro x
  intro h
  sorry

end solve_for_x_l1807_180766


namespace intersection_l1807_180736

def setA : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) }
def setB : Set ℝ := { x | x^2 - 2 * x ≥ 0 }

theorem intersection: setA ∩ setB = { x : ℝ | x ≤ 0 } := by
  sorry

end intersection_l1807_180736


namespace ratio_ravi_kiran_l1807_180788

-- Definitions for the conditions
def ratio_money_ravi_giri := 6 / 7
def money_ravi := 36
def money_kiran := 105

-- The proof problem
theorem ratio_ravi_kiran : (money_ravi : ℕ) / money_kiran = 12 / 35 := 
by 
  sorry

end ratio_ravi_kiran_l1807_180788


namespace vector_dot_product_l1807_180754

variables (a b : ℝ × ℝ)
variables (ha : a = (1, -1)) (hb : b = (-1, 2))

theorem vector_dot_product : 
  ((2 • a + b) • a) = -1 :=
by
  -- This is where the proof would go
  sorry

end vector_dot_product_l1807_180754


namespace largest_integer_n_exists_l1807_180705

theorem largest_integer_n_exists :
  ∃ (x y z n : ℤ), (x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 5 * x + 5 * y + 5 * z - 10 = n^2) ∧ (n = 6) :=
by
  sorry

end largest_integer_n_exists_l1807_180705


namespace james_bike_ride_l1807_180757

variable {D P : ℝ}

theorem james_bike_ride :
  (∃ D P, 3 * D + (18 + 18 * 0.25) = 55.5 ∧ (18 = D * (1 + P / 100))) → P = 20 := by
  sorry

end james_bike_ride_l1807_180757


namespace solve_for_x_l1807_180724

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
by
  sorry

end solve_for_x_l1807_180724


namespace probability_drop_l1807_180708

open Real

noncomputable def probability_of_oil_drop_falling_in_hole (c : ℝ) : ℝ :=
  (0.25 * c^2) / (π * (c^2 / 4))

theorem probability_drop (c : ℝ) (hc : c > 0) : 
  probability_of_oil_drop_falling_in_hole c = 0.25 / π :=
by
  sorry

end probability_drop_l1807_180708


namespace largest_is_A_minus_B_l1807_180781

noncomputable def A := 3 * 1005^1006
noncomputable def B := 1005^1006
noncomputable def C := 1004 * 1005^1005
noncomputable def D := 3 * 1005^1005
noncomputable def E := 1005^1005
noncomputable def F := 1005^1004

theorem largest_is_A_minus_B :
  max (A - B) (max (B - C) (max (C - D) (max (D - E) (E - F)))) = A - B :=
by {
  sorry
}

end largest_is_A_minus_B_l1807_180781


namespace number_of_female_students_l1807_180767

theorem number_of_female_students (T S f_sample : ℕ) (H_total : T = 1600) (H_sample_size : S = 200) (H_females_in_sample : f_sample = 95) : 
  ∃ F, 95 / 200 = F / 1600 ∧ F = 760 := by 
sorry

end number_of_female_students_l1807_180767


namespace set_of_x_values_l1807_180764

theorem set_of_x_values (x : ℝ) : (3 ≤ abs (x + 2) ∧ abs (x + 2) ≤ 6) ↔ (1 ≤ x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x ≤ -5) := by
  sorry

end set_of_x_values_l1807_180764


namespace unique_quadruple_exists_l1807_180773

theorem unique_quadruple_exists :
  ∃! (a b c d : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
  a + b + c + d = 2 ∧
  a^2 + b^2 + c^2 + d^2 = 3 ∧
  (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 18 := by
  sorry

end unique_quadruple_exists_l1807_180773


namespace intersection_of_A_and_B_l1807_180721

def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {x | x^2 - 2*x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x < 1} := 
by
  sorry

end intersection_of_A_and_B_l1807_180721


namespace number_is_seven_l1807_180775

-- We will define the problem conditions and assert the answer
theorem number_is_seven (x : ℤ) (h : 3 * (2 * x + 9) = 69) : x = 7 :=
by 
  -- Proof will be filled in here
  sorry

end number_is_seven_l1807_180775


namespace sum_of_solutions_l1807_180737

-- Define the quadratic equation as a product of linear factors
def quadratic_eq (x : ℚ) : Prop := (4 * x + 6) * (3 * x - 8) = 0

-- Define the roots of the quadratic equation
def root1 : ℚ := -3 / 2
def root2 : ℚ := 8 / 3

-- Sum of the roots of the quadratic equation
def sum_of_roots : ℚ := root1 + root2

-- Theorem stating that the sum of the roots is 7/6
theorem sum_of_solutions : sum_of_roots = 7 / 6 := by
  sorry

end sum_of_solutions_l1807_180737


namespace volume_inequality_find_min_k_l1807_180750

noncomputable def cone_volume (R h : ℝ) : ℝ := (1 / 3) * Real.pi * R^2 * h

noncomputable def cylinder_volume (R h : ℝ) : ℝ :=
    let r := (R * h) / Real.sqrt (R^2 + h^2)
    Real.pi * r^2 * h

noncomputable def k_value (R h : ℝ) : ℝ := (R^2 + h^2) / (3 * h^2)

theorem volume_inequality (R h : ℝ) (h_pos : R > 0 ∧ h > 0) : 
    cone_volume R h ≠ cylinder_volume R h := by sorry

theorem find_min_k (R h : ℝ) (h_pos : R > 0 ∧ h > 0) (k : ℝ) :
    cone_volume R h = k * cylinder_volume R h → k = (R^2 + h^2) / (3 * h^2) := by sorry

end volume_inequality_find_min_k_l1807_180750


namespace solve_for_x_l1807_180792

theorem solve_for_x : (3.6 * 0.48 * x) / (0.12 * 0.09 * 0.5) = 800.0000000000001 → x = 2.5 :=
by
  sorry

end solve_for_x_l1807_180792


namespace triangle_inequality_range_l1807_180731

theorem triangle_inequality_range (x : ℝ) (h1 : 4 + 5 > x) (h2 : 4 + x > 5) (h3 : 5 + x > 4) :
  1 < x ∧ x < 9 := 
by
  sorry

end triangle_inequality_range_l1807_180731


namespace find_a_l1807_180710

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*x + a

theorem find_a :
  (∀ x : ℝ, 0 ≤ f x a) ∧ (∀ y : ℝ, ∃ x : ℝ, y = f x a) ↔ a = 1 := by
  sorry

end find_a_l1807_180710


namespace simplify_and_rationalize_denominator_l1807_180759

theorem simplify_and_rationalize_denominator :
  ( (Real.sqrt 5 / Real.sqrt 2) * (Real.sqrt 9 / Real.sqrt 6) * (Real.sqrt 8 / Real.sqrt 14) = 3 * Real.sqrt 420 / 42 ) := 
by {
  sorry
}

end simplify_and_rationalize_denominator_l1807_180759


namespace number_added_l1807_180778

def initial_number : ℕ := 9
def final_resultant : ℕ := 93

theorem number_added : ∃ x : ℕ, 3 * (2 * initial_number + x) = final_resultant ∧ x = 13 := by
  sorry

end number_added_l1807_180778


namespace smallest_value_of_x_l1807_180782

theorem smallest_value_of_x :
  ∃ x : ℝ, (x / 4 + 2 / (3 * x) = 5 / 6) ∧ (∀ y : ℝ,
    (y / 4 + 2 / (3 * y) = 5 / 6) → x ≤ y) :=
sorry

end smallest_value_of_x_l1807_180782


namespace combined_length_of_trains_l1807_180786

def length_of_train (speed_kmhr : ℕ) (time_sec : ℕ) : ℚ :=
  (speed_kmhr : ℚ) / 3600 * time_sec

theorem combined_length_of_trains :
  let L1 := length_of_train 300 33
  let L2 := length_of_train 250 44
  let L3 := length_of_train 350 28
  L1 + L2 + L3 = 8.52741 := by
  sorry

end combined_length_of_trains_l1807_180786


namespace hexagon_planting_schemes_l1807_180769

theorem hexagon_planting_schemes (n m : ℕ) (h : n = 4 ∧ m = 6) : 
  ∃ k, k = 732 := 
by sorry

end hexagon_planting_schemes_l1807_180769


namespace angle_C_is_70_l1807_180716

namespace TriangleAngleSum

def angle_sum_in_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

def sum_of_two_angles (A B : ℝ) : Prop :=
  A + B = 110

theorem angle_C_is_70 {A B C : ℝ} (h1 : angle_sum_in_triangle A B C) (h2 : sum_of_two_angles A B) : C = 70 :=
by
  sorry

end TriangleAngleSum

end angle_C_is_70_l1807_180716


namespace inscribed_cone_volume_l1807_180722

theorem inscribed_cone_volume
  (H : ℝ) 
  (α : ℝ)
  (h_pos : 0 < H)
  (α_pos : 0 < α ∧ α < π / 2) :
  (1 / 12) * π * H ^ 3 * (Real.sin α) ^ 2 * (Real.sin (2 * α)) ^ 2 = 
  (1 / 3) * π * ((H * Real.sin α * Real.cos α / 2) ^ 2) * (H * (Real.sin α) ^ 2) :=
by sorry

end inscribed_cone_volume_l1807_180722


namespace find_value_am2_bm_minus_7_l1807_180751

variable {a b m : ℝ}

theorem find_value_am2_bm_minus_7
  (h : a * m^2 + b * m + 5 = 0) : a * m^2 + b * m - 7 = -12 :=
by
  sorry

end find_value_am2_bm_minus_7_l1807_180751


namespace passengers_on_bus_l1807_180770

theorem passengers_on_bus (initial_passengers : ℕ) (got_on : ℕ) (got_off : ℕ) (final_passengers : ℕ) :
  initial_passengers = 28 → got_on = 7 → got_off = 9 → final_passengers = initial_passengers + got_on - got_off → final_passengers = 26 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end passengers_on_bus_l1807_180770


namespace range_f_log_l1807_180734

noncomputable def f : ℝ → ℝ := sorry

axiom f_even (x : ℝ) : f x = f (-x)
axiom f_increasing (x y : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ y) : f x ≤ f y
axiom f_at_1 : f 1 = 0

theorem range_f_log (x : ℝ) : f (Real.log x / Real.log (1 / 2)) > 0 ↔ (0 < x ∧ x < 1 / 2) ∨ (2 < x) :=
by
  sorry

end range_f_log_l1807_180734


namespace telephone_number_problem_l1807_180780

theorem telephone_number_problem :
  ∃ A B C D E F G H I J : ℕ,
    (A > B) ∧ (B > C) ∧ (D > E) ∧ (E > F) ∧ (G > H) ∧ (H > I) ∧ (I > J) ∧
    (D = E + 1) ∧ (E = F + 1) ∧ (D % 2 = 0) ∧ 
    (G = H + 2) ∧ (H = I + 2) ∧ (I = J + 2) ∧ (G % 2 = 1) ∧ (H % 2 = 1) ∧ (I % 2 = 1) ∧ (J % 2 = 1) ∧
    (A + B + C = 7) ∧ (B + C + F = 10) ∧ (A = 7) :=
sorry

end telephone_number_problem_l1807_180780
