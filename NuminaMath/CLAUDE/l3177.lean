import Mathlib

namespace number_problem_l3177_317791

theorem number_problem (x : ℝ) : (1/4 : ℝ) * x = (1/5 : ℝ) * (x + 1) + 1 → x = 24 := by
  sorry

end number_problem_l3177_317791


namespace soccer_players_count_l3177_317780

theorem soccer_players_count (total_socks : ℕ) (socks_per_player : ℕ) (h1 : total_socks = 22) (h2 : socks_per_player = 2) :
  total_socks / socks_per_player = 11 := by
  sorry

end soccer_players_count_l3177_317780


namespace cube_dot_path_length_l3177_317766

theorem cube_dot_path_length (cube_edge : ℝ) (h_edge : cube_edge = 2) :
  let face_diagonal := cube_edge * Real.sqrt 2
  let dot_path_radius := face_diagonal / 2
  let dot_path_length := 2 * Real.pi * dot_path_radius
  dot_path_length = 2 * Real.sqrt 2 * Real.pi :=
by sorry

end cube_dot_path_length_l3177_317766


namespace min_cost_to_buy_all_items_l3177_317719

def items : ℕ := 20

-- Define the set of prices
def prices : Finset ℕ := Finset.range items.succ

-- Define the promotion
def promotion_group_size : ℕ := 5
def free_items : ℕ := items / promotion_group_size

-- Define the minimum cost function
def min_cost : ℕ := (Finset.sum prices id) - (Finset.sum (Finset.filter (λ x => x > items - free_items) prices) id)

-- The theorem to prove
theorem min_cost_to_buy_all_items : min_cost = 136 := by
  sorry

end min_cost_to_buy_all_items_l3177_317719


namespace crayon_ratio_l3177_317701

def billies_crayons : ℕ := 18
def bobbies_crayons : ℕ := 3 * billies_crayons
def lizzies_crayons : ℕ := 27

theorem crayon_ratio : 
  (lizzies_crayons : ℚ) / (bobbies_crayons : ℚ) = 1 / 2 := by
  sorry

end crayon_ratio_l3177_317701


namespace quadratic_function_theorem_l3177_317788

/-- A quadratic function with real coefficients -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

/-- The theorem statement -/
theorem quadratic_function_theorem (a b : ℝ) :
  (∀ x, f a b x ≥ 0) →
  (∀ x, x ∈ Set.Ioo 1 7 ↔ f a b x < 9) →
  (∃ c, ∀ x, x ∈ Set.Ioo 1 7 ↔ f a b x < c) →
  (∃! c, ∀ x, x ∈ Set.Ioo 1 7 ↔ f a b x < c) ∧
  (∀ x, x ∈ Set.Ioo 1 7 ↔ f a b x < 9) := by
sorry

end quadratic_function_theorem_l3177_317788


namespace extreme_values_and_tangent_line_l3177_317700

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8

/-- The derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 6 * a * x + 3 * b

theorem extreme_values_and_tangent_line 
  (a b : ℝ) 
  (h1 : f' a b 1 = 0) 
  (h2 : f' a b 2 = 0) :
  (a = -3 ∧ b = 4) ∧ 
  (∃ (k m : ℝ), k = 12 ∧ m = 8 ∧ ∀ (x y : ℝ), y = k * x + m ↔ y = (f' (-3) 4 0) * x + f (-3) 4 0) := by
  sorry

end extreme_values_and_tangent_line_l3177_317700


namespace sunflower_seed_distribution_l3177_317744

theorem sunflower_seed_distribution (total_seeds : ℝ) (num_cans : ℝ) (seeds_per_can : ℝ) 
  (h1 : total_seeds = 54.0)
  (h2 : num_cans = 9.0)
  (h3 : seeds_per_can = total_seeds / num_cans) :
  seeds_per_can = 6.0 := by
sorry

end sunflower_seed_distribution_l3177_317744


namespace self_inverse_matrix_l3177_317734

def A (c d : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![4, -2; c, d]

theorem self_inverse_matrix (c d : ℚ) :
  A c d * A c d = 1 → c = 15/2 ∧ d = -4 := by
  sorry

end self_inverse_matrix_l3177_317734


namespace no_extreme_points_implies_a_leq_two_l3177_317756

/-- Given a function f(x) = x - 1/x - a*ln(x), if f has no extreme value points for x > 0,
    then a ≤ 2 --/
theorem no_extreme_points_implies_a_leq_two (a : ℝ) :
  (∀ x > 0, ∃ y > 0, (x - 1/x - a * Real.log x) < (y - 1/y - a * Real.log y) ∨
                     (x - 1/x - a * Real.log x) > (y - 1/y - a * Real.log y)) →
  a ≤ 2 := by
sorry


end no_extreme_points_implies_a_leq_two_l3177_317756


namespace sum_of_rectangle_areas_l3177_317740

def first_six_odd_numbers : List ℕ := [1, 3, 5, 7, 9, 11]

def rectangle_areas (width : ℕ) (lengths : List ℕ) : List ℕ :=
  lengths.map (λ l => width * l)

theorem sum_of_rectangle_areas :
  let width := 2
  let lengths := first_six_odd_numbers.map (λ n => n * n)
  let areas := rectangle_areas width lengths
  areas.sum = 572 := by sorry

end sum_of_rectangle_areas_l3177_317740


namespace f_negative_two_equals_negative_eight_l3177_317786

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then 3^x - 1 else -3^(-x) + 1

theorem f_negative_two_equals_negative_eight :
  f (-2) = -8 :=
by sorry

end f_negative_two_equals_negative_eight_l3177_317786


namespace average_age_proof_l3177_317787

theorem average_age_proof (a b c : ℕ) : 
  (a + b + c) / 3 = 26 → 
  b = 20 → 
  (a + c) / 2 = 29 :=
by sorry

end average_age_proof_l3177_317787


namespace arithmetic_calculation_l3177_317746

theorem arithmetic_calculation : 5 * 7.5 + 2 * 12 + 8.5 * 4 + 7 * 6 = 137.5 := by
  sorry

end arithmetic_calculation_l3177_317746


namespace isosceles_right_triangle_area_l3177_317722

/-- The area of an isosceles right triangle with hypotenuse 6 is 9 -/
theorem isosceles_right_triangle_area (h : ℝ) (A : ℝ) : 
  h = 6 →  -- The hypotenuse is 6 units
  A = h^2 / 4 →  -- Area formula for isosceles right triangle
  A = 9 := by
sorry

end isosceles_right_triangle_area_l3177_317722


namespace definite_integral_tan_cos_sin_l3177_317765

theorem definite_integral_tan_cos_sin : 
  ∫ x in (π / 4)..(Real.arcsin (2 / Real.sqrt 5)), (4 * Real.tan x - 5) / (4 * Real.cos x ^ 2 - Real.sin (2 * x) + 1) = 2 * Real.log (5 / 4) - (1 / 2) * Real.arctan (1 / 2) := by
  sorry

end definite_integral_tan_cos_sin_l3177_317765


namespace oxford_high_school_population_is_349_l3177_317716

/-- The number of people in Oxford High School -/
def oxford_high_school_population : ℕ :=
  let teachers : ℕ := 48
  let principal : ℕ := 1
  let classes : ℕ := 15
  let students_per_class : ℕ := 20
  let total_students : ℕ := classes * students_per_class
  teachers + principal + total_students

/-- Theorem stating the total number of people in Oxford High School -/
theorem oxford_high_school_population_is_349 :
  oxford_high_school_population = 349 := by
  sorry

end oxford_high_school_population_is_349_l3177_317716


namespace banana_muffins_count_l3177_317771

/-- Represents the types of pastries in the shop -/
inductive Pastry
  | PlainDoughnut
  | GlazedDoughnut
  | ChocolateChipCookie
  | OatmealCookie
  | BlueberryMuffin
  | BananaMuffin

/-- The ratio of pastries in the shop -/
def pastryRatio : Pastry → ℕ
  | Pastry.PlainDoughnut => 5
  | Pastry.GlazedDoughnut => 4
  | Pastry.ChocolateChipCookie => 3
  | Pastry.OatmealCookie => 2
  | Pastry.BlueberryMuffin => 1
  | Pastry.BananaMuffin => 2

/-- The number of plain doughnuts in the shop -/
def numPlainDoughnuts : ℕ := 50

/-- Theorem stating that the number of banana muffins is 20 -/
theorem banana_muffins_count :
  (numPlainDoughnuts / pastryRatio Pastry.PlainDoughnut) * pastryRatio Pastry.BananaMuffin = 20 := by
  sorry

end banana_muffins_count_l3177_317771


namespace evaluate_expression_l3177_317721

theorem evaluate_expression : (8^5 / 8^2) * 2^12 = 2^21 := by
  sorry

end evaluate_expression_l3177_317721


namespace increasing_function_condition_l3177_317710

/-- The function f(x) = x^2 + ax + 1/x is increasing on (1/3, +∞) if and only if a ≥ 25/3 -/
theorem increasing_function_condition (a : ℝ) :
  (∀ x > 1/3, Monotone (fun x => x^2 + a*x + 1/x)) ↔ a ≥ 25/3 := by
  sorry

end increasing_function_condition_l3177_317710


namespace shiela_drawings_l3177_317796

theorem shiela_drawings (neighbors : ℕ) (drawings_per_neighbor : ℕ) 
  (h1 : neighbors = 6) 
  (h2 : drawings_per_neighbor = 9) : 
  neighbors * drawings_per_neighbor = 54 := by
  sorry

end shiela_drawings_l3177_317796


namespace gcd_of_three_numbers_l3177_317713

theorem gcd_of_three_numbers : Nat.gcd 279 (Nat.gcd 372 465) = 93 := by
  sorry

end gcd_of_three_numbers_l3177_317713


namespace only_14_satisfies_l3177_317751

def is_multiple_of_three (n : ℤ) : Prop := ∃ k : ℤ, n = 3 * k

def is_perfect_square (n : ℤ) : Prop := ∃ k : ℤ, n = k * k

def sum_of_digits (n : ℤ) : ℕ :=
  (n.natAbs.repr.toList.map (λ c => c.toNat - '0'.toNat)).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

def satisfies_conditions (n : ℤ) : Prop :=
  ¬(is_multiple_of_three n) ∧
  ¬(is_perfect_square n) ∧
  is_prime (sum_of_digits n)

theorem only_14_satisfies :
  satisfies_conditions 14 ∧
  ¬(satisfies_conditions 12) ∧
  ¬(satisfies_conditions 16) ∧
  ¬(satisfies_conditions 21) ∧
  ¬(satisfies_conditions 26) :=
sorry

end only_14_satisfies_l3177_317751


namespace cos_neg_570_deg_l3177_317727

theorem cos_neg_570_deg : Real.cos ((-570 : ℝ) * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_neg_570_deg_l3177_317727


namespace age_difference_l3177_317793

theorem age_difference (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 27 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 29 := by
sorry

end age_difference_l3177_317793


namespace investment_relationship_l3177_317726

def initial_AA : ℝ := 150
def initial_BB : ℝ := 100
def initial_CC : ℝ := 200

def year1_AA_change : ℝ := 0.10
def year1_BB_change : ℝ := -0.20
def year1_CC_change : ℝ := 0.05

def year2_AA_change : ℝ := -0.05
def year2_BB_change : ℝ := 0.15
def year2_CC_change : ℝ := -0.10

def final_AA : ℝ := initial_AA * (1 + year1_AA_change) * (1 + year2_AA_change)
def final_BB : ℝ := initial_BB * (1 + year1_BB_change) * (1 + year2_BB_change)
def final_CC : ℝ := initial_CC * (1 + year1_CC_change) * (1 + year2_CC_change)

theorem investment_relationship : final_BB < final_AA ∧ final_AA < final_CC := by
  sorry

end investment_relationship_l3177_317726


namespace arctg_sum_quarter_pi_l3177_317799

theorem arctg_sum_quarter_pi (a b : ℝ) : 
  a = (1 : ℝ) / 2 → 
  (a + 1) * (b + 1) = 2 → 
  Real.arctan a + Real.arctan b = π / 4 := by
sorry

end arctg_sum_quarter_pi_l3177_317799


namespace mod_fourteen_power_ninety_six_minus_eight_l3177_317724

theorem mod_fourteen_power_ninety_six_minus_eight :
  (5^96 - 8) % 14 = 7 := by
sorry

end mod_fourteen_power_ninety_six_minus_eight_l3177_317724


namespace janes_babysitting_ratio_l3177_317797

/-- Represents the age ratio between a babysitter and a child -/
structure AgeRatio where
  babysitter : ℕ
  child : ℕ

/-- The problem setup for Jane's babysitting scenario -/
structure BabysittingScenario where
  jane_current_age : ℕ
  years_since_stopped : ℕ
  oldest_child_current_age : ℕ

/-- Calculates the age ratio between Jane and the oldest child she could have babysat -/
def calculate_age_ratio (scenario : BabysittingScenario) : AgeRatio :=
  { babysitter := scenario.jane_current_age - scenario.years_since_stopped,
    child := scenario.oldest_child_current_age - scenario.years_since_stopped }

/-- The main theorem to prove -/
theorem janes_babysitting_ratio :
  let scenario : BabysittingScenario := {
    jane_current_age := 34,
    years_since_stopped := 12,
    oldest_child_current_age := 25
  }
  let ratio := calculate_age_ratio scenario
  ratio.babysitter = 22 ∧ ratio.child = 13 := by sorry

end janes_babysitting_ratio_l3177_317797


namespace simplify_fraction_l3177_317730

theorem simplify_fraction : (90 + 54) / (150 - 90) = 12 / 5 := by
  sorry

end simplify_fraction_l3177_317730


namespace largest_a_when_b_equals_c_l3177_317760

theorem largest_a_when_b_equals_c (A B C : ℕ) 
  (h1 : A = 5 * B + C) 
  (h2 : B = C) : 
  A ≤ 24 ∧ ∃ (A₀ : ℕ), A₀ = 24 ∧ ∃ (B₀ C₀ : ℕ), A₀ = 5 * B₀ + C₀ ∧ B₀ = C₀ :=
by sorry

end largest_a_when_b_equals_c_l3177_317760


namespace convex_polygon_contains_half_homothety_l3177_317762

/-- A convex polygon in a 2D plane -/
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  convex : Convex ℝ (convexHull ℝ vertices)

/-- A homothety transformation -/
def homothety (center : ℝ × ℝ) (k : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (center.1 + k * (p.1 - center.1), center.2 + k * (p.2 - center.2))

/-- The theorem stating that a convex polygon contains its image under a 1/2 homothety -/
theorem convex_polygon_contains_half_homothety (P : ConvexPolygon) :
  ∃ (center : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ P.vertices →
    homothety center (1/2) p ∈ convexHull ℝ P.vertices :=
sorry

end convex_polygon_contains_half_homothety_l3177_317762


namespace greatest_value_quadratic_inequality_l3177_317768

theorem greatest_value_quadratic_inequality :
  ∀ b : ℝ, -b^2 + 8*b - 15 ≥ 0 → b ≤ 5 :=
by sorry

end greatest_value_quadratic_inequality_l3177_317768


namespace real_part_of_z_l3177_317783

theorem real_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) : 
  Complex.re z = 1 := by sorry

end real_part_of_z_l3177_317783


namespace dice_game_winning_probability_l3177_317747

/-- Represents the outcome of rolling three dice -/
inductive DiceOutcome
  | AllSame
  | TwoSame
  | AllDifferent

/-- The probability of winning the dice game -/
def winning_probability : ℚ := 2177 / 10000

/-- The strategy for rerolling dice based on the initial outcome -/
def reroll_strategy (outcome : DiceOutcome) : ℕ :=
  match outcome with
  | DiceOutcome.AllSame => 0
  | DiceOutcome.TwoSame => 1
  | DiceOutcome.AllDifferent => 3

theorem dice_game_winning_probability :
  ∀ (num_rolls : ℕ) (max_rerolls : ℕ),
    num_rolls = 3 ∧ max_rerolls = 2 →
    (∀ (outcome : DiceOutcome), reroll_strategy outcome ≤ num_rolls) →
    winning_probability = 2177 / 10000 := by
  sorry


end dice_game_winning_probability_l3177_317747


namespace class_age_problem_l3177_317711

theorem class_age_problem (total_students : ℕ) (total_avg_age : ℝ) 
  (group_a_students : ℕ) (group_a_avg_age : ℝ)
  (group_b_students : ℕ) (group_b_avg_age : ℝ) :
  total_students = 50 →
  total_avg_age = 24 →
  group_a_students = 15 →
  group_a_avg_age = 20 →
  group_b_students = 25 →
  group_b_avg_age = 25 →
  let group_c_students := total_students - (group_a_students + group_b_students)
  let total_age := total_students * total_avg_age
  let group_a_total_age := group_a_students * group_a_avg_age
  let group_b_total_age := group_b_students * group_b_avg_age
  let group_c_total_age := total_age - (group_a_total_age + group_b_total_age)
  let group_c_avg_age := group_c_total_age / group_c_students
  group_c_avg_age = 27.5 := by
    sorry

end class_age_problem_l3177_317711


namespace inequality_solution_set_l3177_317718

/-- The solution set of the inequality 3 - 2x - x^2 < 0 -/
def solution_set : Set ℝ := {x | x < -3 ∨ x > 1}

/-- The inequality function -/
def f (x : ℝ) := 3 - 2*x - x^2

theorem inequality_solution_set :
  ∀ x : ℝ, f x < 0 ↔ x ∈ solution_set :=
by sorry

end inequality_solution_set_l3177_317718


namespace largest_five_digit_with_product_180_l3177_317742

/-- A function that returns true if a number is a five-digit number -/
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

/-- A function that returns the product of digits of a natural number -/
def digit_product (n : ℕ) : ℕ :=
  sorry

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- The theorem to be proved -/
theorem largest_five_digit_with_product_180 :
  ∃ M : ℕ, is_five_digit M ∧
           digit_product M = 180 ∧
           (∀ n : ℕ, is_five_digit n → digit_product n = 180 → n ≤ M) ∧
           digit_sum M = 20 :=
by
  sorry

end largest_five_digit_with_product_180_l3177_317742


namespace revenue_difference_l3177_317776

def viewers_game2 : ℕ := 80
def viewers_game1 : ℕ := viewers_game2 - 20
def viewers_game3 : ℕ := viewers_game2 + 15
def viewers_game4 : ℕ := viewers_game3 + (viewers_game3 / 10) + 1 -- Rounded up

def price_game1 : ℕ := 15
def price_game2 : ℕ := 20
def price_game3 : ℕ := 25
def price_game4 : ℕ := 30

def viewers_last_week : ℕ := 350
def price_last_week : ℕ := 18

def revenue_this_week : ℕ := 
  viewers_game1 * price_game1 + 
  viewers_game2 * price_game2 + 
  viewers_game3 * price_game3 + 
  viewers_game4 * price_game4

def revenue_last_week : ℕ := viewers_last_week * price_last_week

theorem revenue_difference : 
  revenue_this_week - revenue_last_week = 1725 := by
  sorry

end revenue_difference_l3177_317776


namespace tina_win_probability_l3177_317792

theorem tina_win_probability (p_lose : ℚ) (h_lose : p_lose = 3/7) (h_no_tie : True) :
  1 - p_lose = 4/7 := by
  sorry

end tina_win_probability_l3177_317792


namespace negative_difference_l3177_317705

theorem negative_difference (a b : ℝ) : -(a - b) = -a + b := by
  sorry

end negative_difference_l3177_317705


namespace function_decomposition_l3177_317704

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem function_decomposition (f g : ℝ → ℝ) 
  (h_odd : is_odd f) (h_even : is_even g) 
  (h_sum : ∀ x, f x + g x = 2^x + 2*x) :
  (∀ x, g x = (2^x + 2^(-x)) / 2) ∧ 
  (∀ x, f x = 2^(x-1) + 2*x - 2^(-x-1)) :=
sorry

end function_decomposition_l3177_317704


namespace preimage_of_three_l3177_317739

def f (x : ℝ) : ℝ := 2 * x - 1

theorem preimage_of_three (x : ℝ) : f x = 3 ↔ x = 2 := by sorry

end preimage_of_three_l3177_317739


namespace no_real_roots_l3177_317728

def polynomial (x p : ℝ) : ℝ := x^4 + 4*p*x^3 + 6*x^2 + 4*p*x + 1

theorem no_real_roots (p : ℝ) : 
  (∀ x : ℝ, polynomial x p ≠ 0) ↔ p > -Real.sqrt 5 / 2 ∧ p < Real.sqrt 5 / 2 :=
sorry

end no_real_roots_l3177_317728


namespace discount_rate_pony_jeans_discount_rate_pony_jeans_proof_l3177_317795

theorem discount_rate_pony_jeans : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun (fox_price pony_price total_savings discount_sum : ℝ) =>
    let fox_pairs : ℝ := 3
    let pony_pairs : ℝ := 2
    let total_pairs : ℝ := fox_pairs + pony_pairs
    fox_price = 15 ∧ 
    pony_price = 18 ∧ 
    total_savings = 9 ∧ 
    discount_sum = 22 ∧
    ∃ (fox_discount pony_discount : ℝ),
      fox_discount + pony_discount = discount_sum ∧
      fox_pairs * (fox_discount / 100 * fox_price) + 
        pony_pairs * (pony_discount / 100 * pony_price) = total_savings ∧
      pony_discount = 10

theorem discount_rate_pony_jeans_proof 
  (fox_price pony_price total_savings discount_sum : ℝ) :
  discount_rate_pony_jeans fox_price pony_price total_savings discount_sum :=
by sorry

end discount_rate_pony_jeans_discount_rate_pony_jeans_proof_l3177_317795


namespace isosceles_triangle_dimensions_l3177_317753

/-- An isosceles triangle with base b and leg length l -/
structure IsoscelesTriangle where
  b : ℝ
  l : ℝ
  h : ℝ
  isPositive : 0 < b ∧ 0 < l ∧ 0 < h
  isIsosceles : l = b - 1
  areaRelation : (1/2) * b * h = (1/3) * b^2

/-- Theorem about the dimensions of a specific isosceles triangle -/
theorem isosceles_triangle_dimensions (t : IsoscelesTriangle) :
  t.b = 6 ∧ t.l = 5 ∧ t.h = 4 := by
  sorry

end isosceles_triangle_dimensions_l3177_317753


namespace correct_system_of_equations_l3177_317798

theorem correct_system_of_equations :
  ∀ (x y : ℕ),
  (x + y = 12) →
  (4 * x + 3 * y = 40) →
  (∀ (a b : ℕ), (a + b = 12 ∧ 4 * a + 3 * b = 40) → (a = x ∧ b = y)) :=
by sorry

end correct_system_of_equations_l3177_317798


namespace min_price_reduction_l3177_317714

theorem min_price_reduction (price_2004 : ℝ) (h1 : price_2004 > 0) : 
  let price_2005 := price_2004 * (1 - 0.15)
  let min_reduction := (price_2005 - price_2004 * 0.75) / price_2005 * 100
  ∀ ε > 0, ∃ δ > 0, 
    abs (min_reduction - 11.8) < δ ∧ 
    price_2004 * (1 - 0.15) * (1 - (min_reduction + ε) / 100) < price_2004 * 0.75 ∧
    price_2004 * (1 - 0.15) * (1 - (min_reduction - ε) / 100) > price_2004 * 0.75 :=
by sorry

end min_price_reduction_l3177_317714


namespace sufficient_necessary_condition_l3177_317731

theorem sufficient_necessary_condition (a : ℝ) :
  (∀ x : ℝ, x > 0 → x + 1/x > a) ↔ a < 2 := by sorry

end sufficient_necessary_condition_l3177_317731


namespace common_internal_tangent_length_l3177_317764

/-- Given two circles with centers 50 inches apart, with radii 7 inches and 10 inches respectively,
    the length of their common internal tangent is equal to the square root of the difference between
    the square of the distance between their centers and the square of the sum of their radii. -/
theorem common_internal_tangent_length 
  (center_distance : ℝ) 
  (radius1 : ℝ) 
  (radius2 : ℝ) 
  (h1 : center_distance = 50) 
  (h2 : radius1 = 7) 
  (h3 : radius2 = 10) : 
  ∃ (tangent_length : ℝ), tangent_length = Real.sqrt (center_distance^2 - (radius1 + radius2)^2) :=
by sorry

end common_internal_tangent_length_l3177_317764


namespace hyperbola_condition_l3177_317794

/-- The equation represents a hyperbola -/
def is_hyperbola (k : ℝ) : Prop := (3 - k) * (k - 1) < 0

/-- The condition k > 3 -/
def condition (k : ℝ) : Prop := k > 3

theorem hyperbola_condition (k : ℝ) :
  (condition k → is_hyperbola k) ∧ ¬(is_hyperbola k → condition k) :=
by sorry

end hyperbola_condition_l3177_317794


namespace triangle_area_fraction_l3177_317712

/-- The area of a triangle given the coordinates of its vertices -/
def triangleArea (x1 y1 x2 y2 x3 y3 : ℚ) : ℚ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- The theorem stating that the area of the given triangle divided by the area of the grid equals 5/28 -/
theorem triangle_area_fraction :
  let a := (2, 2)
  let b := (6, 3)
  let c := (3, 6)
  let gridArea := 7 * 6
  (triangleArea a.1 a.2 b.1 b.2 c.1 c.2) / gridArea = 5 / 28 := by
  sorry


end triangle_area_fraction_l3177_317712


namespace youtube_video_length_l3177_317720

theorem youtube_video_length (x : ℝ) 
  (h1 : 6 * x + 6 * (x / 2) = 900) : x = 100 := by
  sorry

end youtube_video_length_l3177_317720


namespace expression_decrease_decrease_percentage_l3177_317770

theorem expression_decrease (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) : 
  (2 * ((1/2 * x)^2) * (1/2 * y)) / (2 * x^2 * y) = 1/4 :=
sorry

theorem decrease_percentage : (1 - 1/4) * 100 = 87.5 :=
sorry

end expression_decrease_decrease_percentage_l3177_317770


namespace student_weighted_average_l3177_317778

def weighted_average (courses1 courses2 courses3 : ℕ) (grade1 grade2 grade3 : ℚ) : ℚ :=
  (courses1 * grade1 + courses2 * grade2 + courses3 * grade3) / (courses1 + courses2 + courses3)

theorem student_weighted_average :
  let courses1 := 8
  let courses2 := 6
  let courses3 := 10
  let grade1 := 92
  let grade2 := 88
  let grade3 := 76
  abs (weighted_average courses1 courses2 courses3 grade1 grade2 grade3 - 84.3) < 0.05 := by
  sorry

end student_weighted_average_l3177_317778


namespace geometric_sequence_problem_l3177_317777

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The statement to be proved -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 3)^2 + 7*(a 3) + 9 = 0 →
  (a 7)^2 + 7*(a 7) + 9 = 0 →
  a 5 = -3 := by
  sorry

end geometric_sequence_problem_l3177_317777


namespace interest_rate_problem_l3177_317725

/-- The interest rate problem --/
theorem interest_rate_problem (total_investment : ℝ) (total_interest : ℝ) 
  (amount_at_r : ℝ) (rate_known : ℝ) :
  total_investment = 6000 →
  total_interest = 624 →
  amount_at_r = 1800 →
  rate_known = 0.11 →
  ∃ (r : ℝ), 
    amount_at_r * r + (total_investment - amount_at_r) * rate_known = total_interest ∧
    r = 0.09 := by
  sorry

end interest_rate_problem_l3177_317725


namespace unique_positive_integer_cube_less_than_triple_l3177_317779

theorem unique_positive_integer_cube_less_than_triple :
  ∃! (n : ℕ), n > 0 ∧ n^3 < 3*n :=
by
  sorry

end unique_positive_integer_cube_less_than_triple_l3177_317779


namespace parallel_vectors_imply_x_equals_four_l3177_317717

/-- Given vectors a and b in ℝ², prove that if a + 3b is parallel to a - b, then the x-coordinate of b is 4. -/
theorem parallel_vectors_imply_x_equals_four (a b : ℝ × ℝ) 
  (ha : a = (2, 1)) 
  (hb : b.2 = 2) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a + 3 • b = k • (a - b)) : 
  b.1 = 4 := by
  sorry

end parallel_vectors_imply_x_equals_four_l3177_317717


namespace sufficient_not_necessary_l3177_317758

theorem sufficient_not_necessary (x : ℝ) (h : x ≠ 0) :
  (∀ x > 1, x + 1/x > 2) ∧
  (∃ x, 0 < x ∧ x < 1 ∧ x + 1/x > 2) :=
by sorry

end sufficient_not_necessary_l3177_317758


namespace trig_expression_value_l3177_317774

theorem trig_expression_value (α : Real) (h : Real.tan α = 3) :
  (6 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 8 / 7 := by
  sorry

end trig_expression_value_l3177_317774


namespace simplify_expression_l3177_317767

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 3*(a + b)) :
  a/b + b/a - 3/(a*b) = 1 := by
sorry

end simplify_expression_l3177_317767


namespace bank_queue_theorem_l3177_317784

/-- Represents a bank queue with simple and long operations -/
structure BankQueue where
  total_people : Nat
  simple_ops : Nat
  long_ops : Nat
  simple_time : Nat
  long_time : Nat

/-- Calculates the minimum wasted person-minutes -/
def min_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the maximum wasted person-minutes -/
def max_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the expected wasted person-minutes assuming random order -/
def expected_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Main theorem about the bank queue problem -/
theorem bank_queue_theorem (q : BankQueue) 
  (h1 : q.total_people = 8)
  (h2 : q.simple_ops = 5)
  (h3 : q.long_ops = 3)
  (h4 : q.simple_time = 1)
  (h5 : q.long_time = 5) :
  min_wasted_time q = 40 ∧ 
  max_wasted_time q = 100 ∧ 
  expected_wasted_time q = 84 := by
  sorry

end bank_queue_theorem_l3177_317784


namespace value_in_numerator_l3177_317715

theorem value_in_numerator (N V : ℤ) : 
  N = 1280 → (N + 720) / 125 = V / 462 → V = 7392 := by sorry

end value_in_numerator_l3177_317715


namespace skittle_groups_l3177_317735

/-- The number of groups formed when dividing Skittles into equal-sized groups -/
def number_of_groups (total_skittles : ℕ) (skittles_per_group : ℕ) : ℕ :=
  total_skittles / skittles_per_group

/-- Theorem stating that dividing 5929 Skittles into groups of 77 results in 77 groups -/
theorem skittle_groups : number_of_groups 5929 77 = 77 := by
  sorry

end skittle_groups_l3177_317735


namespace sin_ratio_comparison_l3177_317754

theorem sin_ratio_comparison : 
  (Real.sin (2014 * π / 180)) / (Real.sin (2015 * π / 180)) < 
  (Real.sin (2016 * π / 180)) / (Real.sin (2017 * π / 180)) := by
  sorry

end sin_ratio_comparison_l3177_317754


namespace jakes_third_test_score_l3177_317790

/-- Given Jake's test scores, prove he scored 65 in the third test -/
theorem jakes_third_test_score :
  -- Define the number of tests
  let num_tests : ℕ := 4
  -- Define the average score
  let average_score : ℚ := 75
  -- Define the score of the first test
  let first_test_score : ℕ := 80
  -- Define the score difference between second and first tests
  let second_test_difference : ℕ := 10
  -- Define the condition that third and fourth test scores are equal
  ∀ (third_test_score fourth_test_score : ℕ),
    -- Total score equals average multiplied by number of tests
    (first_test_score + (first_test_score + second_test_difference) + third_test_score + fourth_test_score : ℚ) = num_tests * average_score →
    -- Third and fourth test scores are equal
    third_test_score = fourth_test_score →
    -- Prove that the third test score is 65
    third_test_score = 65 := by
  sorry

end jakes_third_test_score_l3177_317790


namespace soda_price_ratio_l3177_317773

theorem soda_price_ratio (v : ℝ) (p : ℝ) (hv : v > 0) (hp : p > 0) :
  let x_volume := 1.3 * v
  let x_price := 0.85 * p
  let x_unit_price := x_price / x_volume
  let y_unit_price := p / v
  x_unit_price / y_unit_price = 17 / 26 := by
sorry

end soda_price_ratio_l3177_317773


namespace triangle_area_is_two_l3177_317737

/-- The area of the triangle bounded by the y-axis and two lines -/
def triangle_area : ℝ := 2

/-- The first line equation: y - 2x = 1 -/
def line1 (x y : ℝ) : Prop := y - 2 * x = 1

/-- The second line equation: 4y + x = 16 -/
def line2 (x y : ℝ) : Prop := 4 * y + x = 16

/-- The theorem stating that the area of the triangle is 2 -/
theorem triangle_area_is_two :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ = 0 ∧ line1 x₁ y₁ ∧
    x₂ = 0 ∧ line2 x₂ y₂ ∧
    triangle_area = 2 := by
  sorry

end triangle_area_is_two_l3177_317737


namespace ellipse_foci_l3177_317709

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 25 = 1

-- Define the foci coordinates
def foci : Set (ℝ × ℝ) :=
  {(0, 3), (0, -3)}

-- Theorem statement
theorem ellipse_foci :
  ∀ (f : ℝ × ℝ), f ∈ foci ↔
    (∃ (x y : ℝ), ellipse_equation x y ∧
      (x - f.1)^2 + (y - f.2)^2 +
      (x + f.1)^2 + (y + f.2)^2 = 4 * (5^2 + 4^2)) :=
by sorry

end ellipse_foci_l3177_317709


namespace sequence_increasing_l3177_317743

def a (n : ℕ+) : ℚ := (2 * n) / (2 * n + 1)

theorem sequence_increasing (n : ℕ+) : a n < a (n + 1) := by
  sorry

end sequence_increasing_l3177_317743


namespace triangle_is_equilateral_l3177_317752

/-- A triangle with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Predicate for angles forming an arithmetic progression -/
def angles_in_arithmetic_progression (t : Triangle) : Prop :=
  t.A + t.C = 2 * t.B

/-- Predicate for sides forming a geometric progression -/
def sides_in_geometric_progression (t : Triangle) : Prop :=
  t.b^2 = t.a * t.c

/-- Theorem stating that a triangle with angles in arithmetic progression
    and sides in geometric progression is equilateral -/
theorem triangle_is_equilateral (t : Triangle)
  (h1 : angles_in_arithmetic_progression t)
  (h2 : sides_in_geometric_progression t) :
  t.A = 60 ∧ t.B = 60 ∧ t.C = 60 := by
  sorry

end triangle_is_equilateral_l3177_317752


namespace remaining_balloons_l3177_317763

def initial_balloons : ℕ := 709
def given_away : ℕ := 221

theorem remaining_balloons : initial_balloons - given_away = 488 := by
  sorry

end remaining_balloons_l3177_317763


namespace fraction_equality_l3177_317782

theorem fraction_equality (q r s u : ℚ) 
  (h1 : q / r = 8)
  (h2 : s / r = 4)
  (h3 : s / u = 1 / 3) :
  u / q = 3 / 2 := by
sorry

end fraction_equality_l3177_317782


namespace correct_seating_arrangements_l3177_317781

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·+1) 1

/-- The number of people to be seated. -/
def totalPeople : ℕ := 8

/-- The number of ways to seat the people under the given conditions. -/
def seatingArrangements : ℕ :=
  factorial totalPeople - 2 * (factorial (totalPeople - 1) * factorial 2)

theorem correct_seating_arrangements :
  seatingArrangements = 20160 := by sorry

end correct_seating_arrangements_l3177_317781


namespace no_integer_solution_l3177_317759

theorem no_integer_solution : ∀ x y : ℤ, x^2 + 5 ≠ y^3 := by
  sorry

end no_integer_solution_l3177_317759


namespace hot_dogs_remainder_l3177_317702

theorem hot_dogs_remainder : 35867413 % 6 = 1 := by
  sorry

end hot_dogs_remainder_l3177_317702


namespace equation_solution_l3177_317738

theorem equation_solution (x : ℝ) : 
  (1 - 2 * Real.sin (x / 2) * Real.cos (x / 2) = 
   (Real.sin (x / 2) - Real.cos (x / 2)) / Real.cos (x / 2)) ↔ 
  (∃ k : ℤ, x = π / 2 + 2 * k * π) :=
by sorry

end equation_solution_l3177_317738


namespace arithmetic_sequence_sum_l3177_317775

/-- Given an arithmetic sequence {a_n} with the specified conditions, 
    prove that a₅ + a₈ + a₁₁ = 15 -/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) 
  (h1 : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- arithmetic sequence
  (h2 : a 1 + a 4 + a 7 = 39)
  (h3 : a 2 + a 5 + a 8 = 33) :
  a 5 + a 8 + a 11 = 15 := by
  sorry

end arithmetic_sequence_sum_l3177_317775


namespace probability_log_equals_one_l3177_317772

def set_A : Finset ℕ := {1, 2, 3, 4, 5, 6}
def set_B : Finset ℕ := {1, 2, 3}

def favorable_outcomes : Finset (ℕ × ℕ) := 
  {(2, 1), (4, 2), (6, 3)}

def total_outcomes : ℕ := Finset.card set_A * Finset.card set_B

theorem probability_log_equals_one :
  (Finset.card favorable_outcomes : ℚ) / total_outcomes = 1 / 6 := by
  sorry


end probability_log_equals_one_l3177_317772


namespace prob_two_non_defective_pens_l3177_317757

/-- The probability of selecting two non-defective pens from a box of 9 pens with 3 defective pens -/
theorem prob_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 9)
  (h2 : defective_pens = 3)
  (h3 : defective_pens < total_pens) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 5 / 12 := by
  sorry

end prob_two_non_defective_pens_l3177_317757


namespace area_ratio_is_one_fourth_l3177_317707

/-- A square with vertices A, B, C, D -/
structure Square where
  side_length : ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- A particle moving along the edges of a square -/
structure Particle where
  position : ℝ → ℝ × ℝ  -- position as a function of time
  speed : ℝ

/-- The region enclosed by the path of the midpoint -/
def enclosed_region (p1 p2 : Particle) : Set (ℝ × ℝ) := sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating the ratio of areas -/
theorem area_ratio_is_one_fourth (sq : Square) (p1 p2 : Particle) :
  sq.A = (0, 0) ∧ 
  sq.B = (sq.side_length, 0) ∧ 
  sq.C = (sq.side_length, sq.side_length) ∧ 
  sq.D = (0, sq.side_length) ∧
  p1.position 0 = sq.A ∧
  p2.position 0 = ((sq.C.1 + sq.D.1) / 2, sq.C.2) ∧
  p1.speed = p2.speed →
  area (enclosed_region p1 p2) / area {p | p.1 ∈ Set.Icc 0 sq.side_length ∧ p.2 ∈ Set.Icc 0 sq.side_length} = 1 / 4 := by
  sorry

end area_ratio_is_one_fourth_l3177_317707


namespace final_lives_correct_l3177_317741

/-- Given a player's initial lives, lost lives, and gained lives (before bonus),
    calculate the final number of lives after a secret bonus is applied. -/
def final_lives (initial_lives lost_lives gained_lives : ℕ) : ℕ :=
  initial_lives - lost_lives + 3 * gained_lives

/-- Theorem stating that the final_lives function correctly calculates
    the number of lives after the secret bonus is applied. -/
theorem final_lives_correct (X Y Z : ℕ) (h : Y ≤ X) :
  final_lives X Y Z = X - Y + 3 * Z :=
by sorry

end final_lives_correct_l3177_317741


namespace gcd_difference_perfect_square_l3177_317761

theorem gcd_difference_perfect_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, Nat.gcd x (Nat.gcd y z) * (y - x) = k ^ 2 :=
sorry

end gcd_difference_perfect_square_l3177_317761


namespace parabola_properties_l3177_317755

-- Define the parabola and its properties
def parabola (a b c m : ℝ) : Prop :=
  a ≠ 0 ∧ a < 0 ∧ -2 < m ∧ m < -1 ∧
  a * 1^2 + b * 1 + c = 0 ∧
  a * m^2 + b * m + c = 0

-- State the theorem
theorem parabola_properties (a b c m : ℝ) (h : parabola a b c m) :
  a * b * c > 0 ∧ a - b + c > 0 ∧ a * (m + 1) - b + c > 0 := by
  sorry

end parabola_properties_l3177_317755


namespace complement_of_P_l3177_317745

def P : Set ℝ := {x | |x + 3| + |x + 6| = 3}

theorem complement_of_P : 
  {x : ℝ | x < -6 ∨ x > -3} = (Set.univ : Set ℝ) \ P := by sorry

end complement_of_P_l3177_317745


namespace sum_of_reciprocal_pairs_of_roots_l3177_317736

/-- Given a quintic polynomial x^5 + 10x^4 + 20x^3 + 15x^2 + 6x + 3, 
    this theorem states that the sum of reciprocals of products of pairs of its roots is 20/3 -/
theorem sum_of_reciprocal_pairs_of_roots (p q r s t : ℂ) : 
  p^5 + 10*p^4 + 20*p^3 + 15*p^2 + 6*p + 3 = 0 →
  q^5 + 10*q^4 + 20*q^3 + 15*q^2 + 6*q + 3 = 0 →
  r^5 + 10*r^4 + 20*r^3 + 15*r^2 + 6*r + 3 = 0 →
  s^5 + 10*s^4 + 20*s^3 + 15*s^2 + 6*s + 3 = 0 →
  t^5 + 10*t^4 + 20*t^3 + 15*t^2 + 6*t + 3 = 0 →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(p*t) + 1/(q*r) + 1/(q*s) + 1/(q*t) + 1/(r*s) + 1/(r*t) + 1/(s*t) = 20/3 := by
  sorry

end sum_of_reciprocal_pairs_of_roots_l3177_317736


namespace gcf_of_75_and_100_l3177_317750

theorem gcf_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end gcf_of_75_and_100_l3177_317750


namespace complex_product_simplification_l3177_317706

theorem complex_product_simplification (a b x y : ℝ) : 
  (a * x + Complex.I * b * y) * (a * x - Complex.I * b * y) = a^2 * x^2 - b^2 * y^2 := by
  sorry

end complex_product_simplification_l3177_317706


namespace hotel_weekly_loss_l3177_317703

def weekly_profit_loss (operations_expenses taxes employee_salaries : ℚ) : ℚ :=
  let meetings_income := (5 / 8) * operations_expenses
  let events_income := (3 / 10) * operations_expenses
  let rooms_income := (11 / 20) * operations_expenses
  let total_income := meetings_income + events_income + rooms_income
  let total_expenses := operations_expenses + taxes + employee_salaries
  total_income - total_expenses

theorem hotel_weekly_loss :
  weekly_profit_loss 5000 1200 2500 = -1325 :=
by sorry

end hotel_weekly_loss_l3177_317703


namespace train_length_l3177_317748

/-- Calculates the length of a train given its speed and the time it takes to pass through a tunnel of known length. -/
theorem train_length (train_speed : ℝ) (tunnel_length : ℝ) (time_to_pass : ℝ) : 
  train_speed = 54 * 1000 / 3600 →
  tunnel_length = 1200 →
  time_to_pass = 100 →
  (train_speed * time_to_pass) - tunnel_length = 300 := by
sorry

end train_length_l3177_317748


namespace triangle_division_2005_l3177_317723

theorem triangle_division_2005 : ∃ n : ℕ, n^2 + (2005 - n^2)^2 = 2005 := by
  sorry

end triangle_division_2005_l3177_317723


namespace complete_square_sum_l3177_317749

theorem complete_square_sum (a b c : ℤ) (h1 : a > 0) 
  (h2 : ∀ x : ℝ, 49 * x^2 + 56 * x - 64 = 0 ↔ (a * x + b)^2 = c) : 
  a + b + c = 91 := by
sorry

end complete_square_sum_l3177_317749


namespace tangent_sum_simplification_l3177_317769

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / Real.sin (10 * π / 180) =
  1 / (2 * Real.sin (10 * π / 180) ^ 2 * Real.cos (20 * π / 180)) + 4 / (Real.sqrt 3 * Real.sin (10 * π / 180)) := by
  sorry

end tangent_sum_simplification_l3177_317769


namespace expected_subtree_size_ten_vertices_l3177_317789

-- Define a type for rooted trees
structure RootedTree where
  vertices : Nat
  root : Nat

-- Define a function to represent the expected subtree size
def expectedSubtreeSize (t : RootedTree) : ℚ :=
  sorry

-- Theorem statement
theorem expected_subtree_size_ten_vertices :
  ∀ t : RootedTree,
  t.vertices = 10 →
  expectedSubtreeSize t = 7381 / 2520 :=
by sorry

end expected_subtree_size_ten_vertices_l3177_317789


namespace isosceles_triangle_circle_properties_main_theorem_l3177_317729

/-- An isosceles triangle inscribed in a circle -/
structure IsoscelesTriangleInCircle where
  /-- Length of the two equal sides of the isosceles triangle -/
  side : ℝ
  /-- Length of the base of the isosceles triangle -/
  base : ℝ
  /-- Radius of the circumscribed circle -/
  radius : ℝ

/-- Theorem about the radius and area of a circle circumscribing an isosceles triangle -/
theorem isosceles_triangle_circle_properties (t : IsoscelesTriangleInCircle)
  (h_side : t.side = 4)
  (h_base : t.base = 3) :
  t.radius = 3.5 ∧ t.radius^2 * π = 12.25 * π := by
  sorry

/-- Main theorem combining the properties -/
theorem main_theorem :
  ∃ t : IsoscelesTriangleInCircle,
    t.side = 4 ∧
    t.base = 3 ∧
    t.radius = 3.5 ∧
    t.radius^2 * π = 12.25 * π := by
  sorry

end isosceles_triangle_circle_properties_main_theorem_l3177_317729


namespace no_positive_integer_solutions_l3177_317708

theorem no_positive_integer_solutions :
  ¬ ∃ (x₁ x₂ : ℕ), 903 * x₁ + 731 * x₂ = 1106 := by
sorry

end no_positive_integer_solutions_l3177_317708


namespace no_solution_x5_y2_plus4_l3177_317785

theorem no_solution_x5_y2_plus4 : ¬ ∃ (x y : ℕ), x ≥ 1 ∧ y ≥ 1 ∧ x^5 = y^2 + 4 := by
  sorry

end no_solution_x5_y2_plus4_l3177_317785


namespace hall_covering_cost_l3177_317732

/-- Calculates the total expenditure for covering the interior of a rectangular hall with mat -/
def total_expenditure (length width height cost_per_sqm : ℝ) : ℝ :=
  let floor_area := length * width
  let wall_area := 2 * (length * height + width * height)
  let total_area := floor_area + wall_area
  total_area * cost_per_sqm

/-- Proves that the total expenditure for the given hall dimensions and mat cost is Rs. 39,000 -/
theorem hall_covering_cost : 
  total_expenditure 20 15 5 60 = 39000 := by
  sorry

end hall_covering_cost_l3177_317732


namespace complex_fraction_simplification_l3177_317733

theorem complex_fraction_simplification (z : ℂ) (h : z = 1 - 2*I) : 
  (z^2 + 3) / (z - 1) = 2 := by sorry

end complex_fraction_simplification_l3177_317733
