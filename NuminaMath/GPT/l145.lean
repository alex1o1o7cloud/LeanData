import Mathlib

namespace b_arithmetic_sequence_max_S_n_l145_145324

variables {a : ℕ → ℝ} {b : ℕ → ℝ} {S : ℕ → ℝ}

-- Conditions
noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m, a m ≠ 0 → a n = a (n + 1) * a (m-1) / (a m)

axiom a_pos_terms : ∀ n, 0 < a n
axiom a11_eight : a 11 = 8
axiom b_log : ∀ n, b n = Real.log (a n) / Real.log 2
axiom b4_seventeen : b 4 = 17

-- Question I: Prove b_n is an arithmetic sequence with common difference -2
theorem b_arithmetic_sequence (d : ℝ) (h_d : d = (-2)) :
  ∃ d, ∀ n, b (n + 1) - b n = d :=
sorry

-- Question II: Find the maximum value of S_n
theorem max_S_n : ∃ n, S n = 144 :=
sorry

end b_arithmetic_sequence_max_S_n_l145_145324


namespace students_apply_colleges_l145_145254

    -- Define that there are 5 students
    def students : Nat := 5

    -- Each student has 3 choices of colleges
    def choices_per_student : Nat := 3

    -- The number of different ways the students can apply
    def number_of_ways : Nat := choices_per_student ^ students

    theorem students_apply_colleges : number_of_ways = 3 ^ 5 :=
    by
        -- Proof will be done here
        sorry
    
end students_apply_colleges_l145_145254


namespace first_duck_fraction_l145_145756

-- Definitions based on the conditions
variable (total_bread : ℕ) (left_bread : ℕ) (second_duck_bread : ℕ) (third_duck_bread : ℕ)

-- Given values
def given_values : Prop :=
  total_bread = 100 ∧ left_bread = 30 ∧ second_duck_bread = 13 ∧ third_duck_bread = 7

-- Proof statement
theorem first_duck_fraction (h : given_values total_bread left_bread second_duck_bread third_duck_bread) :
  (total_bread - left_bread) - (second_duck_bread + third_duck_bread) = 1/2 * total_bread := by 
  sorry

end first_duck_fraction_l145_145756


namespace max_cylinder_volume_in_cone_l145_145963

theorem max_cylinder_volume_in_cone :
  ∃ x, (0 < x ∧ x < 1) ∧ ∀ y, (0 < y ∧ y < 1 → y ≠ x → ((π * (-2 * y^3 + 2 * y^2)) ≤ (π * (-2 * x^3 + 2 * x^2)))) ∧ 
  (π * (-2 * x^3 + 2 * x^2) = 8 * π / 27) := sorry

end max_cylinder_volume_in_cone_l145_145963


namespace pieces_per_pan_of_brownies_l145_145312

theorem pieces_per_pan_of_brownies (total_guests guests_ala_mode additional_guests total_scoops_per_tub total_tubs_eaten total_pans guests_per_pan second_pan_percentage consumed_pans : ℝ)
    (h1 : total_guests = guests_ala_mode + additional_guests)
    (h2 : total_scoops_per_tub * total_tubs_eaten = guests_ala_mode * 2)
    (h3 : consumed_pans = 1 + second_pan_percentage)
    (h4 : second_pan_percentage = 0.75)
    (h5 : total_guests = guests_per_pan * consumed_pans)
    (h6 : guests_per_pan = 28)
    : total_guests / consumed_pans = 16 :=
by
  have h7 : total_scoops_per_tub * total_tubs_eaten = 48 := by sorry
  have h8 : guests_ala_mode = 24 := by sorry
  have h9 : total_guests = 28 := by sorry
  have h10 : consumed_pans = 1.75 := by sorry
  have h11 : guests_per_pan = 28 := by sorry
  sorry


end pieces_per_pan_of_brownies_l145_145312


namespace square_of_binomial_b_value_l145_145414

theorem square_of_binomial_b_value (b : ℤ) (h : ∃ c : ℤ, 16 * (x : ℤ) * x + 40 * x + b = (4 * x + c) ^ 2) : b = 25 :=
sorry

end square_of_binomial_b_value_l145_145414


namespace compute_expression_l145_145533

theorem compute_expression : 3034 - (1002 / 200.4) = 3029 := by
  sorry

end compute_expression_l145_145533


namespace smallest_distance_l145_145939

noncomputable def a : Complex := 2 + 4 * Complex.I
noncomputable def b : Complex := 5 + 2 * Complex.I

theorem smallest_distance 
  (z w : Complex) 
  (hz : Complex.abs (z - a) = 2) 
  (hw : Complex.abs (w - b) = 4) : 
  Complex.abs (z - w) ≥ 6 - Real.sqrt 13 :=
sorry

end smallest_distance_l145_145939


namespace range_of_b_l145_145147

variable (a b c : ℝ)

theorem range_of_b (h1 : a + b + c = 9) (h2 : a * b + b * c + c * a = 24) : 1 ≤ b ∧ b ≤ 5 :=
by
  sorry

end range_of_b_l145_145147


namespace number_of_spiders_l145_145594

theorem number_of_spiders (total_legs birds dogs snakes : ℕ) (legs_per_bird legs_per_dog legs_per_snake legs_per_spider : ℕ) (h1 : total_legs = 34)
  (h2 : birds = 3) (h3 : dogs = 5) (h4 : snakes = 4) (h5 : legs_per_bird = 2) (h6 : legs_per_dog = 4)
  (h7 : legs_per_snake = 0) (h8 : legs_per_spider = 8) : 
  (total_legs - (birds * legs_per_bird + dogs * legs_per_dog + snakes * legs_per_snake)) / legs_per_spider = 1 :=
by sorry

end number_of_spiders_l145_145594


namespace column_1000_is_B_l145_145234

-- Definition of the column pattern
def columnPattern : List String := ["B", "C", "D", "E", "F", "E", "D", "C", "B", "A"]

-- Function to determine the column for a given integer
def columnOf (n : Nat) : String :=
  columnPattern.get! ((n - 2) % 10)

-- The theorem we want to prove
theorem column_1000_is_B : columnOf 1000 = "B" :=
by
  sorry

end column_1000_is_B_l145_145234


namespace largest_unique_k_l145_145296

theorem largest_unique_k (n : ℕ) :
  (∀ k : ℤ, (8:ℚ)/15 < n / (n + k) ∧ n / (n + k) < 7/13 → False) ∧
  (∃ k : ℤ, (8:ℚ)/15 < n / (n + k) ∧ n / (n + k) < 7/13) → n = 112 :=
by sorry

end largest_unique_k_l145_145296


namespace num_five_digit_ints_l145_145706

open Nat

theorem num_five_digit_ints : 
  let num_ways := (factorial 5) / (factorial 3 * factorial 2)
  num_ways = 10 :=
by
  let num_ways := (factorial 5) / (factorial 3 * factorial 2)
  sorry

end num_five_digit_ints_l145_145706


namespace combine_material_points_l145_145129

variables {K K₁ K₂ : Type} {m m₁ m₂ : ℝ}

-- Assume some properties and operations for type K
noncomputable def add_material_points (K₁ K₂ : K × ℝ) : K × ℝ :=
(K₁.1, K₁.2 + K₂.2)

theorem combine_material_points (K₁ K₂ : K × ℝ) :
  (add_material_points K₁ K₂) = (K₁.1, K₁.2 + K₂.2) :=
sorry

end combine_material_points_l145_145129


namespace vendor_pepsi_volume_l145_145918

theorem vendor_pepsi_volume 
    (liters_maaza : ℕ)
    (liters_sprite : ℕ)
    (num_cans : ℕ)
    (h1 : liters_maaza = 40)
    (h2 : liters_sprite = 368)
    (h3 : num_cans = 69)
    (volume_pepsi : ℕ)
    (total_volume : ℕ)
    (h4 : total_volume = liters_maaza + liters_sprite + volume_pepsi)
    (h5 : total_volume = num_cans * n)
    (h6 : 408 % num_cans = 0) :
  volume_pepsi = 75 :=
sorry

end vendor_pepsi_volume_l145_145918


namespace find_k_l145_145892

-- Define vector a and vector b
def vec_a : (ℝ × ℝ) := (1, 1)
def vec_b : (ℝ × ℝ) := (-3, 1)

-- Define the expression for k * vec_a - vec_b
def k_vec_a_minus_vec_b (k : ℝ) : ℝ × ℝ :=
  (k * vec_a.1 - vec_b.1, k * vec_a.2 - vec_b.2)

-- Define the dot product condition for perpendicular vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- The theorem to be proved: k = -1 is the value that makes the dot product zero
theorem find_k : ∃ k : ℝ, dot_product (k_vec_a_minus_vec_b k) vec_a = 0 :=
by
  use -1
  sorry

end find_k_l145_145892


namespace last_three_digits_7_pow_103_l145_145702

theorem last_three_digits_7_pow_103 : (7 ^ 103) % 1000 = 60 := sorry

end last_three_digits_7_pow_103_l145_145702


namespace find_divisor_l145_145683

theorem find_divisor (d q r : ℕ) (h1 : d = 265) (h2 : q = 12) (h3 : r = 1) :
  ∃ x : ℕ, d = (x * q) + r ∧ x = 22 :=
by {
  sorry
}

end find_divisor_l145_145683


namespace find_a_l145_145590

noncomputable def f (a x : ℝ) := a * Real.exp x + 2 * x^2

noncomputable def f' (a x : ℝ) := a * Real.exp x + 4 * x

theorem find_a (a : ℝ) (h : f' a 0 = 2) : a = 2 :=
by
  unfold f' at h
  simp at h
  exact h

end find_a_l145_145590


namespace product_of_possible_values_l145_145899

theorem product_of_possible_values :
  (∀ x : ℝ, abs (18 / x + 4) = 3 → x = -18 ∨ x = -18 / 7) →
  (∀ x1 x2 : ℝ, x1 = -18 → x2 = -18 / 7 → x1 * x2 = 324 / 7) :=
by
  intros h x1 x2 hx1 hx2
  rw [hx1, hx2]
  norm_num

end product_of_possible_values_l145_145899


namespace factor_tree_value_l145_145521

theorem factor_tree_value :
  ∀ (X Y Z F G : ℕ),
  X = Y * Z → 
  Y = 7 * F → 
  F = 2 * 5 → 
  Z = 11 * G → 
  G = 7 * 3 → 
  X = 16170 := 
by
  intros X Y Z F G
  sorry

end factor_tree_value_l145_145521


namespace linear_increase_y_l145_145279

-- Progressively increase x and track y

theorem linear_increase_y (Δx Δy : ℝ) (x_increase : Δx = 4) (y_increase : Δy = 10) :
  12 * (Δy / Δx) = 30 := by
  sorry

end linear_increase_y_l145_145279


namespace box_weight_no_apples_l145_145303

variable (initialWeight : ℕ) (halfWeight : ℕ) (totalWeight : ℕ)
variable (boxWeight : ℕ)

-- Given conditions
axiom initialWeight_def : initialWeight = 9
axiom halfWeight_def : halfWeight = 5
axiom appleWeight_consistent : ∃ w : ℕ, ∀ n : ℕ, n * w = totalWeight

-- Question: How many kilograms does the empty box weigh?
theorem box_weight_no_apples : (initialWeight - totalWeight) = boxWeight :=
by
  -- The proof steps are omitted as indicated by the 'sorry' placeholder.
  sorry

end box_weight_no_apples_l145_145303


namespace john_gets_30_cans_l145_145183

def normal_price : ℝ := 0.60
def total_paid : ℝ := 9.00

theorem john_gets_30_cans :
  (total_paid / normal_price) * 2 = 30 :=
by
  sorry

end john_gets_30_cans_l145_145183


namespace bottles_from_B_l145_145047

-- Definitions for the bottles from each shop and the total number of bottles Don can buy
def bottles_from_A : Nat := 150
def bottles_from_C : Nat := 220
def total_bottles : Nat := 550

-- Lean statement to prove that the number of bottles Don buys from Shop B is 180
theorem bottles_from_B :
  total_bottles - (bottles_from_A + bottles_from_C) = 180 := 
by
  sorry

end bottles_from_B_l145_145047


namespace num_bags_of_cookies_l145_145387

theorem num_bags_of_cookies (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 703) (h2 : cookies_per_bag = 19) : total_cookies / cookies_per_bag = 37 :=
by
  sorry

end num_bags_of_cookies_l145_145387


namespace total_rooms_booked_l145_145136

variable (S D : ℕ)

theorem total_rooms_booked (h1 : 35 * S + 60 * D = 14000) (h2 : D = 196) : S + D = 260 :=
by
  sorry

end total_rooms_booked_l145_145136


namespace number_of_siblings_l145_145135

-- Definitions for the given conditions
def total_height : ℕ := 330
def sibling1_height : ℕ := 66
def sibling2_height : ℕ := 66
def sibling3_height : ℕ := 60
def last_sibling_height : ℕ := 70  -- Derived from the solution steps
def eliza_height : ℕ := last_sibling_height - 2

-- The final question to validate
theorem number_of_siblings (h : 2 * sibling1_height + sibling3_height + last_sibling_height + eliza_height = total_height) :
  4 = 4 :=
by {
  -- Condition h states that the total height is satisfied
  -- Therefore, it directly justifies our claim without further computation here.
  sorry
}

end number_of_siblings_l145_145135


namespace started_with_l145_145564

-- Define the conditions
def total_eggs : ℕ := 70
def bought_eggs : ℕ := 62

-- Define the statement to prove
theorem started_with (initial_eggs : ℕ) : initial_eggs = total_eggs - bought_eggs → initial_eggs = 8 := by
  intro h
  sorry

end started_with_l145_145564


namespace probability_of_yellow_ball_l145_145453

theorem probability_of_yellow_ball 
  (red_balls : ℕ) 
  (yellow_balls : ℕ) 
  (blue_balls : ℕ) 
  (total_balls : ℕ)
  (h1 : red_balls = 2)
  (h2 : yellow_balls = 5)
  (h3 : blue_balls = 4)
  (h4 : total_balls = red_balls + yellow_balls + blue_balls) :
  (yellow_balls / total_balls : ℚ) = 5 / 11 :=
by 
  rw [h1, h2, h3] at h4  -- Substitute the ball counts into the total_balls definition.
  norm_num at h4  -- Simplify to verify the total is indeed 11.
  rw [h2, h4] -- Use the number of yellow balls and total number of balls to state the ratio.
  norm_num -- Normalize the fraction to show it equals 5/11.

#check probability_of_yellow_ball

end probability_of_yellow_ball_l145_145453


namespace intersect_at_single_point_l145_145552

theorem intersect_at_single_point :
  (∃ (x y : ℝ), y = 3 * x + 5 ∧ y = -5 * x + 20 ∧ y = 4 * x + p) → p = 25 / 8 :=
by
  sorry

end intersect_at_single_point_l145_145552


namespace remainder_n_squared_plus_3n_plus_5_l145_145627

theorem remainder_n_squared_plus_3n_plus_5 (n : ℕ) (h : n % 25 = 24) : (n^2 + 3 * n + 5) % 25 = 3 :=
by
  sorry

end remainder_n_squared_plus_3n_plus_5_l145_145627


namespace circle_radius_l145_145316

theorem circle_radius : 
  ∀ (x y : ℝ), x^2 + y^2 + 12 = 10 * x - 6 * y → ∃ r : ℝ, r = Real.sqrt 22 :=
by
  intros x y h
  -- Additional steps to complete the proof will be added here
  sorry

end circle_radius_l145_145316


namespace quadratic_vertex_l145_145820

theorem quadratic_vertex (x : ℝ) :
  ∃ (h k : ℝ), (h = -3) ∧ (k = -5) ∧ (∀ y, y = -2 * (x + h) ^ 2 + k) :=
sorry

end quadratic_vertex_l145_145820


namespace thieves_cloth_equation_l145_145358

theorem thieves_cloth_equation (x y : ℤ) 
  (h1 : y = 6 * x + 5)
  (h2 : y = 7 * x - 8) :
  6 * x + 5 = 7 * x - 8 :=
by
  sorry

end thieves_cloth_equation_l145_145358


namespace sequence_general_formula_l145_145631

-- Define the sequence S_n and the initial conditions
def S (n : ℕ) : ℕ := 3^(n + 1) - 1

-- Define the sequence a_n
def a (n : ℕ) : ℕ :=
  if n = 1 then 8 else 2 * 3^n

-- Theorem statement proving the general formula
theorem sequence_general_formula (n : ℕ) : 
  a n = if n = 1 then 8 else 2 * 3^n := by
  -- This is where the proof would go
  sorry

end sequence_general_formula_l145_145631


namespace ratio_of_sums_l145_145025

variable {α : Type*} [LinearOrderedField α] 

variable (a : ℕ → α) (S : ℕ → α)
variable (a1 d : α)

def isArithmeticSequence (a : ℕ → α) : Prop :=
  ∃ a1 d, ∀ n, a n = a1 + n * d

def sumArithmeticSequence (a : α) (d : α) (n : ℕ) : α :=
  n / 2 * (2 * a + (n - 1) * d)

theorem ratio_of_sums (h_arith : isArithmeticSequence a) (h_S : ∀ n, S n = sumArithmeticSequence a1 d n)
  (h_a5_5a3 : a 5 = 5 * a 3) : S 9 / S 5 = 9 := by sorry

end ratio_of_sums_l145_145025


namespace percentage_40_number_l145_145714

theorem percentage_40_number (x y z P : ℝ) (hx : x = 93.75) (hy : y = 0.40 * x) (hz : z = 6) (heq : (P / 100) * y = z) :
  P = 16 :=
sorry

end percentage_40_number_l145_145714


namespace common_ratio_l145_145575

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem common_ratio (a₁ : ℝ) (h : a₁ ≠ 0) : 
  (∀ S4 S5 S6, S5 = geometric_sum a₁ q 5 ∧ S4 = geometric_sum a₁ q 4 ∧ S6 = geometric_sum a₁ q 6 → 
  2 * S4 = S5 + S6) → 
  q = -2 := 
by
  sorry

end common_ratio_l145_145575


namespace sin_inequality_in_triangle_l145_145786

theorem sin_inequality_in_triangle (A B C : ℝ) (h_sum : A + B + C = Real.pi) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) :
  Real.sin A * Real.sin (A / 2) + Real.sin B * Real.sin (B / 2) + Real.sin C * Real.sin (C / 2) ≤ (4 * Real.sqrt 3) / 3 :=
by
  sorry

end sin_inequality_in_triangle_l145_145786


namespace problem_inequality_l145_145338

theorem problem_inequality (n : ℕ) (x : ℝ) (hn : n ≥ 2) (hx : |x| < 1) :
  2^n > (1 - x)^n + (1 + x)^n :=
sorry

end problem_inequality_l145_145338


namespace polynomial_abc_l145_145695

theorem polynomial_abc {a b c : ℝ} (h : a * x^2 + b * x + c = x^2 - 3 * x + 2) : a * b * c = -6 := by
  sorry

end polynomial_abc_l145_145695


namespace union_of_A_and_B_l145_145202

def set_A : Set Int := {0, 1}
def set_B : Set Int := {0, -1}

theorem union_of_A_and_B : set_A ∪ set_B = {-1, 0, 1} := by
  sorry

end union_of_A_and_B_l145_145202


namespace jacobs_hourly_wage_l145_145292

theorem jacobs_hourly_wage (jake_total_earnings : ℕ) (jake_days : ℕ) (hours_per_day : ℕ) (jake_thrice_jacob : ℕ) 
    (h_total_jake : jake_total_earnings = 720) 
    (h_jake_days : jake_days = 5) 
    (h_hours_per_day : hours_per_day = 8)
    (h_jake_thrice_jacob : jake_thrice_jacob = 3) 
    (jacob_hourly_wage : ℕ) :
  jacob_hourly_wage = 6 := 
by
  sorry

end jacobs_hourly_wage_l145_145292


namespace right_triangle_construction_condition_l145_145527

theorem right_triangle_construction_condition
  (b s : ℝ) 
  (h_b_pos : b > 0)
  (h_s_pos : s > 0)
  (h_perimeter : ∃ (AC BC AB : ℝ), AC = b ∧ AC + BC + AB = 2 * s ∧ (AC^2 + BC^2 = AB^2)) :
  b < s := 
sorry

end right_triangle_construction_condition_l145_145527


namespace example_theorem_l145_145837

-- Definitions of the conditions
def parallel (l1 l2 : Line) : Prop := sorry

def Angle (A B C : Point) : ℝ := sorry

-- Given conditions
def DC_parallel_AB (DC AB : Line) : Prop := parallel DC AB
def DCA_eq_55 (D C A : Point) : Prop := Angle D C A = 55
def ABC_eq_60 (A B C : Point) : Prop := Angle A B C = 60

-- Proof that angle ACB equals 5 degrees given the conditions
theorem example_theorem (D C A B : Point) (DC AB : Line) :
  DC_parallel_AB DC AB →
  DCA_eq_55 D C A →
  ABC_eq_60 A B C →
  Angle A C B = 5 := by
  sorry

end example_theorem_l145_145837


namespace no_six_consecutive010101_l145_145985

def unit_digit (n: ℕ) : ℕ := n % 10

def sequence : ℕ → ℕ
| 0     => 1
| 1     => 0
| 2     => 1
| 3     => 0
| 4     => 1
| 5     => 0
| (n + 6) => unit_digit (sequence n + sequence (n + 1) + sequence (n + 2) + sequence (n + 3) + sequence (n + 4) + sequence (n + 5))

theorem no_six_consecutive010101 : ∀ n, ¬ (sequence n = 0 ∧ sequence (n + 1) = 1 ∧ sequence (n + 2) = 0 ∧ sequence (n + 3) = 1 ∧ sequence (n + 4) = 0 ∧ sequence (n + 5) = 1) :=
sorry

end no_six_consecutive010101_l145_145985


namespace integer_count_between_sqrt8_and_sqrt78_l145_145938

theorem integer_count_between_sqrt8_and_sqrt78 :
  ∃ (n : ℕ), n = 6 ∧ ∀ (x : ℤ), (⌈Real.sqrt 8⌉ ≤ x ∧ x ≤ ⌊Real.sqrt 78⌋) ↔ (3 ≤ x ∧ x ≤ 8) := by
  sorry

end integer_count_between_sqrt8_and_sqrt78_l145_145938


namespace passengers_remaining_l145_145001

theorem passengers_remaining :
  let initial_passengers := 64
  let reduction_factor := (2 / 3)
  ∀ (n : ℕ), n = 4 → initial_passengers * reduction_factor^n = 1024 / 81 := by
sorry

end passengers_remaining_l145_145001


namespace complex_square_eq_l145_145004

variables {a b : ℝ} {i : ℂ}

theorem complex_square_eq :
  a + i = 2 - b * i → (a + b * i) ^ 2 = 3 - 4 * i :=
by sorry

end complex_square_eq_l145_145004


namespace angle_B_value_value_of_k_l145_145179

variable {A B C a b c : ℝ}
variable {k : ℝ}
variable {m n : ℝ × ℝ}

theorem angle_B_value
  (h1 : (2 * a - c) * Real.cos B = b * Real.cos C) :
  B = Real.pi / 3 :=
by sorry

theorem value_of_k
  (hA : 0 < A ∧ A < 2 * Real.pi / 3)
  (hm : m = (Real.sin A, Real.cos (2 * A)))
  (hn : n = (4 * k, 1))
  (hM : 4 * k * Real.sin A + Real.cos (2 * A) = 7) :
  k = 2 :=
by sorry

end angle_B_value_value_of_k_l145_145179


namespace find_x_l145_145089

theorem find_x (x : ℝ) (h : 0.25 * x = 200 - 30) : x = 680 := 
by
  sorry

end find_x_l145_145089


namespace JungMinBoughtWire_l145_145242

theorem JungMinBoughtWire
  (side_length : ℕ)
  (number_of_sides : ℕ)
  (remaining_wire : ℕ)
  (total_wire_bought : ℕ)
  (h1 : side_length = 13)
  (h2 : number_of_sides = 5)
  (h3 : remaining_wire = 8)
  (h4 : total_wire_bought = side_length * number_of_sides + remaining_wire) :
    total_wire_bought = 73 :=
by {
  sorry
}

end JungMinBoughtWire_l145_145242


namespace height_of_balcony_l145_145095

variable (t : ℝ) (v₀ : ℝ) (g : ℝ) (h₀ : ℝ)

axiom cond1 : t = 6
axiom cond2 : v₀ = 20
axiom cond3 : g = 10

theorem height_of_balcony : h₀ + v₀ * t - (1/2 : ℝ) * g * t^2 = 0 → h₀ = 60 :=
by
  intro h'
  sorry

end height_of_balcony_l145_145095


namespace probability_3_queens_or_at_least_2_aces_l145_145677

-- Definitions of drawing from a standard deck and probabilities involved
def num_cards : ℕ := 52
def num_queens : ℕ := 4
def num_aces : ℕ := 4

def probability_all_queens : ℚ := (4/52) * (3/51) * (2/50)
def probability_2_aces_1_non_ace : ℚ := (4/52) * (3/51) * (48/50)
def probability_3_aces : ℚ := (4/52) * (3/51) * (2/50)
def probability_at_least_2_aces : ℚ := (probability_2_aces_1_non_ace) + (probability_3_aces)

def total_probability : ℚ := probability_all_queens + probability_at_least_2_aces

-- Statement to be proved
theorem probability_3_queens_or_at_least_2_aces :
  total_probability = 220 / 581747 :=
sorry

end probability_3_queens_or_at_least_2_aces_l145_145677


namespace emily_total_spent_l145_145972

def total_cost (art_supplies_cost skirt_cost : ℕ) (number_of_skirts : ℕ) : ℕ :=
  art_supplies_cost + (skirt_cost * number_of_skirts)

theorem emily_total_spent :
  total_cost 20 15 2 = 50 :=
by
  sorry

end emily_total_spent_l145_145972


namespace loads_ratio_l145_145694

noncomputable def loads_wednesday : ℕ := 6
noncomputable def loads_friday (T : ℕ) : ℕ := T / 2
noncomputable def loads_saturday : ℕ := loads_wednesday / 3
noncomputable def total_loads_week (T : ℕ) : ℕ := loads_wednesday + T + loads_friday T + loads_saturday

theorem loads_ratio (T : ℕ) (h : total_loads_week T = 26) : T / loads_wednesday = 2 := 
by 
  -- proof steps would go here
  sorry

end loads_ratio_l145_145694


namespace largest_number_is_310_l145_145017

def largest_number_formed (a b c : ℕ) : ℕ :=
  max (a * 100 + b * 10 + c) (max (a * 100 + c * 10 + b) (max (b * 100 + a * 10 + c) 
  (max (b * 100 + c * 10 + a) (max (c * 100 + a * 10 + b) (c * 100 + b * 10 + a)))))

theorem largest_number_is_310 : largest_number_formed 3 1 0 = 310 :=
by simp [largest_number_formed]; sorry

end largest_number_is_310_l145_145017


namespace radius_of_cookie_l145_145384

theorem radius_of_cookie (x y : ℝ) : 
  (x^2 + y^2 + x - 5 * y = 10) → 
  ∃ r, (r = Real.sqrt (33 / 2)) :=
by
  sorry

end radius_of_cookie_l145_145384


namespace maximum_cookies_by_andy_l145_145240

-- Define the conditions
def total_cookies := 36
def cookies_by_andry (a : ℕ) := a
def cookies_by_alexa (a : ℕ) := 3 * a
def cookies_by_alice (a : ℕ) := 2 * a
def sum_cookies (a : ℕ) := cookies_by_andry a + cookies_by_alexa a + cookies_by_alice a

-- The theorem stating the problem and solution
theorem maximum_cookies_by_andy :
  ∃ a : ℕ, sum_cookies a = total_cookies ∧ a = 6 :=
by
  sorry

end maximum_cookies_by_andy_l145_145240


namespace smallest_base_l145_145282

theorem smallest_base : ∃ b : ℕ, (b^2 ≤ 120 ∧ 120 < b^3) ∧ ∀ n : ℕ, (n^2 ≤ 120 ∧ 120 < n^3) → b ≤ n :=
by sorry

end smallest_base_l145_145282


namespace omar_remaining_coffee_l145_145593

noncomputable def remaining_coffee : ℝ := 
  let initial_coffee := 12
  let after_first_drink := initial_coffee - (initial_coffee * 1/4)
  let after_office_drink := after_first_drink - (after_first_drink * 1/3)
  let espresso_in_ounces := 75 / 29.57
  let after_espresso := after_office_drink + espresso_in_ounces
  let after_lunch_drink := after_espresso - (after_espresso * 0.75)
  let iced_tea_addition := 4 * 1/2
  let after_iced_tea := after_lunch_drink + iced_tea_addition
  let after_cold_drink := after_iced_tea - (after_iced_tea * 0.6)
  after_cold_drink

theorem omar_remaining_coffee : remaining_coffee = 1.654 :=
by 
  sorry

end omar_remaining_coffee_l145_145593


namespace candidates_appeared_l145_145942

theorem candidates_appeared (x : ℝ) (h1 : 0.07 * x = 0.06 * x + 82) : x = 8200 :=
by
  sorry

end candidates_appeared_l145_145942


namespace student_departments_l145_145520

variable {Student : Type}
variable (Anna Vika Masha : Student)

-- Let Department be an enumeration type representing the three departments
inductive Department
| Literature : Department
| History : Department
| Biology : Department

open Department

variables (isLit : Student → Prop) (isHist : Student → Prop) (isBio : Student → Prop)

-- Conditions
axiom cond1 : isLit Anna → ¬isHist Masha
axiom cond2 : ¬isHist Vika → isLit Anna
axiom cond3 : ¬isLit Masha → isBio Vika

-- Target conclusion
theorem student_departments :
  isHist Vika ∧ isLit Masha ∧ isBio Anna :=
sorry

end student_departments_l145_145520


namespace area_of_triangle_l145_145732

noncomputable def findAreaOfTriangle (a b : ℝ) (cosAOF : ℝ) : ℝ := sorry

theorem area_of_triangle (a b cosAOF : ℝ)
  (ha : a = 15 / 7)
  (hb : b = Real.sqrt 21)
  (hcos : cosAOF = 2 / 5) :
  findAreaOfTriangle a b cosAOF = 6 := by
  rw [ha, hb, hcos]
  sorry

end area_of_triangle_l145_145732


namespace find_K_l145_145838

theorem find_K (K m n : ℝ) (p : ℝ) (hp : p = 0.3333333333333333)
  (eq1 : m = K * n + 5)
  (eq2 : m + 2 = K * (n + p) + 5) : 
  K = 6 := 
by
  sorry

end find_K_l145_145838


namespace gcd_360_504_l145_145310

theorem gcd_360_504 : Int.gcd 360 504 = 72 := by
  sorry

end gcd_360_504_l145_145310


namespace downstream_speed_l145_145476

theorem downstream_speed 
  (upstream_speed : ℕ) 
  (still_water_speed : ℕ) 
  (hm_upstream : upstream_speed = 27) 
  (hm_still_water : still_water_speed = 31) 
  : (still_water_speed + (still_water_speed - upstream_speed)) = 35 :=
by
  sorry

end downstream_speed_l145_145476


namespace part1_part2_l145_145215

theorem part1 (a b : ℝ) (h1 : ∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) (hb : b > 1) : a = 1 ∧ b = 2 :=
sorry

theorem part2 (k : ℝ) (x y : ℝ) (hx : x > 0) (hy : y > 0) (a b : ℝ) 
  (ha : a = 1) (hb : b = 2) 
  (h2 : a / x + b / y = 1)
  (h3 : 2 * x + y ≥ k^2 + k + 2) : -3 ≤ k ∧ k ≤ 2 :=
sorry

end part1_part2_l145_145215


namespace determine_constants_l145_145675

theorem determine_constants (P Q R : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 4 → x ≠ 6 → 
    (x^2 - 5*x + 6) / ((x - 1) * (x - 4) * (x - 6)) =
    P / (x - 1) + Q / (x - 4) + R / (x - 6)) →
  P = 2 / 15 ∧ Q = 1 / 3 ∧ R = 0 :=
by {
  sorry
}

end determine_constants_l145_145675


namespace max_acceptable_ages_l145_145534

noncomputable def acceptable_ages (avg_age std_dev : ℕ) : ℕ :=
  let lower_limit := avg_age - 2 * std_dev
  let upper_limit := avg_age + 2 * std_dev
  upper_limit - lower_limit + 1

theorem max_acceptable_ages : acceptable_ages 40 10 = 41 :=
by
  sorry

end max_acceptable_ages_l145_145534


namespace find_a_if_perpendicular_l145_145064

theorem find_a_if_perpendicular (a : ℝ) :
  (∀ x y : ℝ, x + a * y + 2 = 0 → 2 * x + 3 * y + 1 = 0 → False) →
  a = -2 / 3 :=
by
  sorry

end find_a_if_perpendicular_l145_145064


namespace susie_large_rooms_count_l145_145990

theorem susie_large_rooms_count:
  (∀ small_rooms medium_rooms large_rooms : ℕ,  
    (small_rooms = 4) → 
    (medium_rooms = 3) → 
    (large_rooms = x) → 
    (225 = small_rooms * 15 + medium_rooms * 25 + large_rooms * 35) → 
    x = 2) :=
by
  intros small_rooms medium_rooms large_rooms
  intros h1 h2 h3 h4
  sorry

end susie_large_rooms_count_l145_145990


namespace pencils_distributed_l145_145562

-- Define the conditions as a Lean statement
theorem pencils_distributed :
  let friends := 4
  let pencils := 8
  let at_least_one := 1
  ∃ (ways : ℕ), ways = 35 := sorry

end pencils_distributed_l145_145562


namespace graphs_intersect_at_three_points_l145_145887

noncomputable def is_invertible (f : ℝ → ℝ) := ∃ (g : ℝ → ℝ), ∀ x : ℝ, f (g x) = x ∧ g (f x) = x

theorem graphs_intersect_at_three_points (f : ℝ → ℝ) (h_inv : is_invertible f) :
  ∃ (xs : Finset ℝ), (∀ x ∈ xs, f (x^2) = f (x^6)) ∧ xs.card = 3 :=
by 
  sorry

end graphs_intersect_at_three_points_l145_145887


namespace magician_earnings_l145_145356

noncomputable def total_earnings (price_per_deck : ℕ) (initial_decks : ℕ) (end_decks : ℕ) (promotion_price : ℕ) (exchange_rate_start : ℚ) (exchange_rate_mid : ℚ) (foreign_sales_1 : ℕ) (domestic_sales : ℕ) (foreign_sales_2 : ℕ) : ℕ :=
  let foreign_earnings_1 := (foreign_sales_1 / 2) * promotion_price
  let foreign_earnings_2 := foreign_sales_2 * price_per_deck
  (domestic_sales / 2) * promotion_price + foreign_earnings_1 + foreign_earnings_2
  

-- Given conditions:
-- price_per_deck = 2
-- initial_decks = 5
-- end_decks = 3
-- promotion_price = 3
-- exchange_rate_start = 1
-- exchange_rate_mid = 1.5
-- foreign_sales_1 = 4
-- domestic_sales = 2
-- foreign_sales_2 = 1

theorem magician_earnings :
  total_earnings 2 5 3 3 1 1.5 4 2 1 = 11 :=
by
   sorry

end magician_earnings_l145_145356


namespace find_e_l145_145955

theorem find_e (b e : ℝ) (f g : ℝ → ℝ)
    (h1 : ∀ x, f x = 5 * x + b)
    (h2 : ∀ x, g x = b * x + 3)
    (h3 : ∀ x, f (g x) = 15 * x + e) : e = 18 :=
by
  sorry

end find_e_l145_145955


namespace students_solved_both_l145_145492

theorem students_solved_both (total_students solved_set_problem solved_function_problem both_problems_wrong: ℕ) 
  (h1: total_students = 50)
  (h2 : solved_set_problem = 40)
  (h3 : solved_function_problem = 31)
  (h4 : both_problems_wrong = 4) :
  (solved_set_problem + solved_function_problem - x + both_problems_wrong = total_students) → x = 25 := by
  sorry

end students_solved_both_l145_145492


namespace consequence_of_implication_l145_145205

-- Define the conditions
variable (A B : Prop)

-- State the theorem to prove
theorem consequence_of_implication (h : B → A) : A → B := 
  sorry

end consequence_of_implication_l145_145205


namespace sidney_thursday_jacks_l145_145583

open Nat

-- Define the number of jumping jacks Sidney did on each day
def monday_jacks := 20
def tuesday_jacks := 36
def wednesday_jacks := 40

-- Define the total number of jumping jacks done by Sidney
-- on Monday, Tuesday, and Wednesday
def sidney_mon_wed_jacks := monday_jacks + tuesday_jacks + wednesday_jacks

-- Define the total number of jumping jacks done by Brooke
def brooke_jacks := 438

-- Define the relationship between Brooke's and Sidney's total jumping jacks
def sidney_total_jacks := brooke_jacks / 3

-- Prove the number of jumping jacks Sidney did on Thursday
theorem sidney_thursday_jacks :
  sidney_total_jacks - sidney_mon_wed_jacks = 50 :=
by
  sorry

end sidney_thursday_jacks_l145_145583


namespace lowest_possible_sale_price_percentage_l145_145596

def list_price : ℝ := 80
def initial_discount : ℝ := 0.5
def additional_discount : ℝ := 0.2

theorem lowest_possible_sale_price_percentage 
  (list_price : ℝ) (initial_discount : ℝ) (additional_discount : ℝ) :
  ( (list_price - (list_price * initial_discount)) - (list_price * additional_discount) ) / list_price * 100 = 30 :=
by
  sorry

end lowest_possible_sale_price_percentage_l145_145596


namespace max_flow_increase_proof_l145_145038

noncomputable def max_flow_increase : ℕ :=
  sorry

theorem max_flow_increase_proof
  (initial_pipes_AB: ℕ) (initial_pipes_BC: ℕ) (flow_increase_per_pipes_swap: ℕ)
  (swap_increase: initial_pipes_AB = 10)
  (swap_increase_2: initial_pipes_BC = 10)
  (flow_increment: flow_increase_per_pipes_swap = 30) : 
  max_flow_increase = 150 :=
  sorry

end max_flow_increase_proof_l145_145038


namespace kelly_raisins_l145_145013

theorem kelly_raisins (weight_peanuts : ℝ) (total_weight_snacks : ℝ) (h1 : weight_peanuts = 0.1) (h2 : total_weight_snacks = 0.5) : total_weight_snacks - weight_peanuts = 0.4 := by
  sorry

end kelly_raisins_l145_145013


namespace trigonometric_identity_l145_145895

theorem trigonometric_identity :
  7 * 6 * (1 / Real.tan (2 * Real.pi * 10 / 360) + Real.tan (2 * Real.pi * 5 / 360)) 
  = 7 * 6 * (1 / Real.sin (2 * Real.pi * 10 / 360)) := 
sorry

end trigonometric_identity_l145_145895


namespace percentage_difference_between_M_and_J_is_34_74_percent_l145_145073

-- Definitions of incomes and relationships
variables (J T M : ℝ)
variables (h1 : T = 0.80 * J)
variables (h2 : M = 1.60 * T)

-- Definitions of savings and expenses
variables (Msavings : ℝ := 0.15 * M)
variables (Mexpenses : ℝ := 0.25 * M)
variables (Tsavings : ℝ := 0.12 * T)
variables (Texpenses : ℝ := 0.30 * T)
variables (Jsavings : ℝ := 0.18 * J)
variables (Jexpenses : ℝ := 0.20 * J)

-- Total savings and expenses
variables (Mtotal : ℝ := Msavings + Mexpenses)
variables (Jtotal : ℝ := Jsavings + Jexpenses)

-- Prove the percentage difference between Mary's and Juan's total savings and expenses combined
theorem percentage_difference_between_M_and_J_is_34_74_percent :
  M = 1.28 * J → 
  Mtotal = 0.40 * M →
  Jtotal = 0.38 * J →
  ( (Mtotal - Jtotal) / Jtotal ) * 100 = 34.74 :=
by
  sorry

end percentage_difference_between_M_and_J_is_34_74_percent_l145_145073


namespace pairs_bought_after_donation_l145_145874

-- Definitions from conditions
def initial_pairs : ℕ := 80
def donation_percentage : ℕ := 30
def post_donation_pairs : ℕ := 62

-- The theorem to be proven
theorem pairs_bought_after_donation : (initial_pairs - (donation_percentage * initial_pairs / 100) + 6 = post_donation_pairs) :=
by
  sorry

end pairs_bought_after_donation_l145_145874


namespace ratio_of_amount_spent_on_movies_to_weekly_allowance_l145_145066

-- Define weekly allowance
def weekly_allowance : ℕ := 10

-- Define final amount after all transactions
def final_amount : ℕ := 11

-- Define earnings from washing the car
def earnings : ℕ := 6

-- Define amount left before washing the car
def amount_left_before_wash : ℕ := final_amount - earnings

-- Define amount spent on movies
def amount_spent_on_movies : ℕ := weekly_allowance - amount_left_before_wash

-- Define the ratio function
def ratio (a b : ℕ) : ℚ := a / b

-- Prove the required ratio
theorem ratio_of_amount_spent_on_movies_to_weekly_allowance :
  ratio amount_spent_on_movies weekly_allowance = 1 / 2 :=
by
  sorry

end ratio_of_amount_spent_on_movies_to_weekly_allowance_l145_145066


namespace cube_inequality_sufficient_and_necessary_l145_145301

theorem cube_inequality_sufficient_and_necessary (a b : ℝ) :
  (a > b ↔ a^3 > b^3) := 
sorry

end cube_inequality_sufficient_and_necessary_l145_145301


namespace product_of_possible_values_of_N_l145_145883

theorem product_of_possible_values_of_N (M L N : ℝ) (h1 : M = L + N) (h2 : M - 5 = (L + N) - 5) (h3 : L + 3 = L + 3) (h4 : |(L + N - 5) - (L + 3)| = 2) : 10 * 6 = 60 := by
  sorry

end product_of_possible_values_of_N_l145_145883


namespace coordinates_of_point_in_fourth_quadrant_l145_145765

theorem coordinates_of_point_in_fourth_quadrant 
  (P : ℝ × ℝ)
  (h₁ : P.1 > 0) -- P is in the fourth quadrant, so x > 0
  (h₂ : P.2 < 0) -- P is in the fourth quadrant, so y < 0
  (dist_x_axis : P.2 = -5) -- Distance from P to x-axis is 5 (absolute value of y)
  (dist_y_axis : P.1 = 3)  -- Distance from P to y-axis is 3 (absolute value of x)
  : P = (3, -5) :=
sorry

end coordinates_of_point_in_fourth_quadrant_l145_145765


namespace number_of_three_digit_integers_l145_145196

-- Defining the set of available digits
def digits : List ℕ := [3, 5, 8, 9]

-- Defining the property for selecting a digit without repetition
def no_repetition (l : List ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ l → l.filter (fun x => x = d) = [d]

-- The main theorem stating the number of three-digit integers that can be formed
theorem number_of_three_digit_integers (h : no_repetition digits) : 
  ∃ n : ℕ, n = 24 :=
by
  sorry

end number_of_three_digit_integers_l145_145196


namespace algebraic_expression_value_l145_145284

theorem algebraic_expression_value (a : ℝ) (h : a^2 - 4 * a + 3 = 0) : -2 * a^2 + 8 * a - 5 = 1 := 
by 
  sorry 

end algebraic_expression_value_l145_145284


namespace system_of_inequalities_l145_145235

theorem system_of_inequalities (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : (0.5 < p ∧ p < 5 / 9) :=
by sorry

end system_of_inequalities_l145_145235


namespace circumscribed_sphere_radius_l145_145977

theorem circumscribed_sphere_radius (a b c : ℝ) : 
  R = (1/2) * Real.sqrt (a^2 + b^2 + c^2) := sorry

end circumscribed_sphere_radius_l145_145977


namespace evaluate_expression_l145_145204

theorem evaluate_expression (a : ℝ) (h : a = 2) : 
    (a / (a^2 - 1) - 1 / (a^2 - 1)) = 1 / 3 := by
  sorry

end evaluate_expression_l145_145204


namespace total_apartments_in_building_l145_145101

theorem total_apartments_in_building (A k m n : ℕ)
  (cond1 : 5 = A)
  (cond2 : 636 = (m-1) * k + n)
  (cond3 : 242 = (A-m) * k + n) :
  A * k = 985 :=
by
  sorry

end total_apartments_in_building_l145_145101


namespace sum_of_digits_7_pow_11_l145_145393

/-
  Let's define the problem with the given conditions and expected answer.
  We need to prove that the sum of the tens digit and the ones digit of 7^11 is 7.
-/

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_7_pow_11 : tens_digit (7 ^ 11) + ones_digit (7 ^ 11) = 7 :=
sorry

end sum_of_digits_7_pow_11_l145_145393


namespace second_root_of_system_l145_145630

def system_of_equations (x y : ℝ) : Prop :=
  (2 * x^2 + 3 * x * y + y^2 = 70) ∧ (6 * x^2 + x * y - y^2 = 50)

theorem second_root_of_system :
  system_of_equations 3 4 →
  system_of_equations (-3) (-4) :=
by
  sorry

end second_root_of_system_l145_145630


namespace sum_remainder_mod_9_l145_145400

theorem sum_remainder_mod_9 : 
  (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 9 = 6 :=
by
  sorry

end sum_remainder_mod_9_l145_145400


namespace social_studies_score_l145_145346

-- Step d): Translate to Lean 4
theorem social_studies_score 
  (K E S SS : ℝ)
  (h1 : (K + E + S) / 3 = 89)
  (h2 : (K + E + S + SS) / 4 = 90) :
  SS = 93 :=
by
  -- We'll leave the mathematics formal proof details to Lean.
  sorry

end social_studies_score_l145_145346


namespace return_trip_time_l145_145684

-- Define the given conditions
def run_time : ℕ := 20
def jog_time : ℕ := 10
def trip_time := run_time + jog_time
def multiplier: ℕ := 3

-- State the theorem
theorem return_trip_time : trip_time * multiplier = 90 := by
  sorry

end return_trip_time_l145_145684


namespace min_value_quadratic_l145_145712

theorem min_value_quadratic (x : ℝ) : x = -1 ↔ (∀ y : ℝ, x^2 + 2*x + 4 ≤ y) := by
  sorry

end min_value_quadratic_l145_145712


namespace odd_integer_95th_l145_145470

theorem odd_integer_95th : (2 * 95 - 1) = 189 := 
by
  -- The proof would go here
  sorry

end odd_integer_95th_l145_145470


namespace solve_inequality_l145_145367

theorem solve_inequality (x : ℝ) : 2 * x + 4 > 0 ↔ x > -2 := sorry

end solve_inequality_l145_145367


namespace second_school_more_students_l145_145870

theorem second_school_more_students (S1 S2 S3 : ℕ) 
  (hS3 : S3 = 200) 
  (hS1 : S1 = 2 * S2) 
  (h_total : S1 + S2 + S3 = 920) : 
  S2 - S3 = 40 :=
by
  sorry

end second_school_more_students_l145_145870


namespace christmas_tree_seller_l145_145831

theorem christmas_tree_seller 
  (cost_spruce : ℕ := 220) 
  (cost_pine : ℕ := 250) 
  (cost_fir : ℕ := 330) 
  (total_revenue : ℕ := 36000) 
  (equal_trees: ℕ) 
  (h_costs : cost_spruce + cost_pine + cost_fir = 800) 
  (h_revenue : equal_trees * 800 = total_revenue):
  3 * equal_trees = 135 :=
sorry

end christmas_tree_seller_l145_145831


namespace LCM_of_fractions_l145_145968

theorem LCM_of_fractions (x : ℕ) (h : x > 0) : 
  lcm (1 / (4 * x)) (lcm (1 / (6 * x)) (1 / (9 * x))) = 1 / (36 * x) :=
by
  sorry

end LCM_of_fractions_l145_145968


namespace problem_solution_l145_145751

theorem problem_solution (x y z : ℝ) (h1 : x * y + y * z + z * x = 4) (h2 : x * y * z = 6) :
  (x * y - (3 / 2) * (x + y)) * (y * z - (3 / 2) * (y + z)) * (z * x - (3 / 2) * (z + x)) = 81 / 4 :=
by
  sorry

end problem_solution_l145_145751


namespace difference_of_x_values_l145_145697

theorem difference_of_x_values : 
  ∀ x y : ℝ, ( (x + 3) ^ 2 / (3 * x + 29) = 2 ∧ (y + 3) ^ 2 / (3 * y + 29) = 2 ) → |x - y| = 14 := 
sorry

end difference_of_x_values_l145_145697


namespace intersection_of_sets_l145_145842

def setA : Set ℝ := {x | x^2 - 1 ≥ 0}
def setB : Set ℝ := {x | 0 < x ∧ x < 4}

theorem intersection_of_sets : (setA ∩ setB) = {x | 1 ≤ x ∧ x < 4} := 
by 
  sorry

end intersection_of_sets_l145_145842


namespace train_length_l145_145749

def train_speed_kmph := 25 -- speed of train in km/h
def man_speed_kmph := 2 -- speed of man in km/h
def crossing_time_sec := 52 -- time to cross in seconds

noncomputable def length_of_train : ℝ :=
  let relative_speed_kmph := train_speed_kmph + man_speed_kmph -- relative speed in km/h
  let relative_speed_mps := relative_speed_kmph * (5 / 18) -- convert to m/s
  relative_speed_mps * crossing_time_sec -- length of train in meters

theorem train_length : length_of_train = 390 :=
  by sorry -- proof omitted

end train_length_l145_145749


namespace sum_of_three_numbers_l145_145256

theorem sum_of_three_numbers (x y z : ℝ) (h1 : x + y = 31) (h2 : y + z = 41) (h3 : z + x = 55) :
  x + y + z = 63.5 :=
by
  sorry

end sum_of_three_numbers_l145_145256


namespace z_value_l145_145736

theorem z_value (x y z : ℝ) (h : 1 / x + 1 / y = 2 / z) : z = (x * y) / 2 :=
by
  sorry

end z_value_l145_145736


namespace area_increase_factor_l145_145970

theorem area_increase_factor (s : ℝ) :
  let A_original := s^2
  let A_new := (3 * s)^2
  A_new / A_original = 9 := by
  sorry

end area_increase_factor_l145_145970


namespace percent_shaded_area_of_rectangle_l145_145416

theorem percent_shaded_area_of_rectangle
  (side_length : ℝ)
  (length_rectangle : ℝ)
  (width_rectangle : ℝ)
  (overlap_length : ℝ)
  (h1 : side_length = 12)
  (h2 : length_rectangle = 20)
  (h3 : width_rectangle = 12)
  (h4 : overlap_length = 4)
  : (overlap_length * width_rectangle) / (length_rectangle * width_rectangle) * 100 = 20 :=
  sorry

end percent_shaded_area_of_rectangle_l145_145416


namespace remainder_when_divided_l145_145857

theorem remainder_when_divided (m : ℤ) (h : m % 5 = 2) : (m + 2535) % 5 = 2 := 
by sorry

end remainder_when_divided_l145_145857


namespace factorization_example_l145_145798

theorem factorization_example :
  (x : ℝ) → (x^2 + 6 * x + 9 = (x + 3)^2) :=
by
  sorry

end factorization_example_l145_145798


namespace selling_price_of_radio_l145_145663

theorem selling_price_of_radio (CP LP : ℝ) (hCP : CP = 1500) (hLP : LP = 14.000000000000002) : 
  CP - (LP / 100 * CP) = 1290 :=
by
  -- Given definitions
  have h1 : CP - (LP / 100 * CP) = 1290 := sorry
  exact h1

end selling_price_of_radio_l145_145663


namespace g_neg_two_is_zero_l145_145458

theorem g_neg_two_is_zero {f g : ℤ → ℤ} 
  (h_odd: ∀ x: ℤ, f (-x) + (-x) = -(f x + x)) 
  (hf_two: f 2 = 1) 
  (hg_def: ∀ x: ℤ, g x = f x + 1):
  g (-2) = 0 := 
sorry

end g_neg_two_is_zero_l145_145458


namespace number_of_cows_l145_145029

variable (D C : Nat)

theorem number_of_cows (h : 2 * D + 4 * C = 2 * (D + C) + 30) : C = 15 :=
by
  sorry

end number_of_cows_l145_145029


namespace set_intersection_l145_145603

open Set

variable (U : Set ℝ)
variable (A B : Set ℝ)

def complement (s : Set ℝ) := {x : ℝ | x ∉ s}

theorem set_intersection (hU : U = univ)
                         (hA : A = {x : ℝ | x > 0})
                         (hB : B = {x : ℝ | x > 1}) :
  A ∩ complement B = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end set_intersection_l145_145603


namespace division_result_l145_145154

-- Definitions for the values used in the problem
def numerator := 0.0048 * 3.5
def denominator := 0.05 * 0.1 * 0.004

-- Theorem statement
theorem division_result : numerator / denominator = 840 := by 
  sorry

end division_result_l145_145154


namespace jacks_walking_rate_l145_145746

variable (distance : ℝ) (hours : ℝ) (minutes : ℝ)

theorem jacks_walking_rate (h_distance : distance = 4) (h_hours : hours = 1) (h_minutes : minutes = 15) :
  distance / (hours + minutes / 60) = 3.2 :=
by
  sorry

end jacks_walking_rate_l145_145746


namespace glasses_in_smaller_box_l145_145602

variable (x : ℕ)

theorem glasses_in_smaller_box (h : (x + 16) / 2 = 15) : x = 14 :=
by
  sorry

end glasses_in_smaller_box_l145_145602


namespace greatest_int_satisfying_inequality_l145_145495

theorem greatest_int_satisfying_inequality : 
  ∃ m : ℤ, (∀ x : ℤ, x - 5 > 4 * x - 1 → x ≤ -2) ∧ (∀ k : ℤ, k < -2 → k - 5 > 4 * k - 1) :=
by
  sorry

end greatest_int_satisfying_inequality_l145_145495


namespace length_of_CD_l145_145846

theorem length_of_CD
  (radius : ℝ)
  (length : ℝ)
  (total_volume : ℝ)
  (cylinder_volume : ℝ := π * radius^2 * length)
  (hemisphere_volume : ℝ := (2 * (2/3) * π * radius^3))
  (h1 : radius = 4)
  (h2 : total_volume = 432 * π)
  (h3 : total_volume = cylinder_volume + hemisphere_volume) :
  length = 22 := by
sorry

end length_of_CD_l145_145846


namespace integral_log_eq_ln2_l145_145890

theorem integral_log_eq_ln2 :
  ∫ x in (0 : ℝ)..(1 : ℝ), (1 / (x + 1)) = Real.log 2 :=
by
  sorry

end integral_log_eq_ln2_l145_145890


namespace beetle_crawls_100th_segment_in_1300_seconds_l145_145609

def segment_length (n : ℕ) : ℕ :=
  (n / 4) + 1

def total_length (s : ℕ) : ℕ :=
  (s / 4) * 4 * (segment_length (s - 1)) * (segment_length (s - 1) + 1) / 2

theorem beetle_crawls_100th_segment_in_1300_seconds :
  total_length 100 = 1300 :=
  sorry

end beetle_crawls_100th_segment_in_1300_seconds_l145_145609


namespace find_a1000_l145_145465

noncomputable def seq (a : ℕ → ℤ) : Prop :=
a 1 = 1009 ∧
a 2 = 1010 ∧
(∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = 2 * n)

theorem find_a1000 (a : ℕ → ℤ) (h : seq a) : a 1000 = 1675 :=
sorry

end find_a1000_l145_145465


namespace ratio_of_construction_paper_packs_l145_145554

-- Definitions for conditions
def marie_glue_sticks : Nat := 15
def marie_construction_paper : Nat := 30
def allison_total_items : Nat := 28
def allison_additional_glue_sticks : Nat := 8

-- Define the main quantity to prove
def allison_glue_sticks : Nat := marie_glue_sticks + allison_additional_glue_sticks
def allison_construction_paper : Nat := allison_total_items - allison_glue_sticks

-- The ratio should be of type Rat or Nat
theorem ratio_of_construction_paper_packs : (marie_construction_paper : Nat) / allison_construction_paper = 6 / 1 := by
  -- This is a placeholder for the actual proof
  sorry

end ratio_of_construction_paper_packs_l145_145554


namespace direct_proportion_function_decrease_no_first_quadrant_l145_145190

-- Part (1)
theorem direct_proportion_function (a b : ℝ) (h : y = (2*a-4)*x + (3-b)) : a ≠ 2 ∧ b = 3 :=
sorry

-- Part (2)
theorem decrease_no_first_quadrant (a b : ℝ) (h : y = (2*a-4)*x + (3-b)) : a < 2 ∧ b ≥ 3 :=
sorry

end direct_proportion_function_decrease_no_first_quadrant_l145_145190


namespace find_c_l145_145499

variables {α : Type*} [LinearOrderedField α]

def p (x : α) : α := 3 * x - 9
def q (x : α) (c : α) : α := 4 * x - c

-- We aim to prove that if p(q(3,c)) = 6, then c = 7
theorem find_c (c : α) : p (q 3 c) = 6 → c = 7 :=
by
  sorry

end find_c_l145_145499


namespace train_speed_l145_145951

theorem train_speed (length1 length2 speed2 : ℝ) (time_seconds speed1 : ℝ)
    (h_length1 : length1 = 111)
    (h_length2 : length2 = 165)
    (h_speed2 : speed2 = 90)
    (h_time : time_seconds = 6.623470122390208)
    (h_speed1 : speed1 = 60) :
    (length1 / 1000.0) + (length2 / 1000.0) / (time_seconds / 3600) = speed1 + speed2 :=
by
  sorry

end train_speed_l145_145951


namespace fraction_half_l145_145506

theorem fraction_half {A : ℕ} (h : 8 * (A + 8) - 8 * (A - 8) = 128) (age_eq : A = 64) :
  (64 : ℚ) / (128 : ℚ) = 1 / 2 :=
by
  sorry

end fraction_half_l145_145506


namespace chemistry_more_than_physics_l145_145537

variables (M P C x : ℤ)

-- Condition 1: The total marks in mathematics and physics is 50
def condition1 : Prop := M + P = 50

-- Condition 2: The average marks in mathematics and chemistry together is 35
def condition2 : Prop := (M + C) / 2 = 35

-- Condition 3: The score in chemistry is some marks more than that in physics
def condition3 : Prop := C = P + x

theorem chemistry_more_than_physics :
  condition1 M P ∧ condition2 M C ∧ (∃ x : ℤ, condition3 P C x ∧ x = 20) :=
sorry

end chemistry_more_than_physics_l145_145537


namespace prove_solution_l145_145033

noncomputable def problem_statement : Prop := ∀ x : ℝ, (16 : ℝ)^(2 * x - 3) = (4 : ℝ)^(3 - x) → x = 9 / 5

theorem prove_solution : problem_statement :=
by
  intro x h
  -- The proof would go here
  sorry

end prove_solution_l145_145033


namespace percentage_apples_basket_l145_145719

theorem percentage_apples_basket :
  let initial_apples := 10
  let initial_oranges := 5
  let added_oranges := 5
  let total_apples := initial_apples
  let total_oranges := initial_oranges + added_oranges
  let total_fruits := total_apples + total_oranges
  (total_apples / total_fruits) * 100 = 50 :=
by
  sorry

end percentage_apples_basket_l145_145719


namespace rate_of_stream_is_5_l145_145915

-- Define the conditions
def boat_speed : ℝ := 16  -- Boat speed in still water
def time_downstream : ℝ := 3  -- Time taken downstream
def distance_downstream : ℝ := 63  -- Distance covered downstream

-- Define the rate of the stream as an unknown variable
def rate_of_stream (v : ℝ) : Prop := 
  distance_downstream = (boat_speed + v) * time_downstream

-- Statement to prove
theorem rate_of_stream_is_5 : 
  ∃ (v : ℝ), rate_of_stream v ∧ v = 5 :=
by
  use 5
  simp [boat_speed, time_downstream, distance_downstream, rate_of_stream]
  sorry

end rate_of_stream_is_5_l145_145915


namespace simplify_fraction_l145_145710

theorem simplify_fraction (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) : 
  (8 * a^4 * b^2 * c) / (4 * a^3 * b) = 2 * a * b * c :=
by
  sorry

end simplify_fraction_l145_145710


namespace cannot_achieve_1_5_percent_salt_solution_l145_145773

-- Define the initial concentrations and volumes
def initial_state (V1 V2 : ℝ) (C1 C2 : ℝ) : Prop :=
  V1 = 1 ∧ C1 = 0 ∧ V2 = 1 ∧ C2 = 0.02

-- Define the transfer and mixing operation
noncomputable def transfer_and_mix (V1_old V2_old C1_old C2_old : ℝ) (amount_to_transfer : ℝ)
  (new_V1 new_V2 new_C1 new_C2 : ℝ) : Prop :=
  amount_to_transfer ≤ V2_old ∧
  new_V1 = V1_old + amount_to_transfer ∧
  new_V2 = V2_old - amount_to_transfer ∧
  new_C1 = (V1_old * C1_old + amount_to_transfer * C2_old) / new_V1 ∧
  new_C2 = (V2_old * C2_old - amount_to_transfer * C2_old) / new_V2

-- Prove that it is impossible to achieve a 1.5% salt concentration in container 1
theorem cannot_achieve_1_5_percent_salt_solution :
  ∀ V1 V2 C1 C2, initial_state V1 V2 C1 C2 →
  ¬ ∃ V1' V2' C1' C2', transfer_and_mix V1 V2 C1 C2 0.5 V1' V2' C1' C2' ∧ C1' = 0.015 :=
by
  intros
  sorry

end cannot_achieve_1_5_percent_salt_solution_l145_145773


namespace sum_of_coefficients_l145_145511

noncomputable def simplify (x : ℝ) : ℝ := 
  (x^3 + 11 * x^2 + 38 * x + 40) / (x + 3)

theorem sum_of_coefficients : 
  (∀ x : ℝ, (x ≠ -3) → (simplify x = x^2 + 8 * x + 14)) ∧
  (1 + 8 + 14 + -3 = 20) :=
by      
  sorry

end sum_of_coefficients_l145_145511


namespace max_choir_members_l145_145605

theorem max_choir_members (n : ℕ) (x y : ℕ) : 
  n = x^2 + 11 ∧ n = y * (y + 3) → n = 54 :=
by
  sorry

end max_choir_members_l145_145605


namespace show_length_50_l145_145471

def Gina_sSis_three_as_often (G S : ℕ) : Prop := G = 3 * S
def sister_total_shows (G S : ℕ) : Prop := G + S = 24
def Gina_total_minutes (G : ℕ) (minutes : ℕ) : Prop := minutes = 900
def length_of_each_show (minutes shows length : ℕ) : Prop := length = minutes / shows

theorem show_length_50 (G S : ℕ) (length : ℕ) :
  Gina_sSis_three_as_often G S →
  sister_total_shows G S →
  Gina_total_minutes G 900 →
  length_of_each_show 900 G length →
  length = 50 :=
by
  intros h1 h2 h3 h4
  sorry

end show_length_50_l145_145471


namespace total_bricks_used_l145_145428

def numberOfCoursesPerWall := 6
def bricksPerCourse := 10
def numberOfWalls := 4
def incompleteCourses := 2

theorem total_bricks_used :
  (numberOfCoursesPerWall * bricksPerCourse * (numberOfWalls - 1)) + ((numberOfCoursesPerWall - incompleteCourses) * bricksPerCourse) = 220 :=
by
  -- Proof goes here
  sorry

end total_bricks_used_l145_145428


namespace male_students_outnumber_female_students_l145_145277

-- Define the given conditions
def total_students : ℕ := 928
def male_students : ℕ := 713
def female_students : ℕ := total_students - male_students

-- The theorem to be proven
theorem male_students_outnumber_female_students :
  male_students - female_students = 498 :=
by
  sorry

end male_students_outnumber_female_students_l145_145277


namespace average_daily_visitors_l145_145022

theorem average_daily_visitors
    (avg_sun : ℕ)
    (avg_other : ℕ)
    (days : ℕ)
    (starts_sun : Bool)
    (H1 : avg_sun = 630)
    (H2 : avg_other = 240)
    (H3 : days = 30)
    (H4 : starts_sun = true) :
    (5 * avg_sun + 25 * avg_other) / days = 305 :=
by
  sorry

end average_daily_visitors_l145_145022


namespace buffalo_theft_l145_145563

theorem buffalo_theft (initial_apples falling_apples remaining_apples stolen_apples : ℕ)
  (h1 : initial_apples = 79)
  (h2 : falling_apples = 26)
  (h3 : remaining_apples = 8) :
  initial_apples - falling_apples - stolen_apples = remaining_apples ↔ stolen_apples = 45 :=
by sorry

end buffalo_theft_l145_145563


namespace proof_complex_magnitude_z_l145_145295

noncomputable def complex_magnitude_z : Prop :=
  ∀ (z : ℂ),
    (z * (Complex.cos (Real.pi / 9) + Complex.sin (Real.pi / 9) * Complex.I) ^ 6 = 2) →
    Complex.abs z = 2

theorem proof_complex_magnitude_z : complex_magnitude_z :=
by
  intros z h
  sorry

end proof_complex_magnitude_z_l145_145295


namespace problem_statement_l145_145830

theorem problem_statement :
  (3 = 0.25 * x) ∧ (3 = 0.50 * y) → (x - y = 6) ∧ (x + y = 18) :=
by
  sorry

end problem_statement_l145_145830


namespace vacation_cost_division_l145_145385

theorem vacation_cost_division 
  (total_cost : ℝ) 
  (initial_people : ℝ) 
  (initial_cost_per_person : ℝ) 
  (cost_difference : ℝ) 
  (new_cost_per_person : ℝ) 
  (new_people : ℝ) 
  (h1 : total_cost = 1000) 
  (h2 : initial_people = 4) 
  (h3 : initial_cost_per_person = total_cost / initial_people) 
  (h4 : initial_cost_per_person = 250) 
  (h5 : cost_difference = 50) 
  (h6 : new_cost_per_person = initial_cost_per_person - cost_difference) 
  (h7 : new_cost_per_person = 200) 
  (h8 : total_cost / new_people = new_cost_per_person) :
  new_people = 5 := 
sorry

end vacation_cost_division_l145_145385


namespace bike_cost_l145_145935

theorem bike_cost (h1: 8 > 0) (h2: 35 > 0) (weeks_in_month: ℕ := 4) (saved: ℕ := 720):
  let hourly_wage := 8
  let weekly_hours := 35
  let weekly_earnings := weekly_hours * hourly_wage
  let monthly_earnings := weekly_earnings * weeks_in_month
  let cost_of_bike := monthly_earnings - saved
  cost_of_bike = 400 :=
by
  sorry

end bike_cost_l145_145935


namespace ineq_power_sum_lt_pow_two_l145_145137

theorem ineq_power_sum_lt_pow_two (x : ℝ) (n : ℕ) (hx : |x| < 1) (hn : 2 ≤ n) : 
  (1 + x)^n + (1 - x)^n < 2^n :=
by
  sorry

end ineq_power_sum_lt_pow_two_l145_145137


namespace division_quotient_remainder_l145_145565

theorem division_quotient_remainder :
  ∃ (q r : ℝ), 76.6 = 1.8 * q + r ∧ 0 ≤ r ∧ r < 1.8 ∧ q = 42 ∧ r = 1 := by
  sorry

end division_quotient_remainder_l145_145565


namespace isosceles_triangle_vertex_angle_l145_145222

theorem isosceles_triangle_vertex_angle (θ : ℝ) (h₀ : θ = 80) (h₁ : ∃ (x y z : ℝ), (x = y ∨ y = z ∨ z = x) ∧ x + y + z = 180) : θ = 80 ∨ θ = 20 := 
sorry

end isosceles_triangle_vertex_angle_l145_145222


namespace ratio_of_x_to_y_l145_145651

theorem ratio_of_x_to_y (x y : ℚ) (h : (8*x - 5*y)/(10*x - 3*y) = 4/7) : x/y = 23/16 :=
by 
  sorry

end ratio_of_x_to_y_l145_145651


namespace complex_solutions_x2_eq_neg4_l145_145078

-- Lean statement for the proof problem
theorem complex_solutions_x2_eq_neg4 (x : ℂ) (hx : x^2 = -4) : x = 2 * Complex.I ∨ x = -2 * Complex.I :=
by 
  sorry

end complex_solutions_x2_eq_neg4_l145_145078


namespace area_of_border_correct_l145_145130

def height_of_photograph : ℕ := 12
def width_of_photograph : ℕ := 16
def border_width : ℕ := 3
def lining_width : ℕ := 1

def area_of_photograph : ℕ := height_of_photograph * width_of_photograph

def total_height : ℕ := height_of_photograph + 2 * (lining_width + border_width)
def total_width : ℕ := width_of_photograph + 2 * (lining_width + border_width)

def area_of_framed_area : ℕ := total_height * total_width

def area_of_border_including_lining : ℕ := area_of_framed_area - area_of_photograph

theorem area_of_border_correct : area_of_border_including_lining = 288 := by
  sorry

end area_of_border_correct_l145_145130


namespace roots_square_sum_l145_145353

theorem roots_square_sum (r s p q : ℝ) 
  (root_cond : ∀ x : ℝ, x^2 - 2 * p * x + 3 * q = 0 → (x = r ∨ x = s)) :
  r^2 + s^2 = 4 * p^2 - 6 * q :=
by
  sorry

end roots_square_sum_l145_145353


namespace moles_of_NaCl_formed_l145_145114

-- Given conditions
def sodium_bisulfite_moles : ℕ := 2
def hydrochloric_acid_moles : ℕ := 2
def balanced_reaction : Prop :=
  ∀ (NaHSO3 HCl NaCl H2O SO2 : ℕ), 
    NaHSO3 + HCl = NaCl + H2O + SO2

-- Target to prove:
theorem moles_of_NaCl_formed :
  balanced_reaction → sodium_bisulfite_moles = hydrochloric_acid_moles → 
  sodium_bisulfite_moles = 2 := 
sorry

end moles_of_NaCl_formed_l145_145114


namespace xiao_liang_correct_l145_145315

theorem xiao_liang_correct :
  ∀ (x : ℕ), (0 ≤ x ∧ x ≤ 26 ∧ 30 - x ≤ 24 ∧ 26 - x ≤ 20) →
  let boys_A := x
  let girls_A := 30 - x
  let boys_B := 26 - x
  let girls_B := 24 - girls_A
  ∃ k : ℤ, boys_A - girls_B = 6 := 
by 
  sorry

end xiao_liang_correct_l145_145315


namespace tim_buys_loaves_l145_145297

theorem tim_buys_loaves (slices_per_loaf : ℕ) (paid : ℕ) (change : ℕ) (price_per_slice_cents : ℕ) 
    (h1 : slices_per_loaf = 20) 
    (h2 : paid = 2 * 20) 
    (h3 : change = 16) 
    (h4 : price_per_slice_cents = 40) : 
    (paid - change) / (slices_per_loaf * price_per_slice_cents / 100) = 3 := 
by 
  -- proof omitted 
  sorry

end tim_buys_loaves_l145_145297


namespace total_money_received_a_l145_145194

-- Define the partners and their capitals
structure Partner :=
  (name : String)
  (capital : ℕ)
  (isWorking : Bool)

def a : Partner := { name := "a", capital := 3500, isWorking := true }
def b : Partner := { name := "b", capital := 2500, isWorking := false }

-- Define the total profit
def totalProfit : ℕ := 9600

-- Define the managing fee as 10% of total profit
def managingFee (total : ℕ) : ℕ := (10 * total) / 100

-- Define the remaining profit after deducting the managing fee
def remainingProfit (total : ℕ) (fee : ℕ) : ℕ := total - fee

-- Calculate the share of remaining profit based on capital contribution
def share (capital totalCapital remaining : ℕ) : ℕ := (capital * remaining) / totalCapital

-- Theorem to prove the total money received by partner a
theorem total_money_received_a :
  let totalCapitals := a.capital + b.capital
  let fee := managingFee totalProfit
  let remaining := remainingProfit totalProfit fee
  let aShare := share a.capital totalCapitals remaining
  (fee + aShare) = 6000 :=
by
  sorry

end total_money_received_a_l145_145194


namespace Katie_has_more_games_than_friends_l145_145160

def katie_new_games : ℕ := 57
def katie_old_games : ℕ := 39
def friends_new_games : ℕ := 34

theorem Katie_has_more_games_than_friends :
  (katie_new_games + katie_old_games) - friends_new_games = 62 := by
  sorry

end Katie_has_more_games_than_friends_l145_145160


namespace edward_can_buy_candies_l145_145791

theorem edward_can_buy_candies (whack_a_mole_tickets skee_ball_tickets candy_cost : ℕ)
  (h1 : whack_a_mole_tickets = 3) (h2 : skee_ball_tickets = 5) (h3 : candy_cost = 4) :
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost = 2 :=
by
  sorry

end edward_can_buy_candies_l145_145791


namespace quadratic_roots_difference_l145_145755

theorem quadratic_roots_difference (p q : ℝ) (hp : 0 < p) (hq : 0 < q) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 - x2 = 2 ∧ x1 * x2 = q ∧ x1 + x2 = -p) → p = 2 * Real.sqrt (q + 1) :=
by
  sorry

end quadratic_roots_difference_l145_145755


namespace cost_of_each_box_of_pencils_l145_145311

-- Definitions based on conditions
def cartons_of_pencils : ℕ := 20
def boxes_per_carton_of_pencils : ℕ := 10
def cartons_of_markers : ℕ := 10
def boxes_per_carton_of_markers : ℕ := 5
def cost_per_carton_of_markers : ℕ := 4
def total_spent : ℕ := 600

-- Variable to define cost per box of pencils
variable (P : ℝ)

-- Main theorem to prove
theorem cost_of_each_box_of_pencils :
  cartons_of_pencils * boxes_per_carton_of_pencils * P + 
  cartons_of_markers * cost_per_carton_of_markers = total_spent → 
  P = 2.80 :=
by
  sorry

end cost_of_each_box_of_pencils_l145_145311


namespace find_k_l145_145577

-- Definition of vectors a and b
def vec_a (k : ℝ) : ℝ × ℝ := (-1, k)
def vec_b : ℝ × ℝ := (3, 1)

-- Definition of dot product
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Property of perpendicular vectors (dot product is zero)
def is_perpendicular (u v : ℝ × ℝ) : Prop := dot_product u v = 0

-- Problem statement
theorem find_k (k : ℝ) :
  is_perpendicular (vec_a k) (vec_a k) →
  (k = -2 ∨ k = 1) :=
sorry

end find_k_l145_145577


namespace find_245th_digit_in_decimal_rep_of_13_div_17_l145_145914

-- Definition of the repeating sequence for the fractional division
def repeating_sequence_13_div_17 : List Char := ['7', '6', '4', '7', '0', '5', '8', '8', '2', '3', '5', '2', '9', '4', '1', '1']

-- Period of the repeating sequence
def period : ℕ := 16

-- Function to find the n-th digit in a repeating sequence
def nth_digit_in_repeating_sequence (seq : List Char) (period : ℕ) (n : ℕ) : Char :=
  seq.get! ((n - 1) % period)

-- Hypothesis: The repeating sequence of 13/17 and its period
axiom repeating_sequence_period : repeating_sequence_13_div_17.length = period

-- The theorem to prove
theorem find_245th_digit_in_decimal_rep_of_13_div_17 : nth_digit_in_repeating_sequence repeating_sequence_13_div_17 period 245 = '7' := 
  by
  sorry

end find_245th_digit_in_decimal_rep_of_13_div_17_l145_145914


namespace no_integer_solutions_l145_145478

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), x^3 + 21 * y^2 + 5 = 0 :=
by {
  sorry
}

end no_integer_solutions_l145_145478


namespace inequality_proof_l145_145959

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) :
    a * (b^2 + c^2) + b * (c^2 + a^2) ≥ 4 * a * b * c :=
by
  sorry

end inequality_proof_l145_145959


namespace geometric_sequence_a8_l145_145172

theorem geometric_sequence_a8 (a : ℕ → ℝ) (q : ℝ) 
  (h₁ : a 3 = 3)
  (h₂ : a 6 = 24)
  (h₃ : ∀ n, a (n + 1) = a n * q) : 
  a 8 = 96 :=
by
  sorry

end geometric_sequence_a8_l145_145172


namespace price_difference_correct_l145_145978

-- Define the list price of Camera Y
def list_price : ℚ := 52.50

-- Define the discount at Mega Deals
def mega_deals_discount : ℚ := 12

-- Define the discount rate at Budget Buys
def budget_buys_discount_rate : ℚ := 0.30

-- Calculate the sale prices
def mega_deals_price : ℚ := list_price - mega_deals_discount
def budget_buys_price : ℚ := (1 - budget_buys_discount_rate) * list_price

-- Calculate the price difference in dollars and convert to cents
def price_difference_in_cents : ℚ := (mega_deals_price - budget_buys_price) * 100

-- Theorem to prove the computed price difference in cents equals 375
theorem price_difference_correct : price_difference_in_cents = 375 := by
  sorry

end price_difference_correct_l145_145978


namespace mark_donates_cans_of_soup_l145_145370

theorem mark_donates_cans_of_soup:
  let n_shelters := 6
  let p_per_shelter := 30
  let c_per_person := 10
  let total_people := n_shelters * p_per_shelter
  let total_cans := total_people * c_per_person
  total_cans = 1800 :=
by sorry

end mark_donates_cans_of_soup_l145_145370


namespace locus_of_P_l145_145206

variables {x y : ℝ}
variables {x0 y0 : ℝ}

-- The initial ellipse equation
def ellipse (x y : ℝ) : Prop :=
  x^2 / 20 + y^2 / 16 = 1

-- Point M is on the ellipse
def point_M (x0 y0 : ℝ) : Prop :=
  ellipse x0 y0

-- The equation of P, symmetric to transformations applied to point Q derived from M
theorem locus_of_P 
  (hx0 : x0^2 / 20 + y0^2 / 16 = 1) :
  ∃ x y, (x^2 / 20 + y^2 / 36 = 1) ∧ y ≠ 0 :=
sorry

end locus_of_P_l145_145206


namespace three_digit_numbers_with_repeated_digits_l145_145711

theorem three_digit_numbers_with_repeated_digits :
  let total_three_digit_numbers := 900
  let without_repeats := 9 * 9 * 8
  total_three_digit_numbers - without_repeats = 252 := by
{
  let total_three_digit_numbers := 900
  let without_repeats := 9 * 9 * 8
  show total_three_digit_numbers - without_repeats = 252
  sorry
}

end three_digit_numbers_with_repeated_digits_l145_145711


namespace constant_term_expansion_l145_145893

theorem constant_term_expansion (r : Nat) (h : 12 - 3 * r = 0) :
  (Nat.choose 6 r) * 2^r = 240 :=
sorry

end constant_term_expansion_l145_145893


namespace sum_of_roots_cubic_l145_145243

theorem sum_of_roots_cubic :
  let a := 3
  let b := 7
  let c := -12
  let d := -4
  let roots_sum := -(b / a)
  roots_sum = -2.33 :=
by
  sorry

end sum_of_roots_cubic_l145_145243


namespace whale_plankton_feeding_frenzy_l145_145472

theorem whale_plankton_feeding_frenzy
  (x y : ℕ)
  (h1 : x + 5 * y = 54)
  (h2 : 9 * x + 36 * y = 450) :
  y = 4 :=
sorry

end whale_plankton_feeding_frenzy_l145_145472


namespace represent_same_function_l145_145287

noncomputable def f1 (x : ℝ) : ℝ := (x^3 + x) / (x^2 + 1)
def f2 (x : ℝ) : ℝ := x

theorem represent_same_function : ∀ x : ℝ, f1 x = f2 x := 
by
  sorry

end represent_same_function_l145_145287


namespace g_at_12_l145_145432

def g (n : ℤ) : ℤ := n^2 + 2*n + 23

theorem g_at_12 : g 12 = 191 := by
  -- proof skipped
  sorry

end g_at_12_l145_145432


namespace total_chairs_in_canteen_l145_145637

theorem total_chairs_in_canteen 
    (round_tables : ℕ) 
    (chairs_per_round_table : ℕ) 
    (rectangular_tables : ℕ) 
    (chairs_per_rectangular_table : ℕ) 
    (square_tables : ℕ) 
    (chairs_per_square_table : ℕ) 
    (extra_chairs : ℕ) 
    (h1 : round_tables = 3)
    (h2 : chairs_per_round_table = 6)
    (h3 : rectangular_tables = 4)
    (h4 : chairs_per_rectangular_table = 7)
    (h5 : square_tables = 2)
    (h6 : chairs_per_square_table = 4)
    (h7 : extra_chairs = 5) :
    round_tables * chairs_per_round_table +
    rectangular_tables * chairs_per_rectangular_table +
    square_tables * chairs_per_square_table +
    extra_chairs = 59 := by
  sorry

end total_chairs_in_canteen_l145_145637


namespace find_letter_l145_145720

def consecutive_dates (A B C D E F G : ℕ) : Prop :=
  B = A + 1 ∧ C = A + 2 ∧ D = A + 3 ∧ E = A + 4 ∧ F = A + 5 ∧ G = A + 6

theorem find_letter (A B C D E F G : ℕ) 
  (h_consecutive : consecutive_dates A B C D E F G) 
  (h_condition : ∃ y, (B + y = 2 * A + 6)) :
  y = F :=
by
  sorry

end find_letter_l145_145720


namespace power_mod_result_l145_145042

theorem power_mod_result :
  9^1002 % 50 = 1 := by
  sorry

end power_mod_result_l145_145042


namespace largest_share_received_l145_145981

noncomputable def largest_share (total_profit : ℝ) (ratio : List ℝ) : ℝ :=
  let total_parts := ratio.foldl (· + ·) 0
  let part_value := total_profit / total_parts
  let max_part := ratio.foldl max 0
  max_part * part_value

theorem largest_share_received
  (total_profit : ℝ)
  (h_total_profit : total_profit = 42000)
  (ratio : List ℝ)
  (h_ratio : ratio = [2, 3, 4, 4, 6]) :
  largest_share total_profit ratio = 12600 :=
by
  sorry

end largest_share_received_l145_145981


namespace students_called_in_sick_l145_145643

-- Conditions
def total_cupcakes : ℕ := 2 * 12 + 6
def total_people : ℕ := 27 + 1 + 1
def cupcakes_left : ℕ := 4
def cupcakes_given_out : ℕ := total_cupcakes - cupcakes_left

-- Statement to prove
theorem students_called_in_sick : total_people - cupcakes_given_out = 3 := by
  -- The proof steps would be implemented here
  sorry

end students_called_in_sick_l145_145643


namespace maria_bottles_proof_l145_145345

theorem maria_bottles_proof 
    (initial_bottles : ℕ)
    (drank_bottles : ℕ)
    (current_bottles : ℕ)
    (bought_bottles : ℕ) 
    (h1 : initial_bottles = 14)
    (h2 : drank_bottles = 8)
    (h3 : current_bottles = 51)
    (h4 : current_bottles = initial_bottles - drank_bottles + bought_bottles) :
  bought_bottles = 45 :=
by
  sorry

end maria_bottles_proof_l145_145345


namespace part1_part2_l145_145152

def f (x : ℝ) : ℝ := abs (x + 2) - 2 * abs (x - 1)

theorem part1 : { x : ℝ | f x ≥ -2 } = { x : ℝ | -2/3 ≤ x ∧ x ≤ 6 } :=
by
  sorry

theorem part2 (a : ℝ) :
  (∀ x ≥ a, f x ≤ x - a) ↔ a ≤ -2 ∨ a ≥ 4 :=
by
  sorry

end part1_part2_l145_145152


namespace max_value_of_expression_l145_145524

theorem max_value_of_expression (x y z : ℝ) (h : 3 * x + 4 * y + 2 * z = 12) :
  x^2 * y + x^2 * z + y * z^2 ≤ 3 := sorry

end max_value_of_expression_l145_145524


namespace sum_of_first_n_terms_geom_sequence_l145_145121

theorem sum_of_first_n_terms_geom_sequence (a₁ q : ℚ) (S : ℕ → ℚ)
  (h : ∀ n, S n = a₁ * (1 - q^n) / (1 - q))
  (h_ratio : S 4 / S 2 = 3) :
  S 6 / S 4 = 7 / 3 :=
by
  sorry

end sum_of_first_n_terms_geom_sequence_l145_145121


namespace y_real_for_all_x_l145_145788

theorem y_real_for_all_x (x : ℝ) : ∃ y : ℝ, 9 * y^2 + 3 * x * y + x - 3 = 0 :=
by
  sorry

end y_real_for_all_x_l145_145788


namespace automobile_travel_distance_l145_145379

theorem automobile_travel_distance
  (a r : ℝ) : 
  let feet_per_yard := 3
  let seconds_per_minute := 60
  let travel_feet := a / 4
  let travel_seconds := 2 * r
  let rate_yards_per_second := (travel_feet / travel_seconds) / feet_per_yard
  let total_seconds := 10 * seconds_per_minute
  let total_yards := rate_yards_per_second * total_seconds
  total_yards = 25 * a / r := by
  sorry

end automobile_travel_distance_l145_145379


namespace solve_f_l145_145169

open Nat

theorem solve_f (f : ℕ → ℕ) (h : ∀ n : ℕ, f (f n) + f n = 2 * n + 3) : f 1993 = 1994 := by
  -- assumptions and required proof
  sorry

end solve_f_l145_145169


namespace common_factor_of_right_triangle_l145_145688

theorem common_factor_of_right_triangle (d : ℝ) 
  (h_triangle : (2*d)^2 + (4*d)^2 = (5*d)^2) 
  (h_side : 2*d = 45 ∨ 4*d = 45 ∨ 5*d = 45) : 
  d = 9 :=
sorry

end common_factor_of_right_triangle_l145_145688


namespace total_collection_l145_145558

theorem total_collection (n : ℕ) (c : ℕ) (h_n : n = 88) (h_c : c = 88) : 
  (n * c / 100 : ℚ) = 77.44 :=
by
  sorry

end total_collection_l145_145558


namespace discount_comparison_l145_145075

noncomputable def final_price (P : ℝ) (d1 d2 d3 : ℝ) : ℝ :=
  P * (1 - d1) * (1 - d2) * (1 - d3)

theorem discount_comparison (P : ℝ) (d11 d12 d13 d21 d22 d23 : ℝ) :
  P = 20000 →
  d11 = 0.25 → d12 = 0.15 → d13 = 0.10 →
  d21 = 0.30 → d22 = 0.10 → d23 = 0.10 →
  final_price P d11 d12 d13 - final_price P d21 d22 d23 = 135 :=
by
  intros
  sorry

end discount_comparison_l145_145075


namespace geometric_progression_fourth_term_l145_145124

theorem geometric_progression_fourth_term (a b c : ℝ) (r : ℝ) 
  (h1 : a = 2) (h2 : b = 2 * Real.sqrt 2) (h3 : c = 4) (h4 : r = Real.sqrt 2)
  (h5 : b = a * r) (h6 : c = b * r) :
  c * r = 4 * Real.sqrt 2 := 
sorry

end geometric_progression_fourth_term_l145_145124


namespace smallest_number_of_coins_l145_145443

theorem smallest_number_of_coins :
  ∃ (n : ℕ), (∀ (a : ℕ), 5 ≤ a ∧ a < 100 → 
    ∃ (c : ℕ → ℕ), (a = 5 * c 0 + 10 * c 1 + 25 * c 2) ∧ 
    (c 0 + c 1 + c 2 = n)) ∧ n = 9 :=
by
  sorry

end smallest_number_of_coins_l145_145443


namespace planted_fraction_l145_145513

theorem planted_fraction (a b c : ℕ) (x h : ℝ) 
  (h_right_triangle : a = 5 ∧ b = 12)
  (h_hypotenuse : c = 13)
  (h_square_dist : x = 3) : 
  (h * ((a * b) - (x^2))) / (a * b / 2) = (7 : ℝ) / 10 :=
by
  sorry

end planted_fraction_l145_145513


namespace count_negative_numbers_l145_145027

def evaluate (e : String) : Int :=
  match e with
  | "-3^2" => -9
  | "(-3)^2" => 9
  | "-(-3)" => 3
  | "-|-3|" => -3
  | _ => 0

def isNegative (n : Int) : Bool := n < 0

def countNegatives (es : List String) : Int :=
  es.map evaluate |>.filter isNegative |>.length

theorem count_negative_numbers :
  countNegatives ["-3^2", "(-3)^2", "-(-3)", "-|-3|"] = 2 :=
by
  sorry

end count_negative_numbers_l145_145027


namespace sum_of_digits_l145_145770

theorem sum_of_digits (a b : ℕ) (h1 : 4 * 100 + a * 10 + 4 + 258 = 7 * 100 + b * 10 + 2) (h2 : (7 * 100 + b * 10 + 2) % 3 = 0) :
  a + b = 4 :=
sorry

end sum_of_digits_l145_145770


namespace value_of_e_l145_145994

variable (e : ℝ)
noncomputable def eq1 : Prop :=
  ((10 * 0.3 + 2) / 4 - (3 * 0.3 - e) / 18 = (2 * 0.3 + 4) / 3)

theorem value_of_e : eq1 e → e = 6 := by
  intro h
  sorry

end value_of_e_l145_145994


namespace students_in_all_sections_is_six_l145_145088

-- Define the number of students in each section and the total.
variable (total_students : ℕ := 30)
variable (music_students : ℕ := 15)
variable (drama_students : ℕ := 18)
variable (dance_students : ℕ := 12)
variable (at_least_two_sections : ℕ := 14)

-- Define the number of students in all three sections.
def students_in_all_three_sections (total_students music_students drama_students dance_students at_least_two_sections : ℕ) : ℕ :=
  let a := 6 -- the result we want to prove
  a

-- The theorem proving that the number of students in all three sections is 6.
theorem students_in_all_sections_is_six :
  students_in_all_three_sections total_students music_students drama_students dance_students at_least_two_sections = 6 :=
by 
  sorry -- Proof is omitted

end students_in_all_sections_is_six_l145_145088


namespace solution_set_of_inequality_l145_145983

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 2 * x + 3 ≥ 0 } = {x : ℝ | -1 ≤ x ∧ x ≤ 3 } :=
sorry

end solution_set_of_inequality_l145_145983


namespace functional_relationship_maximum_profit_desired_profit_l145_145601

-- Conditions
def cost_price := 80
def y (x : ℝ) : ℝ := -2 * x + 320
def w (x : ℝ) : ℝ := (x - cost_price) * y x

-- Functional relationship
theorem functional_relationship (x : ℝ) (hx : 80 ≤ x ∧ x ≤ 160) :
  w x = -2 * x^2 + 480 * x - 25600 :=
by sorry

-- Maximizing daily profit
theorem maximum_profit :
  ∃ x, 80 ≤ x ∧ x ≤ 160 ∧ w x = 3200 ∧ (∀ y, 80 ≤ y ∧ y ≤ 160 → w y ≤ 3200) ∧ x = 120 :=
by sorry

-- Desired profit of 2400 dollars
theorem desired_profit (w_desired : ℝ) (hw : w_desired = 2400) :
  ∃ x, 80 ≤ x ∧ x ≤ 160 ∧ w x = w_desired ∧ x = 100 :=
by sorry

end functional_relationship_maximum_profit_desired_profit_l145_145601


namespace find_S_30_l145_145724

variable (S : ℕ → ℚ)
variable (a : ℕ → ℚ)
variable (d : ℚ)

-- Definitions based on conditions
def arithmetic_sum (n : ℕ) : ℚ := (n / 2) * (a 1 + a n)
def a_n (n : ℕ) : ℚ := a 1 + (n - 1) * d

-- Given conditions
axiom h1 : S 10 = 20
axiom h2 : S 20 = 15

-- Required Proof (the final statement to be proven)
theorem find_S_30 : S 30 = -15 := sorry

end find_S_30_l145_145724


namespace total_pencils_l145_145699

/-- The conditions defining the number of pencils Sarah buys each day. -/
def pencils_monday : ℕ := 20
def pencils_tuesday : ℕ := 18
def pencils_wednesday : ℕ := 3 * pencils_tuesday

/-- The hypothesis that the total number of pencils bought by Sarah is 92. -/
theorem total_pencils : pencils_monday + pencils_tuesday + pencils_wednesday = 92 :=
by
  -- calculations skipped
  sorry

end total_pencils_l145_145699


namespace pseudo_symmetry_abscissa_l145_145759

noncomputable def f (x : ℝ) : ℝ := x^2 - 6*x + 4 * Real.log x

theorem pseudo_symmetry_abscissa :
  ∃ x0 : ℝ, x0 = Real.sqrt 2 ∧
    (∀ x : ℝ, x ≠ x0 → (f x - ((2*x0 + 4/x0 - 6)*(x - x0) + x0^2 - 6*x0 + 4*Real.log x0)) / (x - x0) > 0) :=
sorry

end pseudo_symmetry_abscissa_l145_145759


namespace find_f2_l145_145046

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^5 + a * x^3 + b * x + 1

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -8 :=
by
  sorry

end find_f2_l145_145046


namespace find_x_l145_145268

def vector := (ℝ × ℝ)

def a (x : ℝ) : vector := (x, 2)
def b : vector := (1, -1)

-- Dot product of two vectors
def dot_product (v1 v2 : vector) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Orthogonality condition rewritten in terms of dot product
def orthogonal (v1 v2 : vector) : Prop := dot_product v1 v2 = 0

-- Main theorem to prove
theorem find_x (x : ℝ) (h : orthogonal ((a x).1 - b.1, (a x).2 - b.2) b) : x = 4 :=
by sorry

end find_x_l145_145268


namespace volleyball_team_l145_145062

theorem volleyball_team :
  let total_combinations := (Nat.choose 15 6)
  let without_triplets := (Nat.choose 12 6)
  total_combinations - without_triplets = 4081 :=
by
  -- Definitions based on the problem conditions
  let team_size := 15
  let starters := 6
  let triplets := 3
  let total_combinations := Nat.choose team_size starters
  let without_triplets := Nat.choose (team_size - triplets) starters
  -- Identify the proof goal
  have h : total_combinations - without_triplets = 4081 := sorry
  exact h

end volleyball_team_l145_145062


namespace solvable_consecutive_integers_solvable_consecutive_even_integers_not_solvable_consecutive_odd_integers_l145_145144

-- Definitions only directly appearing in the conditions problem
def consecutive_integers (x y z : ℤ) : Prop := x = y - 1 ∧ z = y + 1
def consecutive_even_integers (x y z : ℤ) : Prop := x = y - 2 ∧ z = y + 2 ∧ y % 2 = 0
def consecutive_odd_integers (x y z : ℤ) : Prop := x = y - 2 ∧ z = y + 2 ∧ y % 2 = 1

-- Problem Statements
theorem solvable_consecutive_integers : ∃ (x y z : ℤ), x + y + z = 72 ∧ consecutive_integers x y z :=
sorry

theorem solvable_consecutive_even_integers : ∃ (x y z : ℤ), x + y + z = 72 ∧ consecutive_even_integers x y z :=
sorry

theorem not_solvable_consecutive_odd_integers : ¬ ∃ (x y z : ℤ), x + y + z = 72 ∧ consecutive_odd_integers x y z :=
sorry

end solvable_consecutive_integers_solvable_consecutive_even_integers_not_solvable_consecutive_odd_integers_l145_145144


namespace num_both_sports_l145_145431

def num_people := 310
def num_tennis := 138
def num_baseball := 255
def num_no_sport := 11

theorem num_both_sports : (num_tennis + num_baseball - (num_people - num_no_sport)) = 94 :=
by 
-- leave the proof out for now
sorry

end num_both_sports_l145_145431


namespace rectangular_prism_sum_l145_145117

-- Definitions based on conditions
def edges := 12
def corners := 8
def faces := 6

-- Lean statement to prove question == answer given conditions.
theorem rectangular_prism_sum : edges + corners + faces = 26 := by
  sorry

end rectangular_prism_sum_l145_145117


namespace percentage_loss_calculation_l145_145634

theorem percentage_loss_calculation
  (initial_cost_euro : ℝ)
  (retail_price_dollars : ℝ)
  (exchange_rate_initial : ℝ)
  (discount1 : ℝ)
  (discount2 : ℝ)
  (sales_tax : ℝ)
  (exchange_rate_new : ℝ)
  (final_sale_price_dollars : ℝ) :
  initial_cost_euro = 800 ∧
  retail_price_dollars = 900 ∧
  exchange_rate_initial = 1.1 ∧
  discount1 = 0.10 ∧
  discount2 = 0.15 ∧
  sales_tax = 0.10 ∧
  exchange_rate_new = 1.5 ∧
  final_sale_price_dollars = (retail_price_dollars * (1 - discount1) * (1 - discount2) * (1 + sales_tax)) →
  ((initial_cost_euro - final_sale_price_dollars / exchange_rate_new) / initial_cost_euro) * 100 = 36.89 := by
  sorry

end percentage_loss_calculation_l145_145634


namespace phillip_remaining_amount_l145_145161

-- Define the initial amount of money
def initial_amount : ℕ := 95

-- Define the amounts spent on various items
def amount_spent_on_oranges : ℕ := 14
def amount_spent_on_apples : ℕ := 25
def amount_spent_on_candy : ℕ := 6

-- Calculate the total amount spent
def total_spent : ℕ := amount_spent_on_oranges + amount_spent_on_apples + amount_spent_on_candy

-- Calculate the remaining amount of money
def remaining_amount : ℕ := initial_amount - total_spent

-- Statement to be proved
theorem phillip_remaining_amount : remaining_amount = 50 :=
by
  sorry

end phillip_remaining_amount_l145_145161


namespace center_of_large_hexagon_within_small_hexagon_l145_145198

-- Define a structure for a regular hexagon with the necessary properties
structure RegularHexagon (α : Type) [LinearOrderedField α] :=
  (center : α × α)      -- Coordinates of the center
  (side_length : α)      -- Length of the side

-- Define the conditions: two regular hexagons with specific side length relationship
variables {α : Type} [LinearOrderedField α]
def hexagon_large : RegularHexagon α := 
  {center := (0, 0), side_length := 2}

def hexagon_small : RegularHexagon α := 
  {center := (0, 0), side_length := 1}

-- The theorem to prove
theorem center_of_large_hexagon_within_small_hexagon (hl : RegularHexagon α) (hs : RegularHexagon α) 
  (hc : hs.side_length = hl.side_length / 2) : (hl.center = hs.center) → 
  (∀ (x y : α × α), x = hs.center → (∃ r, y = hl.center → (y.1 - x.1) ^ 2 + (y.2 - x.2) ^ 2 < r ^ 2)) :=
by sorry

end center_of_large_hexagon_within_small_hexagon_l145_145198


namespace calculate_expression_l145_145783

theorem calculate_expression : 
  (1 / 2) ^ (-2: ℤ) - 3 * Real.tan (Real.pi / 6) - abs (Real.sqrt 3 - 2) = 2 := 
by
  sorry

end calculate_expression_l145_145783


namespace find_n_tan_eq_l145_145438

theorem find_n_tan_eq (n : ℝ) (h1 : -180 < n) (h2 : n < 180) (h3 : Real.tan (n * Real.pi / 180) = Real.tan (678 * Real.pi / 180)) : 
  n = 138 := 
sorry

end find_n_tan_eq_l145_145438


namespace books_returned_wednesday_correct_l145_145325

def initial_books : Nat := 250
def books_taken_out_Tuesday : Nat := 120
def books_taken_out_Thursday : Nat := 15
def books_remaining_after_Thursday : Nat := 150

def books_after_tuesday := initial_books - books_taken_out_Tuesday
def books_before_thursday := books_remaining_after_Thursday + books_taken_out_Thursday
def books_returned_wednesday := books_before_thursday - books_after_tuesday

theorem books_returned_wednesday_correct : books_returned_wednesday = 35 := by
  sorry

end books_returned_wednesday_correct_l145_145325


namespace number_of_subsets_l145_145792

theorem number_of_subsets (x y : Type) :  ∃ s : Finset (Finset Type), s.card = 4 := 
sorry

end number_of_subsets_l145_145792


namespace fernanda_savings_calculation_l145_145778

theorem fernanda_savings_calculation :
  ∀ (aryan_debt kyro_debt aryan_payment kyro_payment savings total_savings : ℝ),
    aryan_debt = 1200 ∧
    aryan_debt = 2 * kyro_debt ∧
    aryan_payment = (60 / 100) * aryan_debt ∧
    kyro_payment = (80 / 100) * kyro_debt ∧
    savings = 300 ∧
    total_savings = savings + aryan_payment + kyro_payment →
    total_savings = 1500 := by
    sorry

end fernanda_savings_calculation_l145_145778


namespace jaco_payment_l145_145510

theorem jaco_payment :
  let cost_shoes : ℝ := 74
  let cost_socks : ℝ := 2 * 2
  let cost_bag : ℝ := 42
  let total_cost_before_discount : ℝ := cost_shoes + cost_socks + cost_bag
  let discount_threshold : ℝ := 100
  let discount_rate : ℝ := 0.10
  let amount_exceeding_threshold : ℝ := total_cost_before_discount - discount_threshold
  let discount : ℝ := if amount_exceeding_threshold > 0 then discount_rate * amount_exceeding_threshold else 0
  let final_amount : ℝ := total_cost_before_discount - discount
  final_amount = 118 :=
by
  sorry

end jaco_payment_l145_145510


namespace rectangle_to_square_area_ratio_is_24_25_l145_145149

noncomputable def rectangle_to_square_area_ratio
  (s : ℝ) -- length of side of square S
  (longer_side : ℝ := 1.2 * s) -- longer side of rectangle R
  (shorter_side : ℝ := 0.8 * s) -- shorter side of rectangle R
  (area_R : ℝ := longer_side * shorter_side) -- area of rectangle R
  (area_S : ℝ := s^2) -- area of square S
  : ℝ := 
  area_R / area_S

theorem rectangle_to_square_area_ratio_is_24_25 
  (s : ℝ)
  : rectangle_to_square_area_ratio s = 24 / 25 :=
by 
  sorry

end rectangle_to_square_area_ratio_is_24_25_l145_145149


namespace min_max_values_l145_145239

theorem min_max_values (x1 x2 x3 : ℝ) (h1 : x1 ≥ 0) (h2 : x2 ≥ 0) (h3 : x3 ≥ 0) (h_sum : x1 + x2 + x3 = 1) :
  1 ≤ (x1 + 3*x2 + 5*x3) * (x1 + x2 / 3 + x3 / 5) ∧ (x1 + 3*x2 + 5*x3) * (x1 + x2 / 3 + x3 / 5) ≤ 9/5 :=
by sorry

end min_max_values_l145_145239


namespace minimum_value_of_expression_l145_145332

theorem minimum_value_of_expression (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
    ∃ (c : ℝ), (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → x^3 + y^3 - 5 * x * y ≥ c) ∧ c = -125 / 27 :=
by
  sorry

end minimum_value_of_expression_l145_145332


namespace expected_people_with_condition_l145_145610

noncomputable def proportion_of_condition := 1 / 3
def total_population := 450

theorem expected_people_with_condition :
  (proportion_of_condition * total_population) = 150 := by
  sorry

end expected_people_with_condition_l145_145610


namespace necessary_and_sufficient_condition_x_eq_1_l145_145812

theorem necessary_and_sufficient_condition_x_eq_1
    (x : ℝ) :
    (x = 1 ↔ x^2 - 2 * x + 1 = 0) :=
sorry

end necessary_and_sufficient_condition_x_eq_1_l145_145812


namespace general_form_of_line_l_l145_145615

-- Define the point
def pointA : ℝ × ℝ := (1, 2)

-- Define the normal vector
def normalVector : ℝ × ℝ := (1, -3)

-- Define the general form equation
def generalFormEq (x y : ℝ) : Prop := x - 3 * y + 5 = 0

-- Statement to prove
theorem general_form_of_line_l (x y : ℝ) (h_pointA : pointA = (1, 2)) (h_normalVector : normalVector = (1, -3)) :
  generalFormEq x y :=
sorry

end general_form_of_line_l_l145_145615


namespace more_cats_than_dogs_l145_145640

theorem more_cats_than_dogs (total_animals : ℕ) (cats : ℕ) (h1 : total_animals = 60) (h2 : cats = 40) : (cats - (total_animals - cats)) = 20 :=
by 
  sorry

end more_cats_than_dogs_l145_145640


namespace volume_of_water_cylinder_l145_145203

theorem volume_of_water_cylinder :
  let r := 5
  let h := 10
  let depth := 3
  let θ := Real.arccos (3 / 5)
  let sector_area := (2 * θ) / (2 * Real.pi) * Real.pi * r^2
  let triangle_area := r * (2 * r * Real.sin θ)
  let water_surface_area := sector_area - triangle_area
  let volume := h * water_surface_area
  volume = 232.6 * Real.pi - 160 :=
by
  sorry

end volume_of_water_cylinder_l145_145203


namespace divisor_of_1076_plus_least_addend_l145_145327

theorem divisor_of_1076_plus_least_addend (a d : ℕ) (h1 : 1076 + a = 1081) (h2 : d ∣ 1081) (ha : a = 5) : d = 13 := 
sorry

end divisor_of_1076_plus_least_addend_l145_145327


namespace sqrt_expression_l145_145622

theorem sqrt_expression : 
  (Real.sqrt 75 + Real.sqrt 8 - Real.sqrt 18 - Real.sqrt 6 * Real.sqrt 2 = 3 * Real.sqrt 3 - Real.sqrt 2) := 
by 
  sorry

end sqrt_expression_l145_145622


namespace LaShawn_twice_Kymbrea_after_25_months_l145_145649

theorem LaShawn_twice_Kymbrea_after_25_months : 
  ∀ (x : ℕ), (10 + 6 * x = 2 * (30 + 2 * x)) → x = 25 :=
by
  intro x
  sorry

end LaShawn_twice_Kymbrea_after_25_months_l145_145649


namespace ratio_horizontal_to_checkered_l145_145037

/--
In a cafeteria, 7 people are wearing checkered shirts, while the rest are wearing vertical stripes
and horizontal stripes. There are 40 people in total, and 5 of them are wearing vertical stripes.
What is the ratio of the number of people wearing horizontal stripes to the number of people wearing
checkered shirts?
-/
theorem ratio_horizontal_to_checkered
  (total_people : ℕ)
  (checkered_people : ℕ)
  (vertical_people : ℕ)
  (horizontal_people : ℕ)
  (ratio : ℕ)
  (h_total : total_people = 40)
  (h_checkered : checkered_people = 7)
  (h_vertical : vertical_people = 5)
  (h_horizontal : horizontal_people = total_people - checkered_people - vertical_people)
  (h_ratio : ratio = horizontal_people / checkered_people) :
  ratio = 4 :=
by
  sorry

end ratio_horizontal_to_checkered_l145_145037


namespace initial_population_l145_145290

theorem initial_population (rate_decrease : ℝ) (population_after_2_years : ℝ) (P : ℝ) : 
  rate_decrease = 0.1 → 
  population_after_2_years = 8100 → 
  ((1 - rate_decrease) ^ 2) * P = population_after_2_years → 
  P = 10000 :=
by
  intros h1 h2 h3
  sorry

end initial_population_l145_145290


namespace articles_profit_l145_145011

variable {C S : ℝ}

theorem articles_profit (h1 : 20 * C = x * S) (h2 : S = 1.25 * C) : x = 16 :=
by
  sorry

end articles_profit_l145_145011


namespace neither_necessary_nor_sufficient_condition_l145_145591

def red_balls := 5
def yellow_balls := 3
def white_balls := 2
def total_balls := red_balls + yellow_balls + white_balls

def event_A_occurs := ∃ (r : ℕ) (y : ℕ), (r ≤ red_balls) ∧ (y ≤ yellow_balls) ∧ (r = 1) ∧ (y = 1)
def event_B_occurs := ∃ (x y : ℕ), (x ≤ total_balls) ∧ (y ≤ total_balls) ∧ (x ≠ y)

theorem neither_necessary_nor_sufficient_condition :
  ¬(¬event_A_occurs → ¬event_B_occurs) ∧ ¬(¬event_B_occurs → ¬event_A_occurs) := 
sorry

end neither_necessary_nor_sufficient_condition_l145_145591


namespace flour_needed_l145_145468

-- Define the given conditions
def F_total : ℕ := 9
def F_added : ℕ := 3

-- State the main theorem to be proven
theorem flour_needed : (F_total - F_added) = 6 := by
  sorry -- Placeholder for the proof

end flour_needed_l145_145468


namespace james_initial_friends_l145_145668

theorem james_initial_friends (x : ℕ) (h1 : 19 = x - 2 + 1) : x = 20 :=
  by sorry

end james_initial_friends_l145_145668


namespace point_c_in_second_quadrant_l145_145248

-- Definitions for the points
def PointA : ℝ × ℝ := (1, 2)
def PointB : ℝ × ℝ := (-1, -2)
def PointC : ℝ × ℝ := (-1, 2)
def PointD : ℝ × ℝ := (1, -2)

-- Definition of the second quadrant condition
def in_second_quadrant (p : ℝ × ℝ) : Prop :=
p.1 < 0 ∧ p.2 > 0

theorem point_c_in_second_quadrant : in_second_quadrant PointC :=
sorry

end point_c_in_second_quadrant_l145_145248


namespace geometric_sum_a4_a6_l145_145456

-- Definitions based on the conditions
def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sum_a4_a6 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h_pos : ∀ n, a n > 0) 
(h_cond : a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100) : a 4 + a 6 = 10 :=
by
  sorry

end geometric_sum_a4_a6_l145_145456


namespace meat_per_deer_is_200_l145_145612

namespace wolf_pack

def number_hunting_wolves : ℕ := 4
def number_additional_wolves : ℕ := 16
def meat_needed_per_day : ℕ := 8
def days : ℕ := 5

def total_wolves : ℕ := number_hunting_wolves + number_additional_wolves

def total_meat_needed : ℕ := total_wolves * meat_needed_per_day * days

def number_deer : ℕ := number_hunting_wolves

def meat_per_deer : ℕ := total_meat_needed / number_deer

theorem meat_per_deer_is_200 : meat_per_deer = 200 := by
  sorry

end wolf_pack

end meat_per_deer_is_200_l145_145612


namespace ribbon_tape_length_l145_145155

theorem ribbon_tape_length
  (one_ribbon: ℝ)
  (remaining_cm: ℝ)
  (num_ribbons: ℕ)
  (total_used: ℝ)
  (remaining_meters: remaining_cm = 0.50)
  (ribbon_meter: one_ribbon = 0.84)
  (ribbons_made: num_ribbons = 10)
  (used_len: total_used = one_ribbon * num_ribbons):
  total_used + 0.50 = 8.9 :=
by
  sorry

end ribbon_tape_length_l145_145155


namespace greatest_three_digit_multiple_of_17_l145_145218

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l145_145218


namespace inequality_not_satisfied_integer_values_count_l145_145188

theorem inequality_not_satisfied_integer_values_count :
  ∃ (n : ℕ), n = 5 ∧ ∀ (x : ℤ), 3 * x^2 + 17 * x + 20 ≤ 25 → x ∈ [-4, -3, -2, -1, 0] :=
  sorry

end inequality_not_satisfied_integer_values_count_l145_145188


namespace exact_time_is_3_07_27_l145_145103

theorem exact_time_is_3_07_27 (t : ℝ) (H1 : t > 0) (H2 : t < 60) 
(H3 : 6 * (t + 8) = 89 + 0.5 * t) : t = 7 + 27/60 :=
by
  sorry

end exact_time_is_3_07_27_l145_145103


namespace problem_l145_145501

theorem problem (a b c d e : ℤ) 
  (h1 : a - b + c - e = 7)
  (h2 : b - c + d + e = 8)
  (h3 : c - d + a - e = 4)
  (h4 : d - a + b + e = 3) :
  a + b + c + d + e = 22 := by
  sorry

end problem_l145_145501


namespace leak_empties_tank_in_24_hours_l145_145140

theorem leak_empties_tank_in_24_hours (A L : ℝ) (hA : A = 1 / 8) (h_comb : A - L = 1 / 12) : 1 / L = 24 :=
by
  -- Proof will be here
  sorry

end leak_empties_tank_in_24_hours_l145_145140


namespace sale_in_fourth_month_l145_145543

-- Given conditions
def sales_first_month : ℕ := 5266
def sales_second_month : ℕ := 5768
def sales_third_month : ℕ := 5922
def sales_sixth_month : ℕ := 4937
def required_average_sales : ℕ := 5600
def number_of_months : ℕ := 6

-- Sum of the first, second, third, and sixth month's sales
def total_sales_without_fourth_fifth : ℕ := sales_first_month + sales_second_month + sales_third_month + sales_sixth_month

-- Total sales required to achieve the average required
def required_total_sales : ℕ := required_average_sales * number_of_months

-- The sale in the fourth month should be calculated as follows
def sales_fourth_month : ℕ := required_total_sales - total_sales_without_fourth_fifth

-- Proof statement
theorem sale_in_fourth_month :
  sales_fourth_month = 11707 := by
  sorry

end sale_in_fourth_month_l145_145543


namespace find_teacher_age_l145_145269

theorem find_teacher_age (S T : ℕ) (h1 : S / 19 = 20) (h2 : (S + T) / 20 = 21) : T = 40 :=
sorry

end find_teacher_age_l145_145269


namespace smallest_y_value_l145_145008

theorem smallest_y_value (y : ℝ) : 3 * y ^ 2 + 33 * y - 105 = y * (y + 16) → y = -21 / 2 ∨ y = 5 := sorry

end smallest_y_value_l145_145008


namespace total_revenue_calculation_l145_145099

variables (a b : ℕ) -- Assuming a and b are natural numbers representing the number of newspapers

-- Define the prices
def purchase_price_per_copy : ℝ := 0.4
def selling_price_per_copy : ℝ := 0.5
def return_price_per_copy : ℝ := 0.2

-- Define the revenue and cost calculations
def revenue_from_selling (b : ℕ) : ℝ := selling_price_per_copy * b
def revenue_from_returning (a b : ℕ) : ℝ := return_price_per_copy * (a - b)
def cost_of_purchasing (a : ℕ) : ℝ := purchase_price_per_copy * a

-- Define the total revenue
def total_revenue (a b : ℕ) : ℝ :=
  revenue_from_selling b + revenue_from_returning a b - cost_of_purchasing a

-- The theorem we need to prove
theorem total_revenue_calculation (a b : ℕ) :
  total_revenue a b = 0.3 * b - 0.2 * a :=
by
  sorry

end total_revenue_calculation_l145_145099


namespace consecutive_probability_l145_145380

-- Define the total number of ways to choose 2 episodes out of 6
def total_combinations : ℕ := Nat.choose 6 2

-- Define the number of ways to choose consecutive episodes
def consecutive_combinations : ℕ := 5

-- Define the probability of choosing consecutive episodes
def probability_of_consecutive : ℚ := consecutive_combinations / total_combinations

-- Theorem stating that the calculated probability should equal 1/3
theorem consecutive_probability : probability_of_consecutive = 1 / 3 :=
by
  sorry

end consecutive_probability_l145_145380


namespace maximum_volume_prism_l145_145322

-- Define the conditions
variables {l w h : ℝ}
axiom area_sum_eq : 2 * h * l + l * w = 30

-- Define the volume of the prism
def volume (l w h : ℝ) : ℝ := l * w * h

-- Statement to be proved
theorem maximum_volume_prism : 
  (∃ l w h : ℝ, 2 * h * l + l * w = 30 ∧ 
  ∀ u v t : ℝ, 2 * t * u + u * v = 30 → l * w * h ≥ u * v * t) → volume l w h = 112.5 :=
by
  sorry

end maximum_volume_prism_l145_145322


namespace melt_brown_fabric_scientific_notation_l145_145840

theorem melt_brown_fabric_scientific_notation :
  0.000156 = 1.56 * 10^(-4) :=
sorry

end melt_brown_fabric_scientific_notation_l145_145840


namespace simplify_and_evaluate_expression_l145_145687

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = Real.sqrt 2 - 3) : 
  (a - 3) / (a^2 + 6 * a + 9) / (1 - 6 / (a + 3)) = Real.sqrt 2 / 2 :=
by sorry

end simplify_and_evaluate_expression_l145_145687


namespace original_number_l145_145055

theorem original_number (x y : ℝ) (h1 : 10 * x + 22 * y = 780) (h2 : y = 37.66666666666667) : 
  x + y = 32.7 := 
sorry

end original_number_l145_145055


namespace curve_tangents_intersection_l145_145375

theorem curve_tangents_intersection (a : ℝ) :
  (∃ x₀ y₀, y₀ = Real.exp x₀ ∧ y₀ = (x₀ + a)^2 ∧ Real.exp x₀ = 2 * (x₀ + a)) → a = 2 - Real.log 4 :=
by
  sorry

end curve_tangents_intersection_l145_145375


namespace anagrams_without_three_consecutive_identical_l145_145043

theorem anagrams_without_three_consecutive_identical :
  let total_anagrams := 100800
  let anagrams_with_three_A := 6720
  let anagrams_with_three_B := 6720
  let anagrams_with_three_A_and_B := 720
  let valid_anagrams := total_anagrams - anagrams_with_three_A - anagrams_with_three_B + anagrams_with_three_A_and_B
  valid_anagrams = 88080 := by
  sorry

end anagrams_without_three_consecutive_identical_l145_145043


namespace tangent_line_eq_l145_145799

theorem tangent_line_eq (x y: ℝ):
  (x^2 + y^2 = 4) → ((2, 3) = (x, y)) →
  (x = 2 ∨ 5 * x - 12 * y + 26 = 0) :=
by
  sorry

end tangent_line_eq_l145_145799


namespace MinValue_x3y2z_l145_145526

theorem MinValue_x3y2z (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h : 1/x + 1/y + 1/z = 6) : x^3 * y^2 * z ≥ 1 / 108 :=
by
  sorry

end MinValue_x3y2z_l145_145526


namespace no_rearrangement_of_power_of_two_l145_145048

theorem no_rearrangement_of_power_of_two (k n : ℕ) (hk : k > 3) (hn : n > k) : 
  ∀ m : ℕ, 
    (m.toDigits = (2^k).toDigits → m ≠ 2^n) :=
by
  sorry

end no_rearrangement_of_power_of_two_l145_145048


namespace point_in_second_quadrant_l145_145490

theorem point_in_second_quadrant (a : ℝ) : 
  ∃ q : ℕ, q = 2 ∧ (-1, a^2 + 1).1 < 0 ∧ 0 < (-1, a^2 + 1).2 :=
by
  sorry

end point_in_second_quadrant_l145_145490


namespace birds_landing_l145_145314

theorem birds_landing (initial_birds total_birds birds_landed : ℤ) 
  (h_initial : initial_birds = 12) 
  (h_total : total_birds = 20) :
  birds_landed = total_birds - initial_birds :=
by
  sorry

end birds_landing_l145_145314


namespace solve_xy_l145_145665

theorem solve_xy (x y a b : ℝ) (h1 : x * y = 2 * b) (h2 : (1 / x^2) + (1 / y^2) = a) : 
  (x + y)^2 = 4 * a * b^2 + 4 * b := 
by 
  sorry

end solve_xy_l145_145665


namespace vertex_of_quadratic_l145_145484

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := -3 * x^2 - 6 * x + 5

-- State the theorem for vertex coordinates
theorem vertex_of_quadratic :
  (∀ x : ℝ, quadratic_function (- (-6) / (2 * -3)) = quadratic_function 1)
  → (1, quadratic_function 1) = (1, 8) :=
by
  intros h
  sorry

end vertex_of_quadratic_l145_145484


namespace problem_l145_145775

theorem problem (x : ℝ) (h : x + 2 / x = 4) : - (5 * x) / (x^2 + 2) = -5 / 4 := 
sorry

end problem_l145_145775


namespace highest_score_not_necessarily_12_l145_145717

-- Define the structure of the round-robin tournament setup
structure RoundRobinTournament :=
  (teams : ℕ)
  (matches_per_team : ℕ)
  (points_win : ℕ)
  (points_loss : ℕ)
  (points_draw : ℕ)

-- Tournament conditions
def tournament : RoundRobinTournament :=
  { teams := 12,
    matches_per_team := 11,
    points_win := 2,
    points_loss := 0,
    points_draw := 1 }

-- The statement we want to prove
theorem highest_score_not_necessarily_12 (T : RoundRobinTournament) :
  ∃ team_highest_score : ℕ, team_highest_score < 12 :=
by
  -- Provide a proof here
  sorry

end highest_score_not_necessarily_12_l145_145717


namespace triangle_ratio_and_angle_l145_145569

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (sinA sinB sinC : ℝ)

theorem triangle_ratio_and_angle
  (h_triangle : a / sinA = b / sinB ∧ b / sinB = c / sinC)
  (h_sin_ratio : sinA / sinB = 5 / 7 ∧ sinB / sinC = 7 / 8) :
  (a / b = 5 / 7 ∧ b / c = 7 / 8) ∧ B = 60 :=
by
  sorry

end triangle_ratio_and_angle_l145_145569


namespace book_cost_l145_145051

theorem book_cost (b : ℝ) : (11 * b < 15) ∧ (12 * b > 16.20) → b = 1.36 :=
by
  intros h
  sorry

end book_cost_l145_145051


namespace mysterious_division_l145_145579

theorem mysterious_division (d : ℕ) : (8 * d < 1000) ∧ (7 * d < 900) → d = 124 :=
by
  intro h
  sorry

end mysterious_division_l145_145579


namespace ratio_perimeters_l145_145872

noncomputable def rectangle_length : ℝ := 3
noncomputable def rectangle_width : ℝ := 2
noncomputable def triangle_hypotenuse : ℝ := Real.sqrt ((rectangle_length / 2) ^ 2 + rectangle_width ^ 2)
noncomputable def perimeter_rectangle : ℝ := 2 * (rectangle_length + rectangle_width)
noncomputable def perimeter_rhombus : ℝ := 4 * triangle_hypotenuse

theorem ratio_perimeters (h1 : rectangle_length = 3) (h2 : rectangle_width = 2) :
  (perimeter_rectangle / perimeter_rhombus) = 1 :=
by
  /- proof would go here -/
  sorry

end ratio_perimeters_l145_145872


namespace xy_plus_one_is_perfect_square_l145_145423

theorem xy_plus_one_is_perfect_square (x y : ℕ) (h : 1 / (x : ℝ) + 1 / (y : ℝ) = 1 / (x + 2 : ℝ) + 1 / (y - 2 : ℝ)) :
  ∃ k : ℕ, xy + 1 = k^2 :=
by
  sorry

end xy_plus_one_is_perfect_square_l145_145423


namespace sanjay_homework_fraction_l145_145722

theorem sanjay_homework_fraction :
  let original := 1
  let done_on_monday := 3 / 5
  let remaining_after_monday := original - done_on_monday
  let done_on_tuesday := 1 / 3 * remaining_after_monday
  let remaining_after_tuesday := remaining_after_monday - done_on_tuesday
  remaining_after_tuesday = 4 / 15 :=
by
  -- original := 1
  -- done_on_monday := 3 / 5
  -- remaining_after_monday := 1 - 3 / 5
  -- done_on_tuesday := 1 / 3 * (1 - 3 / 5)
  -- remaining_after_tuesday := (1 - 3 / 5) - (1 / 3 * (1 - 3 / 5))
  sorry

end sanjay_homework_fraction_l145_145722


namespace olivia_hair_length_l145_145879

def emilys_hair_length (logan_hair : ℕ) : ℕ := logan_hair + 6
def kates_hair_length (emily_hair : ℕ) : ℕ := emily_hair / 2
def jacks_hair_length (kate_hair : ℕ) : ℕ := (7 * kate_hair) / 2
def olivias_hair_length (jack_hair : ℕ) : ℕ := (2 * jack_hair) / 3

theorem olivia_hair_length
  (logan_hair : ℕ)
  (h_logan : logan_hair = 20)
  (h_emily : emilys_hair_length logan_hair = logan_hair + 6)
  (h_emily_value : emilys_hair_length logan_hair = 26)
  (h_kate : kates_hair_length (emilys_hair_length logan_hair) = 13)
  (h_jack : jacks_hair_length (kates_hair_length (emilys_hair_length logan_hair)) = 45)
  (h_olivia : olivias_hair_length (jacks_hair_length (kates_hair_length (emilys_hair_length logan_hair))) = 30) :
  olivias_hair_length
    (jacks_hair_length
      (kates_hair_length (emilys_hair_length logan_hair))) = 30 := by
  sorry

end olivia_hair_length_l145_145879


namespace total_cards_l145_145629

theorem total_cards (H F B : ℕ) (hH : H = 200) (hF : F = 4 * H) (hB : B = F - 50) : H + F + B = 1750 := 
by 
  sorry

end total_cards_l145_145629


namespace cade_marbles_left_l145_145971

theorem cade_marbles_left (initial_marbles : ℕ) (given_away : ℕ) (remaining_marbles : ℕ) :
  initial_marbles = 350 → given_away = 175 → remaining_marbles = initial_marbles - given_away → remaining_marbles = 175 :=
by
  intros h_initial h_given h_remaining
  rw [h_initial, h_given] at h_remaining
  exact h_remaining

end cade_marbles_left_l145_145971


namespace intersection_points_l145_145058

theorem intersection_points :
  {p : ℝ × ℝ |
    (∃ x : ℝ, p = (x, 3*x^2 - 4*x + 2) ∧ p = (x, x^3 - 2*x^2 + 5*x - 1))} =
  {(1, 1), (3, 17)} :=
  sorry

end intersection_points_l145_145058


namespace no_set_of_9_numbers_l145_145122

theorem no_set_of_9_numbers (numbers : Finset ℕ) (median : ℕ) (max_value : ℕ) (mean : ℕ) :
  numbers.card = 9 → 
  median = 2 →
  max_value = 13 →
  mean = 7 →
  (∀ x ∈ numbers, x ≤ max_value) →
  (∃ m ∈ numbers, x ≤ median) →
  False :=
by
  sorry

end no_set_of_9_numbers_l145_145122


namespace max_diameter_min_diameter_l145_145901

-- Definitions based on problem conditions
def base_diameter : ℝ := 30
def positive_tolerance : ℝ := 0.03
def negative_tolerance : ℝ := 0.04

-- The corresponding proof problem statements in Lean 4
theorem max_diameter : base_diameter + positive_tolerance = 30.03 := sorry
theorem min_diameter : base_diameter - negative_tolerance = 29.96 := sorry

end max_diameter_min_diameter_l145_145901


namespace smaller_cuboid_length_l145_145080

-- Definitions based on conditions
def original_cuboid_volume : ℝ := 18 * 15 * 2
def smaller_cuboid_volume (L : ℝ) : ℝ := 4 * 3 * L
def smaller_cuboids_total_volume (L : ℝ) : ℝ := 7.5 * smaller_cuboid_volume L

-- Theorem statement
theorem smaller_cuboid_length :
  ∃ L : ℝ, smaller_cuboids_total_volume L = original_cuboid_volume ∧ L = 6 := 
by
  sorry

end smaller_cuboid_length_l145_145080


namespace population_in_2060_l145_145599

noncomputable def population (year : ℕ) : ℕ :=
  if h : (year - 2000) % 20 = 0 then
    250 * 2 ^ ((year - 2000) / 20)
  else
    0 -- This handles non-multiples of 20 cases, which are irrelevant here

theorem population_in_2060 : population 2060 = 2000 := by
  sorry

end population_in_2060_l145_145599


namespace locus_of_midpoint_l145_145739

theorem locus_of_midpoint {P Q M : ℝ × ℝ} (hP_on_circle : P.1^2 + P.2^2 = 13)
  (hQ_perpendicular_to_y_axis : Q.1 = P.1) (h_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (M.1^2 / (13 / 4)) + (M.2^2 / 13) = 1 := 
sorry

end locus_of_midpoint_l145_145739


namespace length_of_AB_l145_145535

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 16 = 1

-- Define the line perpendicular to the x-axis passing through the right focus of the ellipse
def line_perpendicular_y_axis_through_focus (y : ℝ) : Prop := true

-- Define the right focus of the ellipse
def right_focus : ℝ × ℝ := (3, 0)

-- Statement to prove the length of the line segment AB
theorem length_of_AB : 
  ∃ A B : ℝ × ℝ, 
  (ellipse A.1 A.2 ∧ ellipse B.1 B.2) ∧ 
  (A.1 = 3 ∧ B.1 = 3) ∧
  (|A.2 - B.2| = 2 * 16 / 5) :=
sorry

end length_of_AB_l145_145535


namespace age_difference_l145_145728

variable (A B : ℕ)

-- Given conditions
def B_is_95 : Prop := B = 95
def A_after_30_years : Prop := A + 30 = 2 * (B - 30)

-- Theorem to prove
theorem age_difference (h1 : B_is_95 B) (h2 : A_after_30_years A B) : A - B = 5 := 
by
  sorry

end age_difference_l145_145728


namespace line_through_intersection_and_parallel_l145_145730

theorem line_through_intersection_and_parallel
  (x y : ℝ)
  (l1 : 3 * x + 4 * y - 2 = 0)
  (l2 : 2 * x + y + 2 = 0)
  (l3 : ∃ k : ℝ, k * x + y + 2 = 0 ∧ k = -(4 / 3)) :
  ∃ a b c : ℝ, a * x + b * y + c = 0 ∧ a = 4 ∧ b = 3 ∧ c = 2 := 
by
  sorry

end line_through_intersection_and_parallel_l145_145730


namespace minimize_quadratic_l145_145738

theorem minimize_quadratic (x : ℝ) : (∃ x, x = 3 ∧ ∀ y, 3 * (y ^ 2) - 18 * y + 7 ≥ 3 * (x ^ 2) - 18 * x + 7) :=
by
  sorry

end minimize_quadratic_l145_145738


namespace prob1_prob2_l145_145214

variables (x y a b c : ℝ)

-- Proof for the first problem
theorem prob1 :
  3 * x^2 * (-3 * x * y)^2 - x^2 * (x^2 * y^2 - 2 * x) = 26 * x^4 * y^2 + 2 * x^3 := 
sorry

-- Proof for the second problem
theorem prob2 :
  -2 * (-a^2 * b * c)^2 * (1 / 2) * a * (b * c)^3 - (-a * b * c)^3 * (-a * b * c)^2 = 0 :=
sorry

end prob1_prob2_l145_145214


namespace circle_equation_translation_l145_145898

theorem circle_equation_translation (x y : ℝ) :
  x^2 + y^2 - 4 * x + 6 * y - 68 = 0 → (x - 2)^2 + (y + 3)^2 = 81 :=
by
  intro h
  sorry

end circle_equation_translation_l145_145898


namespace no_positive_integer_satisfies_conditions_l145_145553

theorem no_positive_integer_satisfies_conditions : 
  ¬ ∃ (n : ℕ), (100 ≤ n / 4) ∧ (n / 4 ≤ 999) ∧ (100 ≤ 4 * n) ∧ (4 * n ≤ 999) :=
by
  -- Proof will go here.
  sorry

end no_positive_integer_satisfies_conditions_l145_145553


namespace smallest_perfect_square_4_10_18_l145_145545

theorem smallest_perfect_square_4_10_18 :
  ∃ n : ℕ, (∃ k : ℕ, n = k^2) ∧ (4 ∣ n) ∧ (10 ∣ n) ∧ (18 ∣ n) ∧ n = 900 := 
  sorry

end smallest_perfect_square_4_10_18_l145_145545


namespace parts_rate_relation_l145_145071

theorem parts_rate_relation
  (x : ℝ)
  (total_parts_per_hour : ℝ)
  (master_parts : ℝ)
  (apprentice_parts : ℝ)
  (h_total : total_parts_per_hour = 40)
  (h_master : master_parts = 300)
  (h_apprentice : apprentice_parts = 100)
  (h : total_parts_per_hour = x + (40 - x)) :
  (master_parts / x) = (apprentice_parts / (40 - x)) := 
by
  sorry

end parts_rate_relation_l145_145071


namespace integer_roots_condition_l145_145271

theorem integer_roots_condition (n : ℕ) (hn : n > 0) :
  (∃ x : ℤ, x^2 - 4 * x + n = 0) ↔ (n = 3 ∨ n = 4) := 
by
  sorry

end integer_roots_condition_l145_145271


namespace highest_place_value_734_48_l145_145434

theorem highest_place_value_734_48 : 
  (∃ k, 10^4 = k ∧ k * 10^4 ≤ 734 * 48 ∧ 734 * 48 < (k + 1) * 10^4) := 
sorry

end highest_place_value_734_48_l145_145434


namespace parabola_intersects_x_axis_l145_145233

theorem parabola_intersects_x_axis :
  ∀ m : ℝ, (m^2 - m - 1 = 0) → (-2 * m^2 + 2 * m + 2023 = 2021) :=
by 
intros m hm
/-
  Given condition: m^2 - m - 1 = 0
  We need to show: -2 * m^2 + 2 * m + 2023 = 2021
-/
sorry

end parabola_intersects_x_axis_l145_145233


namespace evaluate_x_squared_plus_y_squared_l145_145171

theorem evaluate_x_squared_plus_y_squared
  (x y : ℝ)
  (h1 : x + y = 12)
  (h2 : 3 * x + y = 20) :
  x^2 + y^2 = 80 := by
  sorry

end evaluate_x_squared_plus_y_squared_l145_145171


namespace marissa_initial_ribbon_l145_145378

theorem marissa_initial_ribbon (ribbon_per_box : ℝ) (number_of_boxes : ℝ) (ribbon_left : ℝ) : 
  (ribbon_per_box = 0.7) → (number_of_boxes = 5) → (ribbon_left = 1) → 
  (ribbon_per_box * number_of_boxes + ribbon_left = 4.5) :=
  by
    intros
    sorry

end marissa_initial_ribbon_l145_145378


namespace roots_of_quadratic_l145_145341

theorem roots_of_quadratic (a b c : ℝ) (h : ¬ (a = 0 ∧ b = 0 ∧ c = 0)) :
  ¬ ∃ (x : ℝ), x^2 + (a + b + c) * x + a^2 + b^2 + c^2 = 0 :=
by
  sorry

end roots_of_quadratic_l145_145341


namespace smallest_positive_n_l145_145329

theorem smallest_positive_n
  (a x y : ℤ)
  (h1 : x ≡ a [ZMOD 9])
  (h2 : y ≡ -a [ZMOD 9]) :
  ∃ n : ℕ, n > 0 ∧ (x^2 + x * y + y^2 + n) % 9 = 0 ∧ n = 6 :=
by
  sorry

end smallest_positive_n_l145_145329


namespace valid_p_values_l145_145452

theorem valid_p_values (p : ℕ) (h : p = 3 ∨ p = 4 ∨ p = 5 ∨ p = 12) :
  0 < (4 * p + 34) / (3 * p - 8) ∧ (4 * p + 34) % (3 * p - 8) = 0 :=
by
  sorry

end valid_p_values_l145_145452


namespace abs_ineq_solution_range_l145_145241

theorem abs_ineq_solution_range (a : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 2| > a) → a < 3 :=
by
  sorry

end abs_ineq_solution_range_l145_145241


namespace find_coefficients_l145_145652

theorem find_coefficients
  (a b c : ℝ)
  (hA : ∀ x : ℝ, (x = -3 ∨ x = 4) ↔ (x^2 + a * x - 12 = 0))
  (hB : ∀ x : ℝ, (x = -3 ∨ x = 1) ↔ (x^2 + b * x + c = 0))
  (hAnotB : ¬ (∀ x, (x^2 + a * x - 12 = 0) ↔ (x^2 + b * x + c = 0)))
  (hA_inter_B : ∀ x, x = -3 ↔ (x^2 + a * x - 12 = 0) ∧ (x^2 + b * x + c = 0))
  (hA_union_B : ∀ x, (x = -3 ∨ x = 1 ∨ x = 4) ↔ (x^2 + a * x - 12 = 0) ∨ (x^2 + b * x + c = 0)):
  a = -1 ∧ b = 2 ∧ c = -3 :=
sorry

end find_coefficients_l145_145652


namespace total_students_surveyed_l145_145757

variable (T : ℕ)
variable (F : ℕ)

theorem total_students_surveyed :
  (F = 20 + 60) → (F = 40 * (T / 100)) → (T = 200) :=
by
  intros h1 h2
  sorry

end total_students_surveyed_l145_145757


namespace remainingAreaCalculation_l145_145941

noncomputable def totalArea : ℝ := 9500.0
noncomputable def lizzieGroupArea : ℝ := 2534.1
noncomputable def hilltownTeamArea : ℝ := 2675.95
noncomputable def greenValleyCrewArea : ℝ := 1847.57

theorem remainingAreaCalculation :
  (totalArea - (lizzieGroupArea + hilltownTeamArea + greenValleyCrewArea) = 2442.38) :=
by
  sorry

end remainingAreaCalculation_l145_145941


namespace find_beta_l145_145752

theorem find_beta 
  (α β : ℝ)
  (h1 : Real.cos α = 1 / 7)
  (h2 : Real.cos (α + β) = -11 / 14)
  (h3 : 0 < α ∧ α < Real.pi / 2)
  (h4 : Real.pi / 2 < α + β ∧ α + β < Real.pi) :
  β = Real.pi / 3 := 
sorry

end find_beta_l145_145752


namespace left_handed_rock_lovers_l145_145173

def total_people := 30
def left_handed := 12
def like_rock_music := 20
def right_handed_dislike_rock := 3

theorem left_handed_rock_lovers : ∃ x, x + (left_handed - x) + (like_rock_music - x) + right_handed_dislike_rock = total_people ∧ x = 5 :=
by
  sorry

end left_handed_rock_lovers_l145_145173


namespace true_propositions_count_l145_145035

theorem true_propositions_count {a b c : ℝ} (h : a ≤ b) : 
  (if (c^2 ≥ 0 ∧ a * c^2 ≤ b * c^2) then 1 else 0) +
  (if (c^2 ≥ 0 ∧ a * c^2 > b * c^2) then 1 else 0) +
  (if (c^2 ≥ 0 ∧ ¬(a * c^2 ≤ b * c^2) → ¬(a ≤ b)) then 1 else 0) +
  (if (c^2 ≥ 0 ∧ ¬(a ≤ b) → ¬(a * c^2 ≤ b * c^2)) then 1 else 0) = 2 :=
sorry

end true_propositions_count_l145_145035


namespace range_of_2alpha_minus_beta_over_3_l145_145166

theorem range_of_2alpha_minus_beta_over_3 (α β : ℝ) (hα : 0 < α) (hα' : α < π / 2) (hβ : 0 < β) (hβ' : β < π / 2) : 
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π := 
sorry

end range_of_2alpha_minus_beta_over_3_l145_145166


namespace unique_solution_l145_145619

theorem unique_solution (a b x: ℝ) : 
  (4 * x - 7 + a = (b - 1) * x + 2) ↔ (b ≠ 5) := 
by
  sorry -- proof is omitted as per instructions

end unique_solution_l145_145619


namespace male_athletes_sampled_l145_145363

-- Define the total number of athletes
def total_athletes : Nat := 98

-- Define the number of female athletes
def female_athletes : Nat := 42

-- Define the probability of being selected
def selection_probability : ℚ := 2 / 7

-- Calculate the number of male athletes
def male_athletes : Nat := total_athletes - female_athletes

-- State the theorem about the number of male athletes sampled
theorem male_athletes_sampled : male_athletes * selection_probability = 16 :=
by
  sorry

end male_athletes_sampled_l145_145363


namespace range_of_expression_l145_145996

variable {a b : ℝ}

theorem range_of_expression 
  (h₁ : -1 < a + b) (h₂ : a + b < 3)
  (h₃ : 2 < a - b) (h₄ : a - b < 4) :
  -9 / 2 < 2 * a + 3 * b ∧ 2 * a + 3 * b < 13 / 2 := 
sorry

end range_of_expression_l145_145996


namespace age_difference_between_two_children_l145_145475

theorem age_difference_between_two_children 
  (avg_age_10_years_ago : ℕ)
  (present_avg_age : ℕ)
  (youngest_child_present_age : ℕ)
  (initial_family_members : ℕ)
  (current_family_members : ℕ)
  (H1 : avg_age_10_years_ago = 24)
  (H2 : present_avg_age = 24)
  (H3 : youngest_child_present_age = 3)
  (H4 : initial_family_members = 4)
  (H5 : current_family_members = 6) :
  ∃ (D: ℕ), D = 2 :=
by
  sorry

end age_difference_between_two_children_l145_145475


namespace polygon_with_largest_area_l145_145845

noncomputable def area_of_polygon_A : ℝ := 6
noncomputable def area_of_polygon_B : ℝ := 4
noncomputable def area_of_polygon_C : ℝ := 4 + 2 * (1 / 2 * 1 * 1)
noncomputable def area_of_polygon_D : ℝ := 3 + 3 * (1 / 2 * 1 * 1)
noncomputable def area_of_polygon_E : ℝ := 7

theorem polygon_with_largest_area : 
  area_of_polygon_E > area_of_polygon_A ∧ 
  area_of_polygon_E > area_of_polygon_B ∧ 
  area_of_polygon_E > area_of_polygon_C ∧ 
  area_of_polygon_E > area_of_polygon_D :=
by
  sorry

end polygon_with_largest_area_l145_145845


namespace cards_exchanged_l145_145580

theorem cards_exchanged (x : ℕ) (h : x * (x - 1) = 1980) : x * (x - 1) = 1980 :=
by sorry

end cards_exchanged_l145_145580


namespace starting_number_is_33_l145_145407

theorem starting_number_is_33 (n : ℕ)
  (h1 : ∀ k, (33 + k * 11 ≤ 79) → (k < 5))
  (h2 : ∀ k, (k < 5) → (33 + k * 11 ≤ 79)) :
  n = 33 :=
sorry

end starting_number_is_33_l145_145407


namespace range_of_m_l145_145308

theorem range_of_m (m : ℝ) :
  (∀ x y : ℝ, (x^2 : ℝ) / (2 - m) + (y^2 : ℝ) / (m - 1) = 1 → 2 - m < 0 ∧ m - 1 > 0) →
  (∀ Δ : ℝ, Δ = 16 * (m - 2) ^ 2 - 16 → Δ < 0 → 1 < m ∧ m < 3) →
  (∀ (p q : Prop), p ∨ q ∧ ¬ q → p ∧ ¬ q) →
  m ≥ 3 :=
by
  intros h1 h2 h3
  sorry

end range_of_m_l145_145308


namespace correct_statement_l145_145006

variable (P Q : Prop)
variable (hP : P)
variable (hQ : Q)

theorem correct_statement :
  (P ∧ Q) :=
by
  exact ⟨hP, hQ⟩

end correct_statement_l145_145006


namespace a_six_between_three_and_four_l145_145052

theorem a_six_between_three_and_four (a : ℝ) (h : a^5 - a^3 + a = 2) : 3 < a^6 ∧ a^6 < 4 := 
sorry

end a_six_between_three_and_four_l145_145052


namespace max_value_of_function_l145_145170

theorem max_value_of_function : 
  ∀ (x : ℝ), 0 ≤ x → x ≤ 1 → (3 * x - 4 * x^3) ≤ 1 :=
by
  intro x hx0 hx1
  -- proof goes here
  sorry

end max_value_of_function_l145_145170


namespace average_age_increase_39_l145_145913

variable (n : ℕ) (A : ℝ)
noncomputable def average_age_increase (r : ℝ) : Prop :=
  (r = 7) →
  (n + 1) * (A + r) = n * A + 39 →
  (n + 1) * (A - 1) = n * A + 15 →
  r = 7

theorem average_age_increase_39 : ∀ (n : ℕ) (A : ℝ), average_age_increase n A 7 :=
by
  intros n A
  unfold average_age_increase
  intros hr h1 h2
  exact hr

end average_age_increase_39_l145_145913


namespace number_of_students_only_taking_AMC8_l145_145860

def total_Germain := 13
def total_Newton := 10
def total_Young := 12

def olympiad_Germain := 3
def olympiad_Newton := 2
def olympiad_Young := 4

def number_only_AMC8 :=
  (total_Germain - olympiad_Germain) +
  (total_Newton - olympiad_Newton) +
  (total_Young - olympiad_Young)

theorem number_of_students_only_taking_AMC8 :
  number_only_AMC8 = 26 := by
  sorry

end number_of_students_only_taking_AMC8_l145_145860


namespace fewerEmployeesAbroadThanInKorea_l145_145257

def totalEmployees : Nat := 928
def employeesInKorea : Nat := 713
def employeesAbroad : Nat := totalEmployees - employeesInKorea

theorem fewerEmployeesAbroadThanInKorea :
  employeesInKorea - employeesAbroad = 498 :=
by
  sorry

end fewerEmployeesAbroadThanInKorea_l145_145257


namespace total_crayons_l145_145262

theorem total_crayons (crayons_per_child : ℕ) (number_of_children : ℕ) (h1 : crayons_per_child = 3) (h2 : number_of_children = 6) : 
  crayons_per_child * number_of_children = 18 := by
  sorry

end total_crayons_l145_145262


namespace total_books_l145_145964

theorem total_books (D Loris Lamont : ℕ) 
  (h1 : Loris + 3 = Lamont)
  (h2 : Lamont = 2 * D)
  (h3 : D = 20) : D + Loris + Lamont = 97 := 
by 
  sorry

end total_books_l145_145964


namespace sum_of_remainders_l145_145676

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 11) : 
  (n % 3 + n % 6) = 7 :=
by
  sorry

end sum_of_remainders_l145_145676


namespace max_product_not_less_than_993_squared_l145_145180

theorem max_product_not_less_than_993_squared :
  ∀ (a : Fin 1985 → ℕ), 
    (∀ i, ∃ j, a j = i + 1) →  -- representation of permutation
    (∃ i : Fin 1985, i * (a i) ≥ 993 * 993) :=
by
  intros a h
  sorry

end max_product_not_less_than_993_squared_l145_145180


namespace find_common_difference_l145_145741

variable {a : ℕ → ℝ} (h_arith : ∀ n, a (n + 1) = a n + d)
variable (a7_minus_2a4_eq_6 : a 7 - 2 * a 4 = 6)
variable (a3_eq_2 : a 3 = 2)

theorem find_common_difference (d : ℝ) : d = 4 :=
by
  -- Proof would go here
  sorry

end find_common_difference_l145_145741


namespace cats_remaining_l145_145927

theorem cats_remaining 
  (n_initial n_given_away : ℝ) 
  (h_initial : n_initial = 17.0) 
  (h_given_away : n_given_away = 14.0) : 
  (n_initial - n_given_away) = 3.0 :=
by
  rw [h_initial, h_given_away]
  norm_num

end cats_remaining_l145_145927


namespace shaded_region_area_eq_l145_145703

noncomputable def areaShadedRegion : ℝ :=
  let side_square := 14
  let side_triangle := 18
  let height := 14
  let H := 9 * Real.sqrt 3
  let BF := (side_square + side_triangle, height - H)
  let base_BF := BF.1 - 0
  let height_BF := BF.2
  let area_triangle_BFH := 0.5 * base_BF * height_BF
  let total_triangle_area := 0.5 * side_triangle * height
  let area_half_BFE := 0.5 * total_triangle_area
  area_half_BFE - area_triangle_BFH

theorem shaded_region_area_eq :
  areaShadedRegion = 9 * Real.sqrt 3 :=
by 
 sorry

end shaded_region_area_eq_l145_145703


namespace joels_age_when_dad_twice_l145_145165

theorem joels_age_when_dad_twice
  (joel_age_now : ℕ)
  (dad_age_now : ℕ)
  (years : ℕ)
  (H1 : joel_age_now = 5)
  (H2 : dad_age_now = 32)
  (H3 : years = 22)
  (H4 : dad_age_now + years = 2 * (joel_age_now + years))
  : joel_age_now + years = 27 := 
by sorry

end joels_age_when_dad_twice_l145_145165


namespace area_sum_four_smaller_circles_equals_area_of_large_circle_l145_145814

theorem area_sum_four_smaller_circles_equals_area_of_large_circle (R : ℝ) :
  let radius_large := R
  let radius_small := R / 2
  let area_large := π * radius_large^2
  let area_small := π * radius_small^2
  let total_area_small := 4 * area_small
  area_large = total_area_small :=
by
  sorry

end area_sum_four_smaller_circles_equals_area_of_large_circle_l145_145814


namespace values_of_z_l145_145974

theorem values_of_z (x z : ℝ) 
  (h1 : 3 * x^2 + 9 * x + 7 * z + 2 = 0)
  (h2 : 3 * x + z + 4 = 0) : 
  z^2 + 20 * z - 14 = 0 := 
sorry

end values_of_z_l145_145974


namespace function_satisfies_equation_l145_145226

theorem function_satisfies_equation (y : ℝ → ℝ) (h : ∀ x : ℝ, y x = Real.exp (x + x^2) + 2 * Real.exp x) :
  ∀ x : ℝ, deriv y x - y x = 2 * x * Real.exp (x + x^2) :=
by {
  sorry
}

end function_satisfies_equation_l145_145226


namespace num_of_laborers_is_24_l145_145957

def average_salary_all (L S : Nat) (avg_salary_ls : Nat) (avg_salary_l : Nat) (avg_salary_s : Nat) : Prop :=
  (avg_salary_l * L + avg_salary_s * S) / (L + S) = avg_salary_ls

def average_salary_supervisors (S : Nat) (avg_salary_s : Nat) : Prop :=
  (avg_salary_s * S) / S = avg_salary_s

theorem num_of_laborers_is_24 :
  ∀ (L S : Nat) (avg_salary_ls avg_salary_l avg_salary_s : Nat),
    average_salary_all L S avg_salary_ls avg_salary_l avg_salary_s →
    average_salary_supervisors S avg_salary_s →
    S = 6 → avg_salary_ls = 1250 → avg_salary_l = 950 → avg_salary_s = 2450 →
    L = 24 :=
by
  intros L S avg_salary_ls avg_salary_l avg_salary_s h1 h2 h3 h4 h5 h6
  sorry

end num_of_laborers_is_24_l145_145957


namespace rational_linear_independent_sqrt_prime_l145_145030

theorem rational_linear_independent_sqrt_prime (p : ℕ) (hp : Nat.Prime p) (m n m1 n1 : ℚ) :
  m + n * Real.sqrt p = m1 + n1 * Real.sqrt p → m = m1 ∧ n = n1 :=
sorry

end rational_linear_independent_sqrt_prime_l145_145030


namespace part1_part2_l145_145943

noncomputable def x : ℝ := 1 - Real.sqrt 2
noncomputable def y : ℝ := 1 + Real.sqrt 2

theorem part1 : x^2 + 3 * x * y + y^2 = 3 := by
  sorry

theorem part2 : (y / x) - (x / y) = -4 * Real.sqrt 2 := by
  sorry

end part1_part2_l145_145943


namespace area_of_rectangle_l145_145474

theorem area_of_rectangle (w l : ℕ) (hw : w = 10) (hl : l = 2) : (w * l) = 20 :=
by
  sorry

end area_of_rectangle_l145_145474


namespace number_of_10_yuan_coins_is_1_l145_145420

theorem number_of_10_yuan_coins_is_1
  (n : ℕ) -- number of coins
  (v : ℕ) -- total value of coins
  (c1 c5 c10 c50 : ℕ) -- number of 1, 5, 10, and 50 yuan coins
  (h1 : n = 9) -- there are nine coins in total
  (h2 : v = 177) -- the total value of these coins is 177 yuan
  (h3 : c1 ≥ 1 ∧ c5 ≥ 1 ∧ c10 ≥ 1 ∧ c50 ≥ 1) -- at least one coin of each denomination
  (h4 : c1 + c5 + c10 + c50 = n) -- sum of all coins number is n
  (h5 : c1 * 1 + c5 * 5 + c10 * 10 + c50 * 50 = v) -- total value of all coins is v
  : c10 = 1 := 
sorry

end number_of_10_yuan_coins_is_1_l145_145420


namespace cylinder_cut_is_cylinder_l145_145410

-- Define what it means to be a cylinder
structure Cylinder (r h : ℝ) : Prop :=
(r_pos : r > 0)
(h_pos : h > 0)

-- Define the condition of cutting a cylinder with two parallel planes
def cut_by_parallel_planes (c : Cylinder r h) (d : ℝ) : Prop :=
d > 0 ∧ d < h

-- Prove that the part between the parallel planes is still a cylinder
theorem cylinder_cut_is_cylinder (r h d : ℝ) (c : Cylinder r h) (H : cut_by_parallel_planes c d) :
  ∃ r' h', Cylinder r' h' :=
sorry

end cylinder_cut_is_cylinder_l145_145410


namespace h_odd_l145_145128

variable (f g : ℝ → ℝ)

-- f is odd and g is even
axiom f_odd : ∀ x, -2 ≤ x ∧ x ≤ 2 → f (-x) = -f x
axiom g_even : ∀ x, -2 ≤ x ∧ x ≤ 2 → g (-x) = g x

-- Prove that h(x) = f(x) * g(x) is odd
theorem h_odd : ∀ x, -2 ≤ x ∧ x ≤ 2 → (f x) * (g x) = (f (-x)) * (g (-x)) := by
  sorry

end h_odd_l145_145128


namespace container_capacity_l145_145109

/-- Given a container where 8 liters is 20% of its capacity, calculate the total capacity of 
    40 such containers filled with water. -/
theorem container_capacity (c : ℝ) (h : 8 = 0.20 * c) : 
    40 * c * 40 = 1600 := 
by
  sorry

end container_capacity_l145_145109


namespace greatest_integer_of_set_is_152_l145_145829

-- Define the conditions
def median (s : Set ℤ) : ℤ := 150
def smallest_integer (s : Set ℤ) : ℤ := 140
def consecutive_even_integers (s : Set ℤ) : Prop := 
  ∀ x ∈ s, ∃ y ∈ s, x = y ∨ x = y + 2

-- The main theorem
theorem greatest_integer_of_set_is_152 (s : Set ℤ) 
  (h_median : median s = 150)
  (h_smallest : smallest_integer s = 140)
  (h_consecutive : consecutive_even_integers s) : 
  ∃ greatest : ℤ, greatest = 152 := 
sorry

end greatest_integer_of_set_is_152_l145_145829


namespace batsman_average_after_17th_inning_l145_145398

-- Definitions for the conditions
def runs_scored_in_17th_inning : ℝ := 95
def increase_in_average : ℝ := 2.5

-- Lean statement encapsulating the problem
theorem batsman_average_after_17th_inning (A : ℝ) (h : 16 * A + runs_scored_in_17th_inning = 17 * (A + increase_in_average)) :
  A + increase_in_average = 55 := 
sorry

end batsman_average_after_17th_inning_l145_145398


namespace find_time_l145_145862

variables (V V_0 S g C : ℝ) (t : ℝ)

-- Given conditions.
axiom eq1 : V = 2 * g * t + V_0
axiom eq2 : S = (1 / 3) * g * t^2 + V_0 * t + C * t^3

-- The statement to prove.
theorem find_time : t = (V - V_0) / (2 * g) :=
sorry

end find_time_l145_145862


namespace quadratic_has_two_distinct_real_roots_l145_145875

theorem quadratic_has_two_distinct_real_roots (m : ℝ) : 
  (∃ (x y : ℝ), x ≠ y ∧ x^2 - 2 * x + m = 0 ∧ y^2 - 2 * y + m = 0) ↔ m < 1 :=
sorry

end quadratic_has_two_distinct_real_roots_l145_145875


namespace decrease_in_profit_when_one_loom_idles_l145_145965

def num_looms : ℕ := 125
def total_sales_value : ℕ := 500000
def total_manufacturing_expenses : ℕ := 150000
def monthly_establishment_charges : ℕ := 75000
def sales_value_per_loom : ℕ := total_sales_value / num_looms
def manufacturing_expense_per_loom : ℕ := total_manufacturing_expenses / num_looms
def decrease_in_sales_value : ℕ := sales_value_per_loom
def decrease_in_manufacturing_expenses : ℕ := manufacturing_expense_per_loom
def net_decrease_in_profit : ℕ := decrease_in_sales_value - decrease_in_manufacturing_expenses

theorem decrease_in_profit_when_one_loom_idles : net_decrease_in_profit = 2800 := by
  sorry

end decrease_in_profit_when_one_loom_idles_l145_145965


namespace num_factors_of_60_l145_145320

-- Definition of 60 in terms of its prime factors
def n : ℕ := 60
def a : ℕ := 2
def b : ℕ := 1
def c : ℕ := 1

-- Statement for the number of positive factors
theorem num_factors_of_60 :
  (a + 1) * (b + 1) * (c + 1) = 12 :=
by 
  -- We are skipping the proof part by using sorry.
  sorry

end num_factors_of_60_l145_145320


namespace distance_midpoint_chord_AB_to_y_axis_l145_145947

theorem distance_midpoint_chord_AB_to_y_axis
  (k : ℝ)
  (A B : ℝ × ℝ)
  (hA : A.2 = k * A.1 - k)
  (hB : B.2 = k * B.1 - k)
  (hA_on_parabola : A.2 ^ 2 = 4 * A.1)
  (hB_on_parabola : B.2 ^ 2 = 4 * B.1)
  (h_distance_AB : dist A B = 4) :
  (abs ((A.1 + B.1) / 2)) = 1 :=
by
  sorry

end distance_midpoint_chord_AB_to_y_axis_l145_145947


namespace quadratic_smaller_solution_l145_145210

theorem quadratic_smaller_solution : ∀ (x : ℝ), x^2 - 9 * x + 20 = 0 → x = 4 ∨ x = 5 :=
by
  sorry

end quadratic_smaller_solution_l145_145210


namespace moles_of_NaHCO3_needed_l145_145900

theorem moles_of_NaHCO3_needed 
  (HC2H3O2_moles: ℕ)
  (H2O_moles: ℕ)
  (NaHCO3_HC2H3O2_molar_ratio: ℕ)
  (reaction: NaHCO3_HC2H3O2_molar_ratio = 1 ∧ H2O_moles = 3) :
  ∃ NaHCO3_moles : ℕ, NaHCO3_moles = 3 :=
by
  sorry

end moles_of_NaHCO3_needed_l145_145900


namespace find_x_l145_145500

theorem find_x (x : ℤ) (h : 2 * x = (26 - x) + 19) : x = 15 :=
by
  sorry

end find_x_l145_145500


namespace q_can_do_work_in_10_days_l145_145538

theorem q_can_do_work_in_10_days (R_p R_q R_pq: ℝ)
  (h1 : R_p = 1 / 15)
  (h2 : R_pq = 1 / 6)
  (h3 : R_p + R_q = R_pq) :
  1 / R_q = 10 :=
by
  -- Proof steps go here.
  sorry

end q_can_do_work_in_10_days_l145_145538


namespace mean_equals_d_l145_145107

noncomputable def sqrt (x : ℝ) : ℝ :=
  Real.sqrt x

theorem mean_equals_d
  (a b c d e : ℝ)
  (h_a : a = sqrt 2)
  (h_b : b = sqrt 18)
  (h_c : c = sqrt 200)
  (h_d : d = sqrt 32)
  (h_e : e = sqrt 8) :
  d = (a + b + c + e) / 4 := by
  -- We insert proof steps here normally
  sorry

end mean_equals_d_l145_145107


namespace animal_shelter_kittens_count_l145_145337

def num_puppies : ℕ := 32
def num_kittens_more : ℕ := 14

theorem animal_shelter_kittens_count : 
  ∃ k : ℕ, k = (2 * num_puppies) + num_kittens_more := 
sorry

end animal_shelter_kittens_count_l145_145337


namespace ratio_of_areas_of_squares_l145_145174

theorem ratio_of_areas_of_squares (sideC sideD : ℕ) (hC : sideC = 45) (hD : sideD = 60) : 
  (sideC ^ 2) / (sideD ^ 2) = 9 / 16 := 
by
  sorry

end ratio_of_areas_of_squares_l145_145174


namespace find_cost_of_chocolate_l145_145182

theorem find_cost_of_chocolate
  (C : ℕ)
  (h1 : 5 * C + 10 = 90 - 55)
  (h2 : 5 * 2 = 10)
  (h3 : 55 = 90 - (5 * C + 10)):
  C = 5 :=
by
  sorry

end find_cost_of_chocolate_l145_145182


namespace problem_statement_l145_145496

theorem problem_statement (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x^3 + (1 / (y + 2016)) = y^3 + (1 / (z + 2016))) 
  (h5 : y^3 + (1 / (z + 2016)) = z^3 + (1 / (x + 2016))) : 
  (x^3 + y^3 + z^3) / (x * y * z) = 3 :=
by
  sorry

end problem_statement_l145_145496


namespace compare_flavors_l145_145422

def flavor_ratings_A := [7, 9, 8, 6, 10]
def flavor_ratings_B := [5, 6, 10, 10, 9]

def mean (l : List ℕ) : ℚ := l.sum / l.length

def variance (l : List ℕ) : ℚ :=
  let μ := mean l
  (l.map (λ x => (x - μ) ^ 2)).sum / l.length

theorem compare_flavors : 
  mean flavor_ratings_A = mean flavor_ratings_B ∧ variance flavor_ratings_A < variance flavor_ratings_B := by
  sorry

end compare_flavors_l145_145422


namespace find_number_l145_145354

def number_equal_when_divided_by_3_and_subtracted : Prop :=
  ∃ x : ℝ, (x / 3 = x - 3) ∧ (x = 4.5)

theorem find_number (x : ℝ) : (x / 3 = x - 3) → x = 4.5 :=
by
  sorry

end find_number_l145_145354


namespace obtuse_angle_probability_l145_145221

-- Defining the vertices of the pentagon
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨0, 3⟩
def B : Point := ⟨5, 0⟩
def C : Point := ⟨8, 0⟩
def D : Point := ⟨8, 5⟩
def E : Point := ⟨0, 5⟩

def is_interior (P : Point) : Prop :=
  -- A condition to define if a point is inside the pentagon
  sorry

def is_obtuse_angle (A B P : Point) : Prop :=
  -- Condition for angle APB to be obtuse
  sorry

noncomputable def probability_obtuse_angle :=
  -- Probability calculation
  let area_pentagon := 40
  let area_circle := (34 * Real.pi) / 4
  let area_outside_circle := area_pentagon - area_circle
  area_outside_circle / area_pentagon

theorem obtuse_angle_probability :
  ∀ P : Point, is_interior P → ∃! p : ℝ, p = (160 - 34 * Real.pi) / 160 :=
sorry

end obtuse_angle_probability_l145_145221


namespace largest_possible_perimeter_l145_145405

theorem largest_possible_perimeter (x : ℕ) (h1 : 1 < x) (h2 : x < 15) : 
  (7 + 8 + x) ≤ 29 := 
sorry

end largest_possible_perimeter_l145_145405


namespace cyclist_speed_l145_145540

theorem cyclist_speed (c d : ℕ) (h1 : d = c + 5) (hc : c ≠ 0) (hd : d ≠ 0)
    (H1 : ∀ tC tD : ℕ, 80 = c * tC → 120 = d * tD → tC = tD) : c = 10 := by
  sorry

end cyclist_speed_l145_145540


namespace trigonometric_expression_value_l145_145197

theorem trigonometric_expression_value :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 4 :=
by
  sorry

end trigonometric_expression_value_l145_145197


namespace Danny_caps_vs_wrappers_l145_145110

def park_caps : ℕ := 58
def park_wrappers : ℕ := 25
def beach_caps : ℕ := 34
def beach_wrappers : ℕ := 15
def forest_caps : ℕ := 21
def forest_wrappers : ℕ := 32
def before_caps : ℕ := 12
def before_wrappers : ℕ := 11

noncomputable def total_caps : ℕ := park_caps + beach_caps + forest_caps + before_caps
noncomputable def total_wrappers : ℕ := park_wrappers + beach_wrappers + forest_wrappers + before_wrappers

theorem Danny_caps_vs_wrappers : total_caps - total_wrappers = 42 := by
  sorry

end Danny_caps_vs_wrappers_l145_145110


namespace arithmetic_sequence_inequality_l145_145843

theorem arithmetic_sequence_inequality 
  (a b c : ℝ) 
  (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : b - a = d)
  (h3 : c - b = d) :
  ¬ (a^3 * b + b^3 * c + c^3 * a ≥ a^4 + b^4 + c^4) :=
sorry

end arithmetic_sequence_inequality_l145_145843


namespace group_members_l145_145850

theorem group_members (n : ℕ) (hn : n * n = 1369) : n = 37 :=
by
  sorry

end group_members_l145_145850


namespace hyperbola_equation_l145_145216

theorem hyperbola_equation:
  let F1 := (-Real.sqrt 10, 0)
  let F2 := (Real.sqrt 10, 0)
  ∃ P : ℝ × ℝ, 
    (let PF1 := (P.1 - F1.1, P.2 - F1.2);
     let PF2 := (P.1 - F2.1, P.2 - F2.2);
     (PF1.1 * PF2.1 + PF1.2 * PF2.2 = 0) ∧ 
     ((Real.sqrt (PF1.1^2 + PF1.2^2)) * (Real.sqrt (PF2.1^2 + PF2.2^2)) = 2)) →
    (∃ a b : ℝ, (a^2 = 9 ∧ b^2 = 1) ∧ 
                (∀ x y : ℝ, 
                 (a ≠ 0 ∧ (x^2 / a^2) - (y^2 / b^2) = 1 ↔ 
                  ∃ P : ℝ × ℝ, 
                    let PF1 := (P.1 - F1.1, P.2 - F1.2);
                    let PF2 := (P.1 - F2.1, P.2 - F2.2);
                    PF1.1 * PF2.1 + PF1.2 * PF2.2 = 0 ∧ 
                    (Real.sqrt (PF1.1^2 + PF1.2^2)) * (Real.sqrt (PF2.1^2 + PF2.2^2)) = 2)))
:= by
sorry

end hyperbola_equation_l145_145216


namespace centroid_path_area_correct_l145_145847

noncomputable def centroid_path_area (AB : ℝ) (A B C : ℝ × ℝ) (O : ℝ × ℝ) : ℝ :=
  let R := AB / 2
  let radius_of_path := R / 3
  let area := Real.pi * radius_of_path ^ 2
  area

theorem centroid_path_area_correct (AB : ℝ) (A B C : ℝ × ℝ)
  (hAB : AB = 32)
  (hAB_diameter : (∃ O : ℝ × ℝ, dist O A = dist O B ∧ dist A B = 2 * dist O A))
  (hC_circle : ∃ O : ℝ × ℝ, dist O C = AB / 2 ∧ C ≠ A ∧ C ≠ B):
  centroid_path_area AB A B C (0, 0) = (256 / 9) * Real.pi := by
  sorry

end centroid_path_area_correct_l145_145847


namespace evaluate_expression_l145_145086

theorem evaluate_expression : 
  70 + (5 * 12) / (180 / 3) = 71 :=
  by
  sorry

end evaluate_expression_l145_145086


namespace sufficient_and_necessary_condition_l145_145682

theorem sufficient_and_necessary_condition (a : ℝ) : 
  (0 < a ∧ a < 4) ↔ ∀ x : ℝ, (x^2 - a * x + a) > 0 :=
by sorry

end sufficient_and_necessary_condition_l145_145682


namespace symmetric_scanning_codes_count_l145_145016

theorem symmetric_scanning_codes_count :
  let grid_size := 5
  let total_squares := grid_size * grid_size
  let symmetry_classes := 5 -- Derived from classification in the solution
  let possible_combinations := 2 ^ symmetry_classes
  let invalid_combinations := 2 -- All black or all white grid
  total_squares = 25 
  ∧ (possible_combinations - invalid_combinations) = 30 :=
by sorry

end symmetric_scanning_codes_count_l145_145016


namespace first_consecutive_odd_number_l145_145867

theorem first_consecutive_odd_number :
  ∃ k : Int, 2 * k - 1 + 2 * k + 1 + 2 * k + 3 = 2 * k - 1 + 128 ∧ 2 * k - 1 = 61 :=
by
  sorry

end first_consecutive_odd_number_l145_145867


namespace problem_solution_l145_145447

noncomputable def proof_problem (x1 x2 x3 x4 x5 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) : Prop :=
  ((x1^2 - x3 * x5) * (x2^2 - x3 * x5) ≤ 0) ∧
  ((x2^2 - x4 * x1) * (x3^2 - x4 * x1) ≤ 0) ∧
  ((x3^2 - x5 * x2) * (x4^2 - x5 * x2) ≤ 0) ∧
  ((x4^2 - x1 * x3) * (x5^2 - x1 * x3) ≤ 0) ∧
  ((x5^2 - x2 * x4) * (x1^2 - x2 * x4) ≤ 0) → 
  x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5

theorem problem_solution (x1 x2 x3 x4 x5 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) :
  proof_problem x1 x2 x3 x4 x5 h1 h2 h3 h4 h5 :=
  by
    sorry

end problem_solution_l145_145447


namespace trading_organization_increase_price_l145_145897

theorem trading_organization_increase_price 
  (initial_moisture_content : ℝ)
  (final_moisture_content : ℝ)
  (solid_mass : ℝ)
  (initial_total_mass final_total_mass : ℝ) :
  initial_moisture_content = 0.99 → 
  final_moisture_content = 0.98 →
  initial_total_mass = 100 →
  solid_mass = initial_total_mass * (1 - initial_moisture_content) →
  final_total_mass = solid_mass / (1 - final_moisture_content) →
  (final_total_mass / initial_total_mass) = 0.5 →
  100 * (1 - (final_total_mass / initial_total_mass)) = 100 :=
by sorry

end trading_organization_increase_price_l145_145897


namespace table_mat_length_l145_145750

noncomputable def calculate_y (r : ℝ) (n : ℕ) (w : ℝ) : ℝ :=
  let θ := 2 * Real.pi / n
  let y_side := 2 * r * Real.sin (θ / 2)
  y_side

theorem table_mat_length :
  calculate_y 6 8 1 = 3 * Real.sqrt (2 - Real.sqrt 2) :=
by
  sorry

end table_mat_length_l145_145750


namespace two_times_six_pow_n_plus_one_ne_product_of_consecutive_l145_145598

theorem two_times_six_pow_n_plus_one_ne_product_of_consecutive (n k : ℕ) :
  2 * (6 ^ n + 1) ≠ k * (k + 1) :=
sorry

end two_times_six_pow_n_plus_one_ne_product_of_consecutive_l145_145598


namespace george_team_final_round_average_required_less_than_record_l145_145670

theorem george_team_final_round_average_required_less_than_record :
  ∀ (old_record average_score : ℕ) (players : ℕ) (rounds : ℕ) (current_score : ℕ),
    old_record = 287 →
    players = 4 →
    rounds = 10 →
    current_score = 10440 →
    (old_record - ((rounds * (old_record * players) - current_score) / players)) = 27 :=
by
  -- Given the values and conditions, prove the equality here
  sorry

end george_team_final_round_average_required_less_than_record_l145_145670


namespace find_pair_l145_145779

noncomputable def x_n (n : ℕ) : ℝ := n / (n + 2016)

theorem find_pair :
  ∃ (m n : ℕ), x_n 2016 = (x_n m) * (x_n n) ∧ (m = 6048 ∧ n = 4032) :=
by {
  sorry
}

end find_pair_l145_145779


namespace balance_test_l145_145849

variable (a b h c : ℕ)

theorem balance_test
  (h1 : 4 * a + 2 * b + h = 21 * c)
  (h2 : 2 * a = b + h + 5 * c) :
  b + 2 * h = 11 * c :=
sorry

end balance_test_l145_145849


namespace infinite_series_sum_l145_145061

theorem infinite_series_sum :
  (∑' n : ℕ, if n = 0 then 0 else (3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1)))) = 1 / 4 :=
by
  sorry

end infinite_series_sum_l145_145061


namespace volume_solid_correct_l145_145094

noncomputable def volume_of_solid : ℝ := 
  let area_rhombus := 1250 -- Area of the rhombus calculated from the bounded region
  let height := 10 -- Given height of the solid
  area_rhombus * height -- Volume of the solid

theorem volume_solid_correct (height: ℝ := 10) :
  volume_of_solid = 12500 := by
  sorry

end volume_solid_correct_l145_145094


namespace sqrt_16_eq_pm_4_l145_145864

theorem sqrt_16_eq_pm_4 (x : ℝ) (h : x^2 = 16) : x = 4 ∨ x = -4 := by
  sorry

end sqrt_16_eq_pm_4_l145_145864


namespace goals_per_player_is_30_l145_145321

-- Define the total number of goals scored in the league against Barca
def total_goals : ℕ := 300

-- Define the percentage of goals scored by the two players
def percentage_of_goals : ℝ := 0.20

-- Define the combined goals by the two players
def combined_goals := (percentage_of_goals * total_goals : ℝ)

-- Define the number of players
def number_of_players : ℕ := 2

-- Define the number of goals scored by each player
noncomputable def goals_per_player := combined_goals / number_of_players

-- Proof statement: Each of the two players scored 30 goals.
theorem goals_per_player_is_30 :
  goals_per_player = 30 :=
sorry

end goals_per_player_is_30_l145_145321


namespace find_m_n_l145_145125

theorem find_m_n (m n : ℕ) (h : 26019 * m - 649 * n = 118) : m = 2 ∧ n = 80 :=
by 
  sorry

end find_m_n_l145_145125


namespace region_area_l145_145954

-- Let x and y be real numbers
variables (x y : ℝ)

-- Define the inequality condition
def region_condition (x y : ℝ) : Prop := abs (4 * x - 20) + abs (3 * y + 9) ≤ 6

-- The statement that needs to be proved
theorem region_area : (∃ x y : ℝ, region_condition x y) → ∃ A : ℝ, A = 6 :=
by
  sorry

end region_area_l145_145954


namespace max_area_of_garden_l145_145186

theorem max_area_of_garden (l w : ℝ) 
  (h : 2 * l + w = 400) : 
  l * w ≤ 20000 :=
sorry

end max_area_of_garden_l145_145186


namespace find_line_eq_show_point_on_circle_l145_145839

noncomputable section

variables {x y x0 y0 : ℝ} (P Q : ℝ × ℝ) (h1 : y0 ≠ 0)
  (h2 : P = (x0, y0))
  (h3 : P.1^2/4 + P.2^2/3 = 1)
  (h4 : Q = (x0/4, y0/3))

theorem find_line_eq (M : ℝ × ℝ) (hM : ∀ (M : ℝ × ℝ), 
  ((P.1 - M.1) , (P.2 - M.2)) • (Q.1 , Q.2) = 0) :
  ∀ (x0 y0 : ℝ), y0 ≠ 0 → ∀ (x y : ℝ), 
  (x0 * x / 4 + y0 * y / 3 = 1) :=
by sorry
  
theorem show_point_on_circle (F S : ℝ × ℝ)
  (hF : F = (1, 0)) (hs : ∀ (x0 y0 : ℝ), y0 ≠ 0 → 
  S = (4, 0) ∧ ((S.1 - P.1) ^ 2 + (S.2 - P.2) ^ 2 = 36)) :
  ∀ (x y : ℝ), 
  (x - 1) ^ 2 + y ^ 2 = 36 := 
by sorry

end find_line_eq_show_point_on_circle_l145_145839


namespace center_radius_sum_l145_145265

theorem center_radius_sum (a b r : ℝ) (h : ∀ x y : ℝ, (x^2 - 8*x - 4*y = -y^2 + 2*y + 13) ↔ (x - 4)^2 + (y - 3)^2 = 38) :
  a = 4 ∧ b = 3 ∧ r = Real.sqrt 38 → a + b + r = 7 + Real.sqrt 38 :=
by
  sorry

end center_radius_sum_l145_145265


namespace chandu_work_days_l145_145191

theorem chandu_work_days (W : ℝ) (c : ℝ) 
  (anand_rate : ℝ := W / 7) 
  (bittu_rate : ℝ := W / 8) 
  (chandu_rate : ℝ := W / c) 
  (completed_in_7_days : 3 * anand_rate + 2 * bittu_rate + 2 * chandu_rate = W) : 
  c = 7 :=
by
  sorry

end chandu_work_days_l145_145191


namespace no_matching_formula_l145_145426

def xy_pairs : List (ℕ × ℕ) := [(1, 5), (2, 15), (3, 35), (4, 69), (5, 119)]

def formula_a (x : ℕ) : ℕ := x^3 + x^2 + x + 2
def formula_b (x : ℕ) : ℕ := 3 * x^2 + 2 * x + 1
def formula_c (x : ℕ) : ℕ := 2 * x^3 - x + 4
def formula_d (x : ℕ) : ℕ := 3 * x^3 + 2 * x^2 + x + 1

theorem no_matching_formula :
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_a pair.fst) ∧
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_b pair.fst) ∧
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_c pair.fst) ∧
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_d pair.fst) :=
by
  sorry

end no_matching_formula_l145_145426


namespace average_visitors_on_sundays_is_correct_l145_145833

noncomputable def average_visitors_sundays
  (num_sundays : ℕ) (num_non_sundays : ℕ) 
  (avg_non_sunday_visitors : ℕ) (avg_month_visitors : ℕ) : ℕ :=
  let total_month_days := num_sundays + num_non_sundays
  let total_visitors := avg_month_visitors * total_month_days
  let total_non_sunday_visitors := num_non_sundays * avg_non_sunday_visitors
  let total_sunday_visitors := total_visitors - total_non_sunday_visitors
  total_sunday_visitors / num_sundays

theorem average_visitors_on_sundays_is_correct :
  average_visitors_sundays 5 25 240 290 = 540 :=
by
  sorry

end average_visitors_on_sundays_is_correct_l145_145833


namespace sum_of_thousands_and_units_digit_of_product_l145_145570

def thousands_digit (n : ℕ) : ℕ := (n / 1000) % 10
def units_digit (n : ℕ) : ℕ := n % 10

-- Define the two 102-digit numbers
def num1 : ℕ := 404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404
def num2 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707

-- Define their product
def product : ℕ := num1 * num2

-- Define the conditions for the problem
def A := thousands_digit product
def B := units_digit product

-- Define the problem statement
theorem sum_of_thousands_and_units_digit_of_product : A + B = 13 := 
by
  sorry

end sum_of_thousands_and_units_digit_of_product_l145_145570


namespace invitation_methods_l145_145000

-- Definitions
def num_ways_invite_6_out_of_10 : ℕ := Nat.choose 10 6
def num_ways_both_A_and_B : ℕ := Nat.choose 8 4

-- Theorem statement
theorem invitation_methods : num_ways_invite_6_out_of_10 - num_ways_both_A_and_B = 140 :=
by
  -- Proof should be provided here
  sorry

end invitation_methods_l145_145000


namespace frustum_slant_height_l145_145396

-- The setup: we are given specific conditions for a frustum resulting from cutting a cone
variable {r : ℝ} -- represents the radius of the upper base of the frustum
variable {h : ℝ} -- represents the slant height of the frustum
variable {h_removed : ℝ} -- represents the slant height of the removed cone

-- The given conditions
def upper_base_radius : ℝ := r
def lower_base_radius : ℝ := 4 * r
def slant_height_removed_cone : ℝ := 3

-- The proportion derived from similar triangles
def proportion (h r : ℝ) := (h / (4 * r)) = ((h + 3) / (5 * r))

-- The main statement: proving the slant height of the frustum is 9 cm
theorem frustum_slant_height (r : ℝ) (h : ℝ) (hr : proportion h r) : h = 9 :=
sorry

end frustum_slant_height_l145_145396


namespace boat_travel_l145_145349

theorem boat_travel (T_against T_with : ℝ) (V_b D V_c : ℝ) 
  (hT_against : T_against = 10) 
  (hT_with : T_with = 6) 
  (hV_b : V_b = 12)
  (hD1 : D = (V_b - V_c) * T_against)
  (hD2 : D = (V_b + V_c) * T_with) :
  V_c = 3 ∧ D = 90 :=
by
  sorry

end boat_travel_l145_145349


namespace upper_bound_for_k_squared_l145_145430

theorem upper_bound_for_k_squared :
  (∃ (k : ℤ), k^2 > 121 ∧ ∀ m : ℤ, (m^2 > 121 ∧ m^2 < 323 → m = k + 1)) →
  (k ≤ 17) → (18^2 > 323) := 
by 
  sorry

end upper_bound_for_k_squared_l145_145430


namespace f_neg_2_f_monotonically_decreasing_l145_145815

noncomputable def f : ℝ → ℝ := sorry

axiom f_add (x₁ x₂ : ℝ) : f (x₁ + x₂) = f x₁ + f x₂ - 4
axiom f_2 : f 2 = 0
axiom f_pos_2 (x : ℝ) : x > 2 → f x < 0

-- Statement to prove f(-2) = 8
theorem f_neg_2 : f (-2) = 8 := sorry

-- Statement to prove that f(x) is monotonically decreasing on ℝ
theorem f_monotonically_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ := sorry

end f_neg_2_f_monotonically_decreasing_l145_145815


namespace amount_of_benzene_l145_145715

-- Definitions of the chemical entities involved
def Benzene := Type
def Methane := Type
def Toluene := Type
def Hydrogen := Type

-- The balanced chemical equation as a condition
axiom balanced_equation : ∀ (C6H6 CH4 C7H8 H2 : ℕ), C6H6 + CH4 = C7H8 + H2

-- The proof problem: Prove the amount of Benzene required
theorem amount_of_benzene (moles_methane : ℕ) (moles_toluene : ℕ) (moles_hydrogen : ℕ) :
  moles_methane = 2 → moles_toluene = 2 → moles_hydrogen = 2 → 
  ∃ moles_benzene : ℕ, moles_benzene = 2 := by
  sorry

end amount_of_benzene_l145_145715


namespace jane_mean_score_l145_145551

def quiz_scores : List ℕ := [85, 90, 95, 80, 100]

def total_scores : ℕ := quiz_scores.length

def sum_scores : ℕ := quiz_scores.sum

def mean_score : ℕ := sum_scores / total_scores

theorem jane_mean_score : mean_score = 90 := by
  sorry

end jane_mean_score_l145_145551


namespace largest_angle_in_triangle_l145_145617

noncomputable def angle_sum : ℝ := 120 -- $\frac{4}{3}$ of 90 degrees
noncomputable def angle_difference : ℝ := 20

theorem largest_angle_in_triangle :
  ∃ (a b c : ℝ), a + b + c = 180 ∧ a + b = angle_sum ∧ b = a + angle_difference ∧
  max a (max b c) = 70 :=
by
  sorry

end largest_angle_in_triangle_l145_145617


namespace p_sufficient_condition_neg_q_l145_145403

variables (p q : Prop)

theorem p_sufficient_condition_neg_q (hnecsuff_q : ¬p → q) (hnecsuff_p : ¬q → p) : (p → ¬q) :=
by
  sorry

end p_sufficient_condition_neg_q_l145_145403


namespace probability_of_pink_l145_145454

variable (B P : ℕ) -- number of blue and pink gumballs
variable (h_total : B + P > 0) -- there is at least one gumball in the jar
variable (h_prob_two_blue : (B / (B + P)) * (B / (B + P)) = 16 / 49) -- the probability of drawing two blue gumballs in a row

theorem probability_of_pink : (P / (B + P)) = 3 / 7 :=
sorry

end probability_of_pink_l145_145454


namespace original_selling_price_l145_145041

theorem original_selling_price (P : ℝ) (S : ℝ) (h1 : S = 1.10 * P) (h2 : 1.17 * P = 1.10 * P + 35) : S = 550 := 
by
  sorry

end original_selling_price_l145_145041


namespace sum_of_three_numbers_l145_145681

variable {a b c : ℝ}

theorem sum_of_three_numbers :
  a^2 + b^2 + c^2 = 99 ∧ ab + bc + ca = 131 → a + b + c = 19 :=
by
  sorry

end sum_of_three_numbers_l145_145681


namespace number_of_positive_integers_l145_145988

theorem number_of_positive_integers (n : ℕ) : ∃! k : ℕ, k = 5 ∧
  (∀ n : ℕ, (1 ≤ n) → (12 % (n + 1) = 0)) :=
sorry

end number_of_positive_integers_l145_145988


namespace annie_accident_chance_l145_145808

def temperature_effect (temp: ℤ) : ℚ := ((32 - temp) / 3 * 5)

def road_condition_effect (condition: ℚ) : ℚ := condition

def wind_speed_effect (speed: ℤ) : ℚ := if (speed > 20) then ((speed - 20) / 10 * 3) else 0

def skid_chance (temp: ℤ) (condition: ℚ) (speed: ℤ) : ℚ :=
  temperature_effect temp + road_condition_effect condition + wind_speed_effect speed

def accident_chance (skid_chance: ℚ) (tire_effect: ℚ) : ℚ :=
  skid_chance * tire_effect

theorem annie_accident_chance :
  (temperature_effect 8 + road_condition_effect 15 + wind_speed_effect 35) * 0.75 = 43.5 :=
by sorry

end annie_accident_chance_l145_145808


namespace fruit_salad_weight_l145_145995

theorem fruit_salad_weight (melon berries : ℝ) (h_melon : melon = 0.25) (h_berries : berries = 0.38) : melon + berries = 0.63 :=
by
  sorry

end fruit_salad_weight_l145_145995


namespace isosceles_triangle_congruent_l145_145891

theorem isosceles_triangle_congruent (A B C C1 : ℝ) 
(h₁ : A = B) 
(h₂ : C = C1) 
: A = B ∧ C = C1 :=
by
  sorry

end isosceles_triangle_congruent_l145_145891


namespace sum_of_x_coordinates_of_other_vertices_l145_145645

theorem sum_of_x_coordinates_of_other_vertices {x1 y1 x2 y2 x3 y3 x4 y4: ℝ} 
    (h1 : (x1, y1) = (2, 12))
    (h2 : (x2, y2) = (8, 3))
    (midpoint_eq : (x1 + x2) / 2 = (x3 + x4) / 2) 
    : x3 + x4 = 10 := 
by
  have h4 : (2 + 8) / 2 = 5 := by norm_num
  have h5 : 2 * 5 = 10 := by norm_num
  sorry

end sum_of_x_coordinates_of_other_vertices_l145_145645


namespace total_yards_of_fabric_l145_145291

theorem total_yards_of_fabric (cost_checkered : ℝ) (cost_plain : ℝ) (price_per_yard : ℝ)
  (h1 : cost_checkered = 75) (h2 : cost_plain = 45) (h3 : price_per_yard = 7.50) :
  (cost_checkered / price_per_yard) + (cost_plain / price_per_yard) = 16 := 
by
  sorry

end total_yards_of_fabric_l145_145291


namespace mark_total_cans_l145_145646

theorem mark_total_cans (p1 p2 p3 p4 p5 p6 : ℕ) (c1 c2 c3 c4 c5 c6 : ℕ)
  (h1 : p1 = 30) (h2 : p2 = 25) (h3 : p3 = 35) (h4 : p4 = 40) 
  (h5 : p5 = 28) (h6 : p6 = 32) (hc1 : c1 = 12) (hc2 : c2 = 10) 
  (hc3 : c3 = 15) (hc4 : c4 = 14) (hc5 : c5 = 11) (hc6 : c6 = 13) :
  p1 * c1 + p2 * c2 + p3 * c3 + p4 * c4 + p5 * c5 + p6 * c6 = 2419 := 
by 
  sorry

end mark_total_cans_l145_145646


namespace triangle_projection_inequality_l145_145686

variable (a b c t r μ : ℝ)
variable (h1 : AC_1 = 2 * t * AB)
variable (h2 : BA_1 = 2 * r * BC)
variable (h3 : CB_1 = 2 * μ * AC)
variable (h4 : AB = c)
variable (h5 : AC = b)
variable (h6 : BC = a)

theorem triangle_projection_inequality
  (h1 : AC_1 = 2 * t * AB)  -- condition AC_1 = 2t * AB
  (h2 : BA_1 = 2 * r * BC)  -- condition BA_1 = 2r * BC
  (h3 : CB_1 = 2 * μ * AC)  -- condition CB_1 = 2μ * AC
  (h4 : AB = c)             -- side AB
  (h5 : AC = b)             -- side AC
  (h6 : BC = a)             -- side BC
  : (a^2 / b^2) * (t / (1 - 2 * t))^2 
  + (b^2 / c^2) * (r / (1 - 2 * r))^2 
  + (c^2 / a^2) * (μ / (1 - 2 * μ))^2 
  + 16 * t * r * μ ≥ 1 := 
  sorry

end triangle_projection_inequality_l145_145686


namespace joe_egg_count_l145_145764

theorem joe_egg_count : 
  let clubhouse : ℕ := 12
  let park : ℕ := 5
  let townhall : ℕ := 3
  clubhouse + park + townhall = 20 :=
by
  sorry

end joe_egg_count_l145_145764


namespace geometric_sequence_min_value_l145_145647

theorem geometric_sequence_min_value
  (q : ℝ) (a : ℕ → ℝ)
  (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n)
  (h_geom : ∀ k, a k = q ^ k)
  (h_eq : a m * (a n) ^ 2 = (a 4) ^ 2)
  (h_sum : m + 2 * n = 8) :
  ∀ (f : ℝ), f = (2 / m + 1 / n) → f ≥ 1 :=
by
  sorry

end geometric_sequence_min_value_l145_145647


namespace list_size_is_2017_l145_145906

def has_sum (L : List ℤ) (n : ℤ) : Prop :=
  List.sum L = n

def has_product (L : List ℤ) (n : ℤ) : Prop :=
  List.prod L = n

def includes (L : List ℤ) (n : ℤ) : Prop :=
  n ∈ L

theorem list_size_is_2017 
(L : List ℤ) :
  has_sum L 2018 ∧ 
  has_product L 2018 ∧ 
  includes L 2018 
  → L.length = 2017 :=
by 
  sorry

end list_size_is_2017_l145_145906


namespace total_profit_is_28000_l145_145036

noncomputable def investment_A (investment_B : ℝ) : ℝ := 3 * investment_B
noncomputable def period_A (period_B : ℝ) : ℝ := 2 * period_B
noncomputable def profit_B : ℝ := 4000
noncomputable def total_profit (investment_B period_B : ℝ) : ℝ :=
  let x := investment_B * period_B
  let a_share := 6 * x
  profit_B + a_share

theorem total_profit_is_28000 (investment_B period_B : ℝ) : 
  total_profit investment_B period_B = 28000 :=
by
  have h1 : profit_B = 4000 := rfl
  have h2 : investment_A investment_B = 3 * investment_B := rfl
  have h3 : period_A period_B = 2 * period_B := rfl
  simp [total_profit, h1, h2, h3]
  have x_def : investment_B * period_B = 4000 := by sorry
  simp [x_def]
  sorry

end total_profit_is_28000_l145_145036


namespace highest_power_of_2_dividing_15_pow_4_minus_9_pow_4_l145_145477

theorem highest_power_of_2_dividing_15_pow_4_minus_9_pow_4 :
  (∃ k, 15^4 - 9^4 = 2^k * m ∧ ¬ ∃ m', m = 2 * m') ∧ (k = 5) :=
by
  sorry

end highest_power_of_2_dividing_15_pow_4_minus_9_pow_4_l145_145477


namespace rectangle_length_increase_l145_145164

variable (L B : ℝ) -- Original length and breadth
variable (A : ℝ) -- Original area
variable (p : ℝ) -- Percentage increase in length
variable (A' : ℝ) -- New area

theorem rectangle_length_increase (hA : A = L * B) 
  (hp : L' = L + (p / 100) * L) 
  (hB' : B' = B * 0.9) 
  (hA' : A' = 1.035 * A)
  (hl' : L' = (1 + (p / 100)) * L)
  (hb_length : L' * B' = A') :
  p = 15 :=
by
  sorry

end rectangle_length_increase_l145_145164


namespace haily_cheapest_salon_l145_145854

def cost_Gustran : ℕ := 45 + 22 + 30
def cost_Barbara : ℕ := 40 + 30 + 28
def cost_Fancy : ℕ := 30 + 34 + 20

theorem haily_cheapest_salon : min (min cost_Gustran cost_Barbara) cost_Fancy = 84 := by
  sorry

end haily_cheapest_salon_l145_145854


namespace algebraic_expression_value_l145_145448

theorem algebraic_expression_value (m : ℝ) (h : (2018 + m) * (2020 + m) = 2) : (2018 + m)^2 + (2020 + m)^2 = 8 :=
by
  sorry

end algebraic_expression_value_l145_145448


namespace pipe_empty_cistern_l145_145187

theorem pipe_empty_cistern (h : 1 / 3 * t = 6) : 2 / 3 * t = 12 :=
sorry

end pipe_empty_cistern_l145_145187


namespace fruitseller_apples_l145_145924

theorem fruitseller_apples (x : ℝ) (sold_percent remaining_apples : ℝ) 
  (h_sold : sold_percent = 0.80) 
  (h_remaining : remaining_apples = 500) 
  (h_equation : (1 - sold_percent) * x = remaining_apples) : 
  x = 2500 := 
by 
  sorry

end fruitseller_apples_l145_145924


namespace micah_water_l145_145442

theorem micah_water (x : ℝ) (h1 : 3 * x + x = 6) : x = 1.5 :=
sorry

end micah_water_l145_145442


namespace color_change_probability_is_correct_l145_145641

-- Given definitions
def cycle_time : ℕ := 45 + 5 + 10 + 40

def favorable_time : ℕ := 5 + 5 + 5

def probability_color_change : ℚ := favorable_time / cycle_time

-- Theorem statement to prove the probability
theorem color_change_probability_is_correct :
  probability_color_change = 0.15 := 
sorry

end color_change_probability_is_correct_l145_145641


namespace solve_x_l145_145098

theorem solve_x : ∀ (x y : ℝ), (3 * x - y = 7) ∧ (x + 3 * y = 6) → x = 27 / 10 :=
by
  intros x y h
  sorry

end solve_x_l145_145098


namespace total_bill_l145_145485

-- Definitions from conditions
def num_people : ℕ := 3
def amount_per_person : ℕ := 45

-- Mathematical proof problem statement
theorem total_bill : num_people * amount_per_person = 135 := by
  sorry

end total_bill_l145_145485


namespace tangent_line_at_point_l145_145856

noncomputable def f : ℝ → ℝ := λ x => 2 * Real.log x + x^2 

def tangent_line_equation (x y : ℝ) : Prop :=
  4 * x - y - 3 = 0 

theorem tangent_line_at_point {x y : ℝ} (h : f 1 = 1) : 
  tangent_line_equation 1 1 ∧
  y = 4 * (x - 1) + 1 := 
sorry

end tangent_line_at_point_l145_145856


namespace rolls_to_neighbor_l145_145049

theorem rolls_to_neighbor (total_needed rolls_to_grandmother rolls_to_uncle rolls_needed : ℕ) (h1 : total_needed = 45) (h2 : rolls_to_grandmother = 1) (h3 : rolls_to_uncle = 10) (h4 : rolls_needed = 28) :
  total_needed - rolls_needed - (rolls_to_grandmother + rolls_to_uncle) = 6 := by
  sorry

end rolls_to_neighbor_l145_145049


namespace sufficient_not_necessary_condition_l145_145725

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, abs (x - 1) < 3 → (x + 2) * (x + a) < 0) ∧ 
  (∃ x : ℝ, (x + 2) * (x + a) < 0 ∧ ¬(abs (x - 1) < 3)) →
  a < -4 :=
by
  sorry

end sufficient_not_necessary_condition_l145_145725


namespace right_triangle_area_integer_l145_145669

theorem right_triangle_area_integer (a b c : ℤ) (h : a * a + b * b = c * c) : ∃ (n : ℤ), (1 / 2 : ℚ) * a * b = ↑n := 
sorry

end right_triangle_area_integer_l145_145669


namespace solve_for_x_l145_145212

theorem solve_for_x : ∀ (x : ℤ), (5 * x - 2) * 4 = (3 * (6 * x - 6)) → x = -5 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l145_145212


namespace jerky_batch_size_l145_145881

theorem jerky_batch_size
  (total_order_bags : ℕ)
  (initial_bags : ℕ)
  (days_to_fulfill : ℕ)
  (remaining_bags : ℕ := total_order_bags - initial_bags)
  (production_per_day : ℕ := remaining_bags / days_to_fulfill) :
  total_order_bags = 60 →
  initial_bags = 20 →
  days_to_fulfill = 4 →
  production_per_day = 10 :=
by
  intros
  sorry

end jerky_batch_size_l145_145881


namespace red_tickets_for_one_yellow_l145_145713

-- Define the conditions given in the problem
def yellow_needed := 10
def red_for_yellow (R : ℕ) := R -- This function defines the number of red tickets for one yellow
def blue_for_red := 10

def toms_yellow := 8
def toms_red := 3
def toms_blue := 7
def blue_needed := 163

-- Define the target function that converts the given conditions into a statement.
def red_tickets_for_yellow_proof : Prop :=
  ∀ R : ℕ, (2 * R = 14) → (R = 7)

-- Statement for proof where the condition leads to conclusion
theorem red_tickets_for_one_yellow : red_tickets_for_yellow_proof :=
by
  intros R h
  rw [← h, mul_comm] at h
  sorry

end red_tickets_for_one_yellow_l145_145713


namespace min_abs_sum_l145_145661

theorem min_abs_sum (x : ℝ) : ∃ y : ℝ, y = min ((|x+1| + |x-2| + |x-3|)) 4 :=
sorry

end min_abs_sum_l145_145661


namespace x_y_ge_two_l145_145421

open Real

theorem x_y_ge_two (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + x * y = 3) : 
  x + y ≥ 2 ∧ (x + y = 2 → x = 1 ∧ y = 1) :=
by {
 sorry
}

end x_y_ge_two_l145_145421


namespace find_x_l145_145679

theorem find_x (x : ℤ) (h : 4 * x - 23 = 33) : x = 14 := 
by 
  sorry

end find_x_l145_145679


namespace no_natural_m_n_exists_l145_145231

theorem no_natural_m_n_exists (m n : ℕ) : 
  (0.07 = (1 : ℝ) / m + (1 : ℝ) / n) → False :=
by
  -- Normally, the proof would go here, but it's not required by the prompt
  sorry

end no_natural_m_n_exists_l145_145231


namespace percentage_of_600_equals_150_is_25_l145_145949

theorem percentage_of_600_equals_150_is_25 : (150 / 600 * 100) = 25 := by
  sorry

end percentage_of_600_equals_150_is_25_l145_145949


namespace sin_double_angle_shift_l145_145162

variable (θ : Real)

theorem sin_double_angle_shift (h : Real.cos (θ + Real.pi) = -1 / 3) :
  Real.sin (2 * θ + Real.pi / 2) = -7 / 9 := 
by 
  sorry

end sin_double_angle_shift_l145_145162


namespace first_discount_percentage_l145_145611

theorem first_discount_percentage (D : ℝ) :
  (345 * (1 - D / 100) * 0.75 = 227.70) → (D = 12) :=
by
  intro cond
  sorry

end first_discount_percentage_l145_145611


namespace geometric_sequence_product_l145_145766

theorem geometric_sequence_product
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (hA_seq : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1))
  (hA_not_zero : ∀ n, a n ≠ 0)
  (h_condition : a 4 - 2 * (a 7)^2 + 3 * a 8 = 0)
  (hB_seq : ∀ n, b n = b 1 * (b 2 / b 1)^(n - 1))
  (hB7 : b 7 = a 7) :
  b 3 * b 7 * b 11 = 8 := 
sorry

end geometric_sequence_product_l145_145766


namespace point_in_fourth_quadrant_l145_145698

theorem point_in_fourth_quadrant (a b : ℝ) (h1 : a^2 + 1 > 0) (h2 : -1 - b^2 < 0) : 
  (a^2 + 1 > 0 ∧ -1 - b^2 < 0) ∧ (0 < a^2 + 1) ∧ (-1 - b^2 < 0) :=
by
  sorry

end point_in_fourth_quadrant_l145_145698


namespace area_of_triangle_ABC_l145_145858

theorem area_of_triangle_ABC (BD CE : ℝ) (angle_BD_CE : ℝ) (BD_len : BD = 9) (CE_len : CE = 15) (angle_BD_CE_deg : angle_BD_CE = 60) : 
  ∃ area : ℝ, 
    area = 90 * Real.sqrt 3 := 
by
  sorry

end area_of_triangle_ABC_l145_145858


namespace fermat_numbers_coprime_l145_145853

theorem fermat_numbers_coprime (n m : ℕ) (h : n ≠ m) :
  Nat.gcd (2 ^ 2 ^ (n - 1) + 1) (2 ^ 2 ^ (m - 1) + 1) = 1 :=
sorry

end fermat_numbers_coprime_l145_145853


namespace compare_neg_rational_decimal_l145_145189

theorem compare_neg_rational_decimal : 
  -3 / 4 > -0.8 := 
by 
  sorry

end compare_neg_rational_decimal_l145_145189


namespace incorrect_statement_A_l145_145085

-- conditions as stated in the table
def spring_length (x : ℕ) : ℝ :=
  if x = 0 then 20
  else if x = 1 then 20.5
  else if x = 2 then 21
  else if x = 3 then 21.5
  else if x = 4 then 22
  else if x = 5 then 22.5
  else 0 -- assuming 0 for out of range for simplicity

-- questions with answers
-- Prove that statement A is incorrect
theorem incorrect_statement_A : ¬ (spring_length 0 = 20) := by
  sorry

end incorrect_statement_A_l145_145085


namespace trapezoid_third_largest_angle_l145_145335

theorem trapezoid_third_largest_angle (a d : ℝ)
  (h1 : 2 * a + 3 * d = 200)      -- Condition: 2a + 3d = 200°
  (h2 : a + d = 70) :             -- Condition: a + d = 70°
  a + 2 * d = 130 :=              -- Question: Prove a + 2d = 130°
by
  sorry

end trapezoid_third_largest_angle_l145_145335


namespace sprint_team_total_miles_l145_145486

theorem sprint_team_total_miles (number_of_people : ℝ) (miles_per_person : ℝ) 
  (h1 : number_of_people = 150.0) (h2 : miles_per_person = 5.0) : 
  number_of_people * miles_per_person = 750.0 :=
by
  rw [h1, h2]
  norm_num

end sprint_team_total_miles_l145_145486


namespace john_max_books_l145_145905

theorem john_max_books (h₁ : 4575 ≥ 0) (h₂ : 325 > 0) : 
  ∃ (x : ℕ), x = 14 ∧ ∀ n : ℕ, n ≤ x ↔ n * 325 ≤ 4575 := 
  sorry

end john_max_books_l145_145905


namespace problem_statement_l145_145931

-- Define what it means for a number's tens and ones digits to have a sum of 13
def sum_of_tens_and_ones_equals (n : ℕ) (s : ℕ) : Prop :=
  let tens_digit := (n / 10) % 10
  let ones_digit := n % 10
  tens_digit + ones_digit = s

-- State the theorem with the given conditions and correct answer
theorem problem_statement : sum_of_tens_and_ones_equals (6^11) 13 :=
sorry

end problem_statement_l145_145931


namespace sum_of_squares_l145_145123

theorem sum_of_squares :
  (2^2 + 1^2 + 0^2 + (-1)^2 + (-2)^2 = 10) :=
by
  sorry

end sum_of_squares_l145_145123


namespace function_increasing_l145_145556

noncomputable def f (a x : ℝ) := x^2 + a * x + 1 / x

theorem function_increasing (a : ℝ) :
  (∀ x, (1 / 3) < x → 0 ≤ (2 * x + a - 1 / x^2)) → a ≥ 25 / 3 :=
by
  sorry

end function_increasing_l145_145556


namespace num_int_values_satisfying_inequality_l145_145998

theorem num_int_values_satisfying_inequality (x : ℤ) :
  (x^2 < 9 * x) ↔ (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8) := 
sorry

end num_int_values_satisfying_inequality_l145_145998


namespace circle_passing_origin_l145_145281

theorem circle_passing_origin (a b r : ℝ) :
  ((a^2 + b^2 = r^2) ↔ (∃ (x y : ℝ), (x-a)^2 + (y-b)^2 = r^2 ∧ x = 0 ∧ y = 0)) :=
by
  sorry

end circle_passing_origin_l145_145281


namespace problem_equiv_l145_145692

theorem problem_equiv (a b : ℝ) (h : a ≠ b) : 
  (a^2 - 4 * a + 5 > 0) ∧ (a^2 + b^2 ≥ 2 * (a - b - 1)) :=
by {
  sorry
}

end problem_equiv_l145_145692


namespace range_of_k_l145_145734
noncomputable def quadratic_nonnegative (k : ℝ) : Prop :=
  ∀ x : ℝ, k * x^2 - 4 * x + 3 ≥ 0

theorem range_of_k (k : ℝ) : quadratic_nonnegative k ↔ k ∈ Set.Ici (4 / 3) :=
by
  sorry

end range_of_k_l145_145734


namespace find_possible_f_one_l145_145237

noncomputable def f : ℝ → ℝ := sorry

theorem find_possible_f_one (h : ∀ x y : ℝ, f (x + y) = 2 * f x * f y) :
  f 1 = 0 ∨ (∃ c : ℝ, f 0 = 1/2 ∧ f 1 = c) :=
sorry

end find_possible_f_one_l145_145237


namespace initial_markup_percentage_l145_145331

theorem initial_markup_percentage (C M : ℝ) 
  (h1 : C > 0) 
  (h2 : (1 + M) * 1.25 * 0.92 = 1.38) :
  M = 0.2 :=
sorry

end initial_markup_percentage_l145_145331


namespace tan_a1_a13_eq_sqrt3_l145_145907

-- Definition of required constants and properties of the geometric sequence
noncomputable def a (n : Nat) : ℝ := sorry -- Geometric sequence definition (abstract)

-- Given condition: a_3 * a_11 + 2 * a_7^2 = 4π
axiom geom_seq_cond : a 3 * a 11 + 2 * (a 7)^2 = 4 * Real.pi

-- Property of geometric sequence: a_3 * a_11 = a_7^2
axiom geom_seq_property : a 3 * a 11 = (a 7)^2

-- To prove: tan(a_1 * a_13) = √3
theorem tan_a1_a13_eq_sqrt3 : Real.tan (a 1 * a 13) = Real.sqrt 3 := by
  sorry

end tan_a1_a13_eq_sqrt3_l145_145907


namespace complex_multiplication_l145_145801

theorem complex_multiplication (i : ℂ) (hi : i^2 = -1) : (1 + i) * (1 - i) = 1 := 
by
  sorry

end complex_multiplication_l145_145801


namespace A_share_of_profit_l145_145986

-- Define the conditions
def A_investment : ℕ := 100
def A_months : ℕ := 12
def B_investment : ℕ := 200
def B_months : ℕ := 6
def total_profit : ℕ := 100

-- Calculate the weighted investments (directly from conditions)
def A_weighted_investment : ℕ := A_investment * A_months
def B_weighted_investment : ℕ := B_investment * B_months
def total_weighted_investment : ℕ := A_weighted_investment + B_weighted_investment

-- Prove A's share of the profit
theorem A_share_of_profit : (A_weighted_investment / total_weighted_investment : ℚ) * total_profit = 50 := by
  -- The proof will go here
  sorry

end A_share_of_profit_l145_145986


namespace product_divisibility_l145_145139

theorem product_divisibility (a b c : ℤ)
  (h₁ : (a + b + c) ^ 2 = -(a * b + a * c + b * c))
  (h₂ : a + b ≠ 0)
  (h₃ : b + c ≠ 0)
  (h₄ : a + c ≠ 0) :
  (a + b) * (a + c) % (b + c) = 0 ∧
  (a + b) * (b + c) % (a + c) = 0 ∧
  (a + c) * (b + c) % (a + b) = 0 := by
  sorry

end product_divisibility_l145_145139


namespace calculate_visits_to_water_fountain_l145_145224

-- Define the distance from the desk to the fountain
def distance_desk_to_fountain : ℕ := 30

-- Define the total distance Mrs. Hilt walked
def total_distance_walked : ℕ := 120

-- Define the distance of a round trip (desk to fountain and back)
def round_trip_distance : ℕ := 2 * distance_desk_to_fountain

-- Define the number of round trips and hence the number of times to water fountain
def number_of_visits : ℕ := total_distance_walked / round_trip_distance

theorem calculate_visits_to_water_fountain:
    number_of_visits = 2 := 
by
    sorry

end calculate_visits_to_water_fountain_l145_145224


namespace determine_m_l145_145848

-- Define f and g according to the given conditions
def f (x : ℝ) (m : ℝ) : ℝ := x^2 - 3 * x + m
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - 3 * x + 5 * m

-- Define the value of x
def x := 5

-- State the main theorem we need to prove
theorem determine_m 
  (h : 3 * f x m = 2 * g x m) : m = 10 / 7 :=
by
  -- Proof is omitted
  sorry

end determine_m_l145_145848


namespace a_minus_b_l145_145589

noncomputable def find_a_b (a b : ℝ) :=
  ∃ k : ℝ, ∀ (x : ℝ) (y : ℝ), 
    (k = 2 + a) ∧ 
    (y = k * x + 1) ∧ 
    (y = x^2 + a * x + b) ∧ 
    (x = 1) ∧ (y = 3)

theorem a_minus_b (a b : ℝ) (h : find_a_b a b) : a - b = -2 := by 
  sorry

end a_minus_b_l145_145589


namespace average_of_three_quantities_l145_145976

theorem average_of_three_quantities (a b c d e : ℝ) 
  (h_avg_5 : (a + b + c + d + e) / 5 = 11)
  (h_avg_2 : (d + e) / 2 = 21.5) :
  (a + b + c) / 3 = 4 :=
by
  sorry

end average_of_three_quantities_l145_145976


namespace tail_growth_problem_l145_145219

def initial_tail_length : ℕ := 1
def final_tail_length : ℕ := 864
def transformations (ordinary_count cowardly_count : ℕ) : ℕ := initial_tail_length * 2^ordinary_count * 3^cowardly_count

theorem tail_growth_problem (ordinary_count cowardly_count : ℕ) :
  transformations ordinary_count cowardly_count = final_tail_length ↔ ordinary_count = 5 ∧ cowardly_count = 3 :=
by
  sorry

end tail_growth_problem_l145_145219


namespace who_wears_which_dress_l145_145366

-- Define the possible girls
inductive Girl
| Katya | Olya | Liza | Rita
deriving DecidableEq

-- Define the possible dresses
inductive Dress
| Pink | Green | Yellow | Blue
deriving DecidableEq

-- Define the fact that each girl is wearing a dress
structure Wearing (girl : Girl) (dress : Dress) : Prop

-- Define the conditions
theorem who_wears_which_dress :
  (¬ Wearing Girl.Katya Dress.Pink ∧ ¬ Wearing Girl.Katya Dress.Blue) ∧
  (∀ g1 g2 g3, Wearing g1 Dress.Green → (Wearing g2 Dress.Pink ∧ Wearing g3 Dress.Yellow → (g2 = Girl.Liza ∧ (g3 = Girl.Rita)) ∨ (g3 = Girl.Liza ∧ g2 = Girl.Rita))) ∧
  (¬ Wearing Girl.Rita Dress.Green ∧ ¬ Wearing Girl.Rita Dress.Blue) ∧
  (∀ g1 g2, (Wearing g1 Dress.Pink ∧ Wearing g2 Dress.Yellow) → Girl.Olya = g2 ∧ Girl.Rita = g1) →
  (Wearing Girl.Katya Dress.Green ∧ Wearing Girl.Olya Dress.Blue ∧ Wearing Girl.Liza Dress.Pink ∧ Wearing Girl.Rita Dress.Yellow) :=
by
  sorry

end who_wears_which_dress_l145_145366


namespace remainder_3_pow_405_mod_13_l145_145209

theorem remainder_3_pow_405_mod_13 : (3^405) % 13 = 1 :=
by
  sorry

end remainder_3_pow_405_mod_13_l145_145209


namespace probability_correct_l145_145761

-- Define the conditions of the problem
def total_white_balls : ℕ := 6
def total_black_balls : ℕ := 5
def total_balls : ℕ := total_white_balls + total_black_balls
def total_ways_draw_two_balls : ℕ := Nat.choose total_balls 2
def ways_choose_one_white_ball : ℕ := Nat.choose total_white_balls 1
def ways_choose_one_black_ball : ℕ := Nat.choose total_black_balls 1
def total_successful_outcomes : ℕ := ways_choose_one_white_ball * ways_choose_one_black_ball

-- Define the probability calculation
def probability_drawing_one_white_one_black : ℚ := total_successful_outcomes / total_ways_draw_two_balls

-- State the theorem
theorem probability_correct :
  probability_drawing_one_white_one_black = 6 / 11 :=
by
  sorry

end probability_correct_l145_145761


namespace BC_total_750_l145_145111

theorem BC_total_750 (A B C : ℤ) 
  (h1 : A + B + C = 900) 
  (h2 : A + C = 400) 
  (h3 : C = 250) : 
  B + C = 750 := 
by 
  sorry

end BC_total_750_l145_145111


namespace find_A_and_B_l145_145185

theorem find_A_and_B (A B : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ -6 → (5 * x - 3) / (x^2 + 3 * x - 18) = A / (x - 3) + B / (x + 6)) →
  A = 4 / 3 ∧ B = 11 / 3 :=
by
  intros h
  sorry

end find_A_and_B_l145_145185


namespace value_of_a1_plus_a3_l145_145364

theorem value_of_a1_plus_a3 (a a1 a2 a3 a4 : ℝ) :
  (∀ x : ℝ, (1 + x)^4 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) →
  a1 + a3 = 8 :=
by
  sorry

end value_of_a1_plus_a3_l145_145364


namespace failing_percentage_exceeds_35_percent_l145_145571

theorem failing_percentage_exceeds_35_percent:
  ∃ (n D A B failD failA : ℕ), 
  n = 25 ∧
  D + A - B = n ∧
  (failD * 100) / D = 30 ∧
  (failA * 100) / A = 30 ∧
  ((failD + failA) * 100) / n > 35 := 
by
  sorry

end failing_percentage_exceeds_35_percent_l145_145571


namespace n_minus_two_is_square_of_natural_number_l145_145922

theorem n_minus_two_is_square_of_natural_number 
  (n m : ℕ) 
  (hn: n ≥ 3) 
  (hm: m = n * (n - 1) / 2) 
  (hm_odd: m % 2 = 1)
  (unique_rem: ∀ i j : ℕ, i ≠ j → (i + j) % m ≠ (i + j) % m) :
  ∃ k : ℕ, n - 2 = k * k := 
sorry

end n_minus_two_is_square_of_natural_number_l145_145922


namespace min_value_2x_plus_y_l145_145184

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 2 / (y + 1) = 2) :
  2 * x + y = 3 :=
sorry

end min_value_2x_plus_y_l145_145184


namespace min_sum_m_n_l145_145020

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem min_sum_m_n (m n : ℕ) (h : (binomial m 2) * 2 = binomial (m + n) 2) : m + n = 4 := by
  sorry

end min_sum_m_n_l145_145020


namespace exponential_function_inequality_l145_145903

theorem exponential_function_inequality {a : ℝ} (h0 : 0 < a) (h1 : a < 1) :
  (a^3) * (a^2) < a^2 :=
by
  sorry

end exponential_function_inequality_l145_145903


namespace birds_on_fence_l145_145796

theorem birds_on_fence (B S : ℕ): 
  S = 3 →
  S + 6 = B + 5 →
  B = 4 :=
by
  intros h1 h2
  sorry

end birds_on_fence_l145_145796


namespace debby_deleted_pictures_l145_145648

theorem debby_deleted_pictures :
  ∀ (zoo_pics museum_pics remaining_pics : ℕ), 
  zoo_pics = 24 →
  museum_pics = 12 →
  remaining_pics = 22 →
  (zoo_pics + museum_pics) - remaining_pics = 14 :=
by
  intros zoo_pics museum_pics remaining_pics hz hm hr
  sorry

end debby_deleted_pictures_l145_145648


namespace probability_quarter_circle_is_pi_div_16_l145_145911

open Real

noncomputable def probability_quarter_circle : ℝ :=
  let side_length := 2
  let total_area := side_length * side_length
  let quarter_circle_area := π / 4
  quarter_circle_area / total_area

theorem probability_quarter_circle_is_pi_div_16 :
  probability_quarter_circle = π / 16 :=
by
  sorry

end probability_quarter_circle_is_pi_div_16_l145_145911


namespace cost_price_of_table_l145_145969

theorem cost_price_of_table (SP : ℝ) (CP : ℝ) (h1 : SP = 1.20 * CP) (h2 : SP = 3600) : CP = 3000 :=
by
  sorry

end cost_price_of_table_l145_145969


namespace radian_measure_of_neg_300_degrees_l145_145542

theorem radian_measure_of_neg_300_degrees : (-300 : ℝ) * (Real.pi / 180) = -5 * Real.pi / 3 :=
by
  sorry

end radian_measure_of_neg_300_degrees_l145_145542


namespace triangle_PQR_PR_value_l145_145352

theorem triangle_PQR_PR_value (PQ QR PR : ℕ) (h1 : PQ = 7) (h2 : QR = 20) (h3 : 13 < PR) (h4 : PR < 27) : PR = 21 :=
by sorry

end triangle_PQR_PR_value_l145_145352


namespace davids_profit_l145_145588

-- Definitions of conditions
def weight_of_rice : ℝ := 50
def cost_of_rice : ℝ := 50
def selling_price_per_kg : ℝ := 1.20

-- Theorem stating the expected profit
theorem davids_profit : 
  (selling_price_per_kg * weight_of_rice) - cost_of_rice = 10 := 
by 
  -- Proofs are omitted.
  sorry

end davids_profit_l145_145588


namespace pair_opposites_example_l145_145298

theorem pair_opposites_example :
  (-5)^2 = 25 ∧ -((5)^2) = -25 →
  (∀ a b : ℕ, (|-4|)^2 = 4^2 → 4^2 = 16 → |-4|^2 = 16) →
  (-3)^2 = 9 ∧ 3^2 = 9 →
  (-(|-2|)^2 = -4 ∧ -2^2 = -4) →
  25 = -(-25) :=
by
  sorry

end pair_opposites_example_l145_145298


namespace find_t_l145_145973

theorem find_t : ∀ (p j t x y a b c : ℝ),
  j = 0.75 * p →
  j = 0.80 * t →
  t = p - (t/100) * p →
  x = 0.10 * t →
  y = 0.50 * j →
  x + y = 12 →
  a = x + y →
  b = 0.15 * a →
  c = 2 * b →
  t = 24 := 
by
  intros p j t x y a b c hjp hjt htp hxt hyy hxy ha hb hc
  sorry

end find_t_l145_145973


namespace circle_area_greater_than_hexagon_area_l145_145871

theorem circle_area_greater_than_hexagon_area (h : ℝ) (r : ℝ) (π : ℝ) (sqrt3 : ℝ) (ratio : ℝ) : 
  (h = 1) →
  (r = sqrt3 / 2) →
  (π > 3) →
  (sqrt3 > 1.7) →
  (ratio = (π * sqrt3) / 6) →
  ratio > 0.9 :=
by
  intros h_eq r_eq pi_gt sqrt3_gt ratio_eq
  -- Proof omitted
  sorry

end circle_area_greater_than_hexagon_area_l145_145871


namespace yellow_balls_l145_145530

theorem yellow_balls (total_balls : ℕ) (prob_yellow : ℚ) (x : ℕ) :
  total_balls = 40 ∧ prob_yellow = 0.30 → (x : ℚ) = 12 := 
by 
  sorry

end yellow_balls_l145_145530


namespace ellipse_properties_l145_145401

theorem ellipse_properties (h k a b : ℝ)
  (h_eq : h = 1)
  (k_eq : k = -3)
  (a_eq : a = 7)
  (b_eq : b = 4) :
  h + k + a + b = 9 :=
by
  sorry

end ellipse_properties_l145_145401


namespace Sandy_age_l145_145280

variable (S M : ℕ)

def condition1 (S M : ℕ) : Prop := M = S + 18
def condition2 (S M : ℕ) : Prop := S * 9 = M * 7

theorem Sandy_age (h1 : condition1 S M) (h2 : condition2 S M) : S = 63 := sorry

end Sandy_age_l145_145280


namespace arithmetic_sequence_geometric_sequence_added_number_l145_145910

theorem arithmetic_sequence_geometric_sequence_added_number 
  (a : ℕ → ℤ)
  (h1 : a 1 = -8)
  (h2 : a 2 = -6)
  (h_arith : ∀ n, a n = -8 + (n-1) * 2)  -- derived from the conditions
  (x : ℤ)
  (h_geo : (-8 + x) * x = (-2 + x) * (-2 + x)) :
  x = -1 := 
sorry

end arithmetic_sequence_geometric_sequence_added_number_l145_145910


namespace factor_of_polynomial_l145_145133

theorem factor_of_polynomial (t : ℚ) : (8 * t^2 + 17 * t - 10 = 0) ↔ (t = 5/8 ∨ t = -2) :=
by sorry

end factor_of_polynomial_l145_145133


namespace value_of_a_b_c_l145_145733

theorem value_of_a_b_c 
  (a b c : ℤ) 
  (h1 : x^2 + 12*x + 35 = (x + a)*(x + b)) 
  (h2 : x^2 - 15*x + 56 = (x - b)*(x - c)) : 
  a + b + c = 20 := 
sorry

end value_of_a_b_c_l145_145733


namespace river_depth_l145_145662

theorem river_depth (V : ℝ) (W : ℝ) (F : ℝ) (D : ℝ) 
  (hV : V = 10666.666666666666) 
  (hW : W = 40) 
  (hF : F = 66.66666666666667) 
  (hV_eq : V = W * D * F) : 
  D = 4 :=
by sorry

end river_depth_l145_145662


namespace average_age_of_both_teams_l145_145028

theorem average_age_of_both_teams (n_men : ℕ) (age_men : ℕ) (n_women : ℕ) (age_women : ℕ) :
  n_men = 8 → age_men = 35 → n_women = 6 → age_women = 30 → 
  (8 * 35 + 6 * 30) / (8 + 6) = 32.857 := 
by
  intros h1 h2 h3 h4
  -- Proof is omitted
  sorry

end average_age_of_both_teams_l145_145028


namespace a_fraction_of_capital_l145_145693

theorem a_fraction_of_capital (T : ℝ) (B : ℝ) (C : ℝ) (D : ℝ)
  (profit_A : ℝ) (total_profit : ℝ)
  (h1 : B = T * (1 / 4))
  (h2 : C = T * (1 / 5))
  (h3 : D = T - (T * (1 / 4) + T * (1 / 5) + T * x))
  (h4 : profit_A = 805)
  (h5 : total_profit = 2415) :
  x = 161 / 483 :=
by
  sorry

end a_fraction_of_capital_l145_145693


namespace find_x_l145_145076

theorem find_x (x : ℝ) (hx_pos : x > 0) (hx_ceil_eq : ⌈x⌉ = 15) : x = 14 :=
by
  -- Define the condition
  have h_eq : ⌈x⌉ * x = 210 := sorry
  -- Prove that the only solution is x = 14
  sorry

end find_x_l145_145076


namespace proof_M1M2_product_l145_145445

theorem proof_M1M2_product : 
  (∀ x, (45 * x - 34) / (x^2 - 4 * x + 3) = M_1 / (x - 1) + M_2 / (x - 3)) →
  M_1 * M_2 = -1111 / 4 := 
by
  sorry

end proof_M1M2_product_l145_145445


namespace geometric_seq_arith_condition_half_l145_145419

variables {a : ℕ → ℝ} {q : ℝ}

-- Conditions from the problem
def geometric_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q
def positive_terms (a : ℕ → ℝ) := ∀ n, a n > 0
def arithmetic_condition (a : ℕ → ℝ) (q : ℝ) := 
  a 1 = q * a 0 ∧ (1/2 : ℝ) * a 2 = a 1 + 2 * a 0

-- The statement to be proven
theorem geometric_seq_arith_condition_half (a : ℕ → ℝ) (q : ℝ) :
  geometric_seq a q →
  positive_terms a →
  arithmetic_condition a q →
  q = 2 →
  (a 2 + a 3) / (a 3 + a 4) = 1 / 2 :=
by
  intros h1 h2 h3 hq
  sorry

end geometric_seq_arith_condition_half_l145_145419


namespace three_b_minus_a_eq_neg_five_l145_145077

theorem three_b_minus_a_eq_neg_five (a b : ℤ) (h : |a - 2| + (b + 1)^2 = 0) : 3 * b - a = -5 :=
sorry

end three_b_minus_a_eq_neg_five_l145_145077


namespace variance_scaled_l145_145744

-- Let V represent the variance of the set of data
def original_variance : ℝ := 3
def scale_factor : ℝ := 3

-- Prove that the new variance is 27 
theorem variance_scaled (V : ℝ) (s : ℝ) (hV : V = 3) (hs : s = 3) : s^2 * V = 27 := by
  sorry

end variance_scaled_l145_145744


namespace abs_diff_of_two_numbers_l145_145984

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 34) (h2 : x * y = 240) : abs (x - y) = 14 :=
by
  sorry

end abs_diff_of_two_numbers_l145_145984


namespace Andrews_age_l145_145319

theorem Andrews_age (a g : ℝ) (h1 : g = 15 * a) (h2 : g - a = 55) : a = 55 / 14 :=
by
  /- proof will go here -/
  sorry

end Andrews_age_l145_145319


namespace number_of_rectangles_is_24_l145_145211

-- Define the rectangles on a 1x5 stripe
def rectangles_1x5 : ℕ := 1 + 2 + 3 + 4 + 5

-- Define the rectangles on a 1x4 stripe
def rectangles_1x4 : ℕ := 1 + 2 + 3 + 4

-- Define the overlap (intersection) adjustment
def overlap_adjustment : ℕ := 1

-- Total number of rectangles calculation
def total_rectangles : ℕ := rectangles_1x5 + rectangles_1x4 - overlap_adjustment

theorem number_of_rectangles_is_24 : total_rectangles = 24 := by
  sorry

end number_of_rectangles_is_24_l145_145211


namespace collinear_points_k_value_l145_145908

theorem collinear_points_k_value : 
  (∀ k : ℝ, ∃ (a : ℝ) (b : ℝ), ∀ (x : ℝ) (y : ℝ),
    ((x, y) = (1, -2) ∨ (x, y) = (3, 2) ∨ (x, y) = (6, k / 3)) → y = a * x + b) → k = 24 :=
by
sorry

end collinear_points_k_value_l145_145908


namespace total_pies_l145_145642

def apple_Pies (totalApples : ℕ) (applesPerPie : ℕ) (piesPerBatch : ℕ) : ℕ :=
  (totalApples / applesPerPie) * piesPerBatch

def pear_Pies (totalPears : ℕ) (pearsPerPie : ℕ) (piesPerBatch : ℕ) : ℕ :=
  (totalPears / pearsPerPie) * piesPerBatch

theorem total_pies :
  let apples : ℕ := 27
  let pears : ℕ := 30
  let applesPerPie : ℕ := 9
  let pearsPerPie : ℕ := 15
  let applePiesPerBatch : ℕ := 2
  let pearPiesPerBatch : ℕ := 3
  apple_Pies apples applesPerPie applePiesPerBatch + pear_Pies pears pearsPerPie pearPiesPerBatch = 12 :=
by
  sorry

end total_pies_l145_145642


namespace product_of_roots_l145_145929

-- Let x₁ and x₂ be roots of the quadratic equation x^2 + x - 1 = 0
theorem product_of_roots (x₁ x₂ : ℝ) (h₁ : x₁^2 + x₁ - 1 = 0) (h₂ : x₂^2 + x₂ - 1 = 0) :
  x₁ * x₂ = -1 :=
sorry

end product_of_roots_l145_145929


namespace specific_gravity_cylinder_l145_145729

noncomputable def specific_gravity_of_cylinder (r m : ℝ) : ℝ :=
  (1 / 3) - (Real.sqrt 3 / (4 * Real.pi))

theorem specific_gravity_cylinder
  (r m : ℝ) 
  (cylinder_floats : r > 0 ∧ m > 0)
  (submersion_depth : r / 2 = r / 2) :
  specific_gravity_of_cylinder r m = 0.1955 :=
sorry

end specific_gravity_cylinder_l145_145729


namespace find_a_l145_145852

-- Define sets A and B based on the given real number a
def A (a : ℝ) : Set ℝ := {a^2, a + 1, -3}
def B (a : ℝ) : Set ℝ := {a - 3, 3 * a - 1, a^2 + 1}

-- Given condition
def condition (a : ℝ) : Prop := A a ∩ B a = {-3}

-- Prove that a = -2/3 is the solution satisfying the condition
theorem find_a : ∃ a : ℝ, condition a ∧ a = -2/3 :=
by
  sorry  -- Proof goes here

end find_a_l145_145852


namespace find_real_solutions_l145_145382

noncomputable def polynomial_expression (x : ℝ) : ℝ := (x - 2)^2 * (x - 4) * (x - 1)

theorem find_real_solutions :
  ∀ (x : ℝ), (x ≠ 3) ∧ (x ≠ 5) ∧ (polynomial_expression x = 1) ↔ (x = 1 ∨ x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2) := sorry

end find_real_solutions_l145_145382


namespace number_of_zeros_f_l145_145768

noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 + 2 * x + 5

theorem number_of_zeros_f : 
  (∃ a b : ℝ, f a = 0 ∧ f b = 0 ∧ 0 < a ∧ 0 < b ∧ a ≠ b) ∧ ∀ c, f c = 0 → c = a ∨ c = b :=
by
  sorry

end number_of_zeros_f_l145_145768


namespace find_m_l145_145639

theorem find_m (m : ℝ) (h1 : m^2 - 3 * m + 2 = 0) (h2 : m ≠ 1) : m = 2 :=
sorry

end find_m_l145_145639


namespace geometric_sequence_problem_l145_145318

noncomputable def geometric_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 1 then n * a1 else a1 * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_problem
  (a1 q : ℝ) (a2 : ℝ := a1 * q) (a5 : ℝ := a1 * q^4)
  (S2 : ℝ := geometric_sum a1 q 2) (S4 : ℝ := geometric_sum a1 q 4)
  (h1 : 8 * a2 + a5 = 0) :
  S4 / S2 = 5 :=
by
  sorry

end geometric_sequence_problem_l145_145318


namespace line_passes_through_fixed_point_l145_145469

theorem line_passes_through_fixed_point (k : ℝ) : (k * 2 - 1 + 1 - 2 * k = 0) :=
by
  sorry

end line_passes_through_fixed_point_l145_145469


namespace negation_of_proposition_l145_145704

-- Define the proposition P(x)
def P (x : ℝ) : Prop := x + Real.log x > 0

-- Translate the problem into lean
theorem negation_of_proposition :
  (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x := by
  sorry

end negation_of_proposition_l145_145704


namespace divide_54_degree_angle_l145_145497

theorem divide_54_degree_angle :
  ∃ (angle_div : ℝ), angle_div = 54 / 3 :=
by
  sorry

end divide_54_degree_angle_l145_145497


namespace second_player_always_wins_l145_145784

open Nat

theorem second_player_always_wins (cards : Finset ℕ) (h_card_count : cards.card = 16) :
  ∃ strategy : ℕ → ℕ, ∀ total_score : ℕ,
  total_score ≤ 22 → (total_score + strategy total_score > 22 ∨ 
  (∃ next_score : ℕ, total_score + next_score ≤ 22 ∧ strategy (total_score + next_score) = 1)) :=
sorry

end second_player_always_wins_l145_145784


namespace solve_equation_1_solve_equation_2_l145_145021

theorem solve_equation_1 (x : ℝ) : 2 * x^2 - x = 0 ↔ x = 0 ∨ x = 1 / 2 := 
by sorry

theorem solve_equation_2 (x : ℝ) : (2 * x + 1)^2 - 9 = 0 ↔ x = 1 ∨ x = -2 := 
by sorry

end solve_equation_1_solve_equation_2_l145_145021


namespace average_price_of_pig_l145_145299

theorem average_price_of_pig :
  ∀ (total_cost total_cost_hens total_cost_pigs : ℕ) (num_hens num_pigs avg_price_hen avg_price_pig : ℕ),
  num_hens = 10 →
  num_pigs = 3 →
  total_cost = 1200 →
  avg_price_hen = 30 →
  total_cost_hens = num_hens * avg_price_hen →
  total_cost_pigs = total_cost - total_cost_hens →
  avg_price_pig = total_cost_pigs / num_pigs →
  avg_price_pig = 300 :=
by
  intros total_cost total_cost_hens total_cost_pigs num_hens num_pigs avg_price_hen avg_price_pig h_num_hens h_num_pigs h_total_cost h_avg_price_hen h_total_cost_hens h_total_cost_pigs h_avg_price_pig
  sorry

end average_price_of_pig_l145_145299


namespace amount_paid_correct_l145_145293

-- Defining the conditions and constants
def hourly_rate : ℕ := 60
def hours_per_day : ℕ := 3
def total_days : ℕ := 14

-- The proof statement
theorem amount_paid_correct : hourly_rate * hours_per_day * total_days = 2520 := by
  sorry

end amount_paid_correct_l145_145293


namespace solve_trig_equation_l145_145441

theorem solve_trig_equation (x : ℝ) : 
  (∃ (k : ℤ), x = (Real.pi / 16) * (4 * k + 1)) ↔ 2 * (Real.cos (4 * x) - Real.sin x * Real.cos (3 * x)) = Real.sin (4 * x) + Real.sin (2 * x) :=
by
  -- The full proof detail goes here.
  sorry

end solve_trig_equation_l145_145441


namespace value_of_p_minus_q_plus_r_l145_145657

theorem value_of_p_minus_q_plus_r
  (p q r : ℚ)
  (h1 : 3 / p = 6)
  (h2 : 3 / q = 18)
  (h3 : 5 / r = 15) :
  p - q + r = 2 / 3 :=
by
  sorry

end value_of_p_minus_q_plus_r_l145_145657


namespace max_square_test_plots_l145_145178

theorem max_square_test_plots (length width fence : ℕ)
  (h_length : length = 36)
  (h_width : width = 66)
  (h_fence : fence = 2200) :
  ∃ (n : ℕ), n * (11 / 6) * n = 264 ∧
      (36 * n + (11 * n - 6) * 66) ≤ 2200 := sorry

end max_square_test_plots_l145_145178


namespace students_in_both_clubs_l145_145936

theorem students_in_both_clubs (total_students drama_club art_club drama_or_art in_both_clubs : ℕ)
  (H1 : total_students = 300)
  (H2 : drama_club = 120)
  (H3 : art_club = 150)
  (H4 : drama_or_art = 220) :
  in_both_clubs = drama_club + art_club - drama_or_art :=
by
  -- this is the proof space
  sorry

end students_in_both_clubs_l145_145936


namespace cost_per_mile_l145_145586

theorem cost_per_mile 
    (round_trip_distance : ℝ)
    (num_days : ℕ)
    (total_cost : ℝ) 
    (h1 : round_trip_distance = 200 * 2)
    (h2 : num_days = 7)
    (h3 : total_cost = 7000) 
  : (total_cost / (round_trip_distance * num_days) = 2.5) :=
by
  sorry

end cost_per_mile_l145_145586


namespace find_g_5_l145_145525

variable (g : ℝ → ℝ)

axiom func_eqn : ∀ x y : ℝ, x * g y = y * g x
axiom g_10 : g 10 = 15

theorem find_g_5 : g 5 = 7.5 :=
by
  sorry

end find_g_5_l145_145525


namespace percent_of_a_is_b_percent_of_d_is_c_percent_of_d_is_e_l145_145429

variables (a b c d e : ℝ)

-- Conditions
def condition1 : Prop := c = 0.25 * a
def condition2 : Prop := c = 0.50 * b
def condition3 : Prop := d = 0.40 * a
def condition4 : Prop := d = 0.20 * b
def condition5 : Prop := e = 0.35 * d
def condition6 : Prop := e = 0.15 * c

-- Proof Problem Statements
theorem percent_of_a_is_b (h1 : condition1 a c) (h2 : condition2 c b) : b = 0.5 * a := sorry

theorem percent_of_d_is_c (h1 : condition1 a c) (h3 : condition3 a d) : c = 0.625 * d := sorry

theorem percent_of_d_is_e (h5 : condition5 e d) : e = 0.35 * d := sorry

end percent_of_a_is_b_percent_of_d_is_c_percent_of_d_is_e_l145_145429


namespace new_average_score_after_drop_l145_145069

theorem new_average_score_after_drop
  (avg_score : ℝ) (num_students : ℕ) (drop_score : ℝ) (remaining_students : ℕ) :
  avg_score = 62.5 →
  num_students = 16 →
  drop_score = 70 →
  remaining_students = 15 →
  (num_students * avg_score - drop_score) / remaining_students = 62 :=
by
  intros h_avg h_num h_drop h_remain
  rw [h_avg, h_num, h_drop, h_remain]
  norm_num

end new_average_score_after_drop_l145_145069


namespace carrie_pants_l145_145238

theorem carrie_pants (P : ℕ) (shirts := 4) (pants := P) (jackets := 2)
  (shirt_cost := 8) (pant_cost := 18) (jacket_cost := 60)
  (total_cost := shirts * shirt_cost + jackets * jacket_cost + pants * pant_cost)
  (total_cost_half := 94) :
  total_cost = 188 → total_cost_half = 94 → total_cost = 2 * total_cost_half → P = 2 :=
by
  intros h_total h_half h_relation
  sorry

end carrie_pants_l145_145238


namespace geometric_sum_S6_l145_145567

variable {a : ℕ → ℝ} (S : ℕ → ℝ)

-- Conditions: S_n represents the sum of the first n terms of the geometric sequence {a_n}
-- and we have S_2 = 4 and S_4 = 6
theorem geometric_sum_S6 (S : ℕ → ℝ) (h1 : S 2 = 4) (h2 : S 4 = 6) : S 6 = 7 :=
sorry

end geometric_sum_S6_l145_145567


namespace deepak_walking_speed_l145_145519

noncomputable def speed_deepak (circumference: ℕ) (wife_speed_kmph: ℚ) (meet_time_min: ℚ) : ℚ :=
  let meet_time_hr := meet_time_min / 60
  let wife_speed_mpm := wife_speed_kmph * 1000 / 60
  let distance_wife := wife_speed_mpm * meet_time_min
  let distance_deepak := circumference - distance_wife
  let deepak_speed_mpm := distance_deepak / meet_time_min
  deepak_speed_mpm * 60 / 1000

theorem deepak_walking_speed
  (circumference: ℕ) 
  (wife_speed_kmph: ℚ)
  (meet_time_min: ℚ)
  (H1: circumference = 627)
  (H2: wife_speed_kmph = 3.75)
  (H3: meet_time_min = 4.56) :
  speed_deepak circumference wife_speed_kmph meet_time_min = 4.5 :=
by
  sorry

end deepak_walking_speed_l145_145519


namespace idempotent_elements_are_zero_l145_145767

-- Definitions based on conditions specified in the problem
variables {R : Type*} [Ring R] [CharZero R]
variable {e f g : R}

def idempotent (x : R) : Prop := x * x = x

-- The theorem to be proved
theorem idempotent_elements_are_zero (h_e : idempotent e) (h_f : idempotent f) (h_g : idempotent g) (h_sum : e + f + g = 0) : 
  e = 0 ∧ f = 0 ∧ g = 0 := 
sorry

end idempotent_elements_are_zero_l145_145767


namespace vicentes_total_cost_l145_145518

def total_cost (rice_bought cost_per_kg_rice meat_bought cost_per_lb_meat : Nat) : Nat :=
  (rice_bought * cost_per_kg_rice) + (meat_bought * cost_per_lb_meat)

theorem vicentes_total_cost :
  let rice_bought := 5
  let cost_per_kg_rice := 2
  let meat_bought := 3
  let cost_per_lb_meat := 5
  total_cost rice_bought cost_per_kg_rice meat_bought cost_per_lb_meat = 25 :=
by
  intros
  sorry

end vicentes_total_cost_l145_145518


namespace total_weight_of_snacks_l145_145450

-- Definitions for conditions
def weight_peanuts := 0.1
def weight_raisins := 0.4
def weight_almonds := 0.3

-- Theorem statement
theorem total_weight_of_snacks : weight_peanuts + weight_raisins + weight_almonds = 0.8 := by
  sorry

end total_weight_of_snacks_l145_145450


namespace inequality_solution_l145_145709

theorem inequality_solution (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  (x^2 + a * x > 4 * x + a - 3) ↔ (x < -1 ∨ x > 3)
:=
sorry

end inequality_solution_l145_145709


namespace range_of_constant_c_in_quadrant_I_l145_145745

theorem range_of_constant_c_in_quadrant_I (c : ℝ) (x y : ℝ)
  (h1 : x - 2 * y = 4)
  (h2 : 2 * c * x + y = 5)
  (hx_pos : x > 0)
  (hy_pos : y > 0) : 
  -1 / 4 < c ∧ c < 5 / 8 := 
sorry

end range_of_constant_c_in_quadrant_I_l145_145745


namespace sqrt_of_square_neg_l145_145507

variable {a : ℝ}

theorem sqrt_of_square_neg (h : a < 0) : Real.sqrt (a^2) = -a := 
sorry

end sqrt_of_square_neg_l145_145507


namespace range_of_b_l145_145626

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := Real.exp x * (x*x - b*x)

theorem range_of_b (b : ℝ) : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, 0 < (Real.exp x * ((x*x + (2 - b) * x - b)))) →
  b < 8/3 := 
sorry

end range_of_b_l145_145626


namespace union_of_A_and_B_l145_145504

-- Define the sets A and B
def A := {x : ℝ | 0 < x ∧ x < 16}
def B := {y : ℝ | -1 < y ∧ y < 4}

-- Prove that A ∪ B = (-1, 16)
theorem union_of_A_and_B : A ∪ B = {z : ℝ | -1 < z ∧ z < 16} :=
by sorry

end union_of_A_and_B_l145_145504


namespace xy_value_l145_145727

variable (a b x y : ℝ)
variable (h1 : 2 * a^x * b^3 = - a^2 * b^(1 - y))
variable (hx : x = 2)
variable (hy : y = -2)

theorem xy_value : x * y = -4 := 
by
  sorry

end xy_value_l145_145727


namespace second_group_num_persons_l145_145249

def man_hours (num_persons : ℕ) (days : ℕ) (hours_per_day : ℕ) : ℕ :=
  num_persons * days * hours_per_day

theorem second_group_num_persons :
  ∀ (x : ℕ),
    let first_group_man_hours := man_hours 36 12 5
    let second_group_days := 12
    let second_group_hours_per_day := 6
    (first_group_man_hours = man_hours x second_group_days second_group_hours_per_day) →
    x = 30 :=
by
  intros x first_group_man_hours second_group_days second_group_hours_per_day h
  sorry

end second_group_num_persons_l145_145249


namespace area_of_square_field_l145_145294

-- Define side length
def side_length : ℕ := 20

-- Theorem statement about the area of the square field
theorem area_of_square_field : (side_length * side_length) = 400 := by
  sorry

end area_of_square_field_l145_145294


namespace total_students_experimental_primary_school_l145_145409

theorem total_students_experimental_primary_school : 
  ∃ (n : ℕ), 
  n = (21 + 11) * 28 ∧ 
  n = 896 := 
by {
  -- Since the proof is not required, we use "sorry"
  sorry
}

end total_students_experimental_primary_school_l145_145409


namespace find_z2_l145_145961

theorem find_z2 (z1 z2 : ℂ) (h1 : z1 = 1 - I) (h2 : z1 * z2 = 1 + I) : z2 = I :=
sorry

end find_z2_l145_145961


namespace total_cost_alex_had_to_pay_l145_145317

def baseCost : ℝ := 30
def costPerText : ℝ := 0.04 -- 4 cents in dollars
def textsSent : ℕ := 150
def costPerMinuteOverLimit : ℝ := 0.15 -- 15 cents in dollars
def hoursUsed : ℝ := 26
def freeHours : ℝ := 25

def totalCost : ℝ :=
  baseCost + (costPerText * textsSent) + (costPerMinuteOverLimit * (hoursUsed - freeHours) * 60)

theorem total_cost_alex_had_to_pay :
  totalCost = 45 := by
  sorry

end total_cost_alex_had_to_pay_l145_145317


namespace M_supseteq_P_l145_145673

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 4}
def P : Set ℝ := {y | |y - 3| ≤ 1}

theorem M_supseteq_P : M ⊇ P := 
sorry

end M_supseteq_P_l145_145673


namespace coeff_x5_in_expansion_l145_145769

noncomputable def binomial_expansion_coeff (n k : ℕ) (x : ℝ) : ℝ :=
  Real.sqrt x ^ (n - k) * 2 ^ k * (Nat.choose n k)

theorem coeff_x5_in_expansion :
  (binomial_expansion_coeff 12 2 x) = 264 :=
by
  sorry

end coeff_x5_in_expansion_l145_145769


namespace algebraic_expression_value_l145_145024

theorem algebraic_expression_value (p q : ℝ)
  (h : p * 3^3 + q * 3 + 3 = 2005) :
  p * (-3)^3 + q * (-3) + 3 = -1999 :=
by
   sorry

end algebraic_expression_value_l145_145024


namespace smallest_leading_coefficient_l145_145084

theorem smallest_leading_coefficient :
  ∀ (P : ℤ → ℤ), (∃ (a b c : ℚ), ∀ (x : ℤ), P x = a * (x^2 : ℚ) + b * (x : ℚ) + c) →
  (∀ x : ℤ, ∃ k : ℤ, P x = k) →
  (∃ a : ℚ, (∀ x : ℤ, ∃ k : ℤ, a * (x^2 : ℚ) + b * (x : ℚ) + c = k) ∧ a > 0 ∧ (∀ a' : ℚ, (∀ x : ℤ, ∃ k : ℤ, a' * (x^2 : ℚ) + b * (x : ℚ) + c = k) → a' ≥ a) ∧ a = 1 / 2) := 
sorry

end smallest_leading_coefficient_l145_145084


namespace projected_increase_is_25_l145_145252

variable (R P : ℝ) -- variables for last year's revenue and projected increase in percentage

-- Conditions
axiom h1 : ∀ (R : ℝ), R > 0
axiom h2 : ∀ (P : ℝ), P/100 ≥ 0
axiom h3 : ∀ (R : ℝ), 0.75 * R = 0.60 * (R + (P/100) * R)

-- Goal
theorem projected_increase_is_25 (R : ℝ) : P = 25 :=
by {
    -- import the required axioms and provide the necessary proof
    apply sorry
}

end projected_increase_is_25_l145_145252


namespace BD_is_diameter_of_circle_l145_145473

variables {A B C D X Y : Type} [MetricSpace A] [MetricSpace B] 
  [MetricSpace C] [MetricSpace D] [MetricSpace X] [MetricSpace Y]

-- Assume these four points lie on a circle with certain ordering
variables (circ : Circle A B C D)

-- Given conditions
variables (h1 : circ.AB < circ.AD)
variables (h2 : circ.BC > circ.CD)

-- Points X and Y are where angle bisectors meet the circle again
variables (h3 : circ.bisects_angle_BAD_at X)
variables (h4 : circ.bisects_angle_BCD_at Y)

-- Hexagon sides with four equal lengths
variables (hex_equal : circ.hexagon_sides_equal_length A B X C D Y)

-- Prove that BD is a diameter
theorem BD_is_diameter_of_circle : circ.is_diameter BD := 
by
  sorry

end BD_is_diameter_of_circle_l145_145473


namespace abs_neg_is_2_l145_145904

theorem abs_neg_is_2 (a : ℝ) (h1 : a < 0) (h2 : |a| = 2) : a = -2 :=
by sorry

end abs_neg_is_2_l145_145904


namespace words_count_correct_l145_145127

def number_of_words (n : ℕ) : ℕ :=
if n % 2 = 0 then
  8 * 3^(n / 2 - 1)
else
  14 * 3^((n - 1) / 2)

theorem words_count_correct (n : ℕ) :
  number_of_words n = if n % 2 = 0 then 8 * 3^(n / 2 - 1) else 14 * 3^((n - 1) / 2) :=
by
  sorry

end words_count_correct_l145_145127


namespace convert_polar_to_rectangular_l145_145026

noncomputable def polarToRectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem convert_polar_to_rectangular :
  polarToRectangular 8 (7 * Real.pi / 6) = (-4 * Real.sqrt 3, -4) :=
by
  sorry

end convert_polar_to_rectangular_l145_145026


namespace cherry_sodas_in_cooler_l145_145835

theorem cherry_sodas_in_cooler (C : ℕ) (h1 : (C + 2 * C = 24)) : C = 8 :=
sorry

end cherry_sodas_in_cooler_l145_145835


namespace fair_total_revenue_l145_145142

noncomputable def price_per_ticket : ℝ := 8
noncomputable def total_ticket_revenue : ℝ := 8000
noncomputable def total_tickets_sold : ℝ := total_ticket_revenue / price_per_ticket

noncomputable def food_revenue : ℝ := (3/5) * total_tickets_sold * 10
noncomputable def rounded_ride_revenue : ℝ := (333 : ℝ) * 6
noncomputable def ride_revenue : ℝ := rounded_ride_revenue
noncomputable def rounded_souvenir_revenue : ℝ := (166 : ℝ) * 18
noncomputable def souvenir_revenue : ℝ := rounded_souvenir_revenue
noncomputable def game_revenue : ℝ := (1/10) * total_tickets_sold * 5

noncomputable def total_additional_revenue : ℝ := food_revenue + ride_revenue + souvenir_revenue + game_revenue
noncomputable def total_revenue : ℝ := total_ticket_revenue + total_additional_revenue

theorem fair_total_revenue : total_revenue = 19486 := by
  sorry

end fair_total_revenue_l145_145142


namespace solve_for_k_l145_145413

theorem solve_for_k (k : ℝ) (h₁ : ∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) (h₂ : k ≠ 0) : k = 6 :=
sorry

end solve_for_k_l145_145413


namespace raft_drift_time_l145_145217

theorem raft_drift_time (s : ℝ) (v_down v_up v_c : ℝ) 
  (h1 : v_down = s / 3) 
  (h2 : v_up = s / 4) 
  (h3 : v_down = v_c + v_c)
  (h4 : v_up = v_c - v_c) :
  v_c = s / 24 → (s / v_c) = 24 := 
by
  sorry

end raft_drift_time_l145_145217


namespace chromium_first_alloy_percentage_l145_145707

-- Defining the conditions
def percentage_chromium_first_alloy : ℝ := 10 
def percentage_chromium_second_alloy : ℝ := 6
def mass_first_alloy : ℝ := 15
def mass_second_alloy : ℝ := 35
def percentage_chromium_new_alloy : ℝ := 7.2

-- Proving the percentage of chromium in the first alloy is 10%
theorem chromium_first_alloy_percentage : percentage_chromium_first_alloy = 10 :=
by
  sorry

end chromium_first_alloy_percentage_l145_145707


namespace union_of_sets_l145_145809

-- Definition for set M
def M : Set ℝ := {x | x^2 - 4 * x + 3 < 0}

-- Definition for set N
def N : Set ℝ := {x | 2 * x + 1 < 5}

-- The theorem linking M and N
theorem union_of_sets : M ∪ N = {x | x < 3} :=
by
  -- Proof goes here
  sorry

end union_of_sets_l145_145809


namespace xiao_ming_correct_answers_l145_145618

theorem xiao_ming_correct_answers :
  let prob1 := (-2 - 2) = 0
  let prob2 := (-2 - (-2)) = -4
  let prob3 := (-3 + 5 - 6) = -4
  (if prob1 then 1 else 0) + (if prob2 then 1 else 0) + (if prob3 then 1 else 0) = 1 :=
by
  sorry

end xiao_ming_correct_answers_l145_145618


namespace handshaking_remainder_l145_145917

-- Define number of people
def num_people := 11

-- Define N as the number of possible handshaking ways
def N : ℕ :=
sorry -- This will involve complicated combinatorial calculations

-- Define the target result to be proven
theorem handshaking_remainder : N % 1000 = 120 :=
sorry

end handshaking_remainder_l145_145917


namespace combined_cost_price_is_250_l145_145118

axiom store_selling_conditions :
  ∃ (CP_A CP_B CP_C : ℝ),
    (CP_A = (110 + 70) / 2) ∧
    (CP_B = (90 + 30) / 2) ∧
    (CP_C = (150 + 50) / 2) ∧
    (CP_A + CP_B + CP_C = 250)

theorem combined_cost_price_is_250 : ∃ (CP_A CP_B CP_C : ℝ), CP_A + CP_B + CP_C = 250 :=
by sorry

end combined_cost_price_is_250_l145_145118


namespace width_of_rectangle_l145_145392

-- Define the problem constants and parameters
variable (L W : ℝ)

-- State the main theorem about the width
theorem width_of_rectangle (h₁ : L * W = 50) (h₂ : L + W = 15) : W = 5 :=
sorry

end width_of_rectangle_l145_145392


namespace james_tylenol_daily_intake_l145_145550

def tylenol_per_tablet : ℕ := 375
def tablets_per_dose : ℕ := 2
def hours_per_dose : ℕ := 6
def hours_per_day : ℕ := 24

theorem james_tylenol_daily_intake :
  (hours_per_day / hours_per_dose) * (tablets_per_dose * tylenol_per_tablet) = 3000 := by
  sorry

end james_tylenol_daily_intake_l145_145550


namespace solve_double_inequality_l145_145487

theorem solve_double_inequality (x : ℝ) :
  (-1 < (x^2 - 20 * x + 21) / (x^2 - 4 * x + 5) ∧
   (x^2 - 20 * x + 21) / (x^2 - 4 * x + 5) < 1) ↔ (2 < x ∨ 26 < x) := 
sorry

end solve_double_inequality_l145_145487


namespace fraction_simplification_l145_145795

theorem fraction_simplification : 1 + 1 / (1 - 1 / (2 + 1 / 3)) = 11 / 4 :=
by
  sorry

end fraction_simplification_l145_145795


namespace find_k_l145_145399

theorem find_k (k : ℝ) : (∃ b : ℝ, (x : ℝ) → x^2 - 60 * x + k = (x + b)^2) → k = 900 :=
by 
  sorry

end find_k_l145_145399


namespace lcm_150_456_l145_145667

theorem lcm_150_456 : Nat.lcm 150 456 = 11400 := by
  sorry

end lcm_150_456_l145_145667


namespace evaluate_expression_l145_145896

theorem evaluate_expression :
  (4 * 6) / (12 * 14) * ((8 * 12 * 14) / (4 * 6 * 8)) = 1 := 
by 
  sorry

end evaluate_expression_l145_145896


namespace problem_statement_l145_145362

theorem problem_statement (a b : ℝ) (h : 3 * a - 2 * b = -1) : 3 * a - 2 * b + 2024 = 2023 :=
by
  sorry

end problem_statement_l145_145362


namespace central_angle_of_sector_l145_145232

theorem central_angle_of_sector (r S : ℝ) (h_r : r = 2) (h_S : S = 4) : 
  ∃ α : ℝ, α = 2 ∧ S = (1/2) * α * r^2 := 
by 
  sorry

end central_angle_of_sector_l145_145232


namespace sqrt_of_16_l145_145274

theorem sqrt_of_16 : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_of_16_l145_145274


namespace find_equation_of_perpendicular_line_l145_145916

noncomputable def line_through_point_perpendicular
    (A : ℝ × ℝ) (a b c : ℝ) (hA : A = (2, 3)) (hLine : a = 2 ∧ b = 1 ∧ c = -5) :
    Prop :=
  ∃ (m : ℝ) (b1 : ℝ), (m = (1 / 2)) ∧
    (b1 = 3 - m * 2) ∧
    (∀ (x y : ℝ), y = m * (x - 2) + 3 → a * x + b * y + c = 0 → x - 2 * y + 4 = 0)

theorem find_equation_of_perpendicular_line :
  line_through_point_perpendicular (2, 3) 2 1 (-5) rfl ⟨rfl, rfl, rfl⟩ :=
sorry

end find_equation_of_perpendicular_line_l145_145916


namespace function_symmetry_extremum_l145_145347

noncomputable def f (x θ : ℝ) : ℝ := 3 * Real.cos (Real.pi * x + θ)

theorem function_symmetry_extremum {θ : ℝ} (H : ∀ x : ℝ, f x θ = f (2 - x) θ) : 
  f 1 θ = 3 ∨ f 1 θ = -3 :=
by
  sorry

end function_symmetry_extremum_l145_145347


namespace x_squared_minus_y_squared_l145_145975

theorem x_squared_minus_y_squared {x y : ℚ} 
    (h1 : x + y = 3/8) 
    (h2 : x - y = 5/24) 
    : x^2 - y^2 = 5/64 := 
by 
    -- The proof would go here
    sorry

end x_squared_minus_y_squared_l145_145975


namespace impossible_seed_germinate_without_water_l145_145175

-- Definitions for the conditions
def heats_up_when_conducting (conducts : Bool) : Prop := conducts
def determines_plane (non_collinear : Bool) : Prop := non_collinear
def germinates_without_water (germinates : Bool) : Prop := germinates
def wins_lottery_consecutively (wins_twice : Bool) : Prop := wins_twice

-- The fact that a seed germinates without water is impossible
theorem impossible_seed_germinate_without_water 
  (conducts : Bool) 
  (non_collinear : Bool) 
  (germinates : Bool) 
  (wins_twice : Bool) 
  (h1 : heats_up_when_conducting conducts) 
  (h2 : determines_plane non_collinear) 
  (h3 : ¬germinates_without_water germinates) 
  (h4 : wins_lottery_consecutively wins_twice) :
  ¬germinates_without_water true :=
sorry

end impossible_seed_germinate_without_water_l145_145175


namespace wand_cost_l145_145440

-- Conditions based on the problem
def initialWands := 3
def salePrice (x : ℝ) := x + 5
def totalCollected := 130
def soldWands := 2

-- Proof statement
theorem wand_cost (x : ℝ) : 
  2 * salePrice x = totalCollected → x = 60 := 
by 
  sorry

end wand_cost_l145_145440


namespace set_D_forms_triangle_l145_145113

theorem set_D_forms_triangle (a b c : ℕ) (h1 : a = 4) (h2 : b = 5) (h3 : c = 6) : a + b > c ∧ a + c > b ∧ b + c > a := by
  rw [h1, h2, h3]
  show 4 + 5 > 6 ∧ 4 + 6 > 5 ∧ 5 + 6 > 4
  sorry

end set_D_forms_triangle_l145_145113


namespace rice_in_each_container_l145_145181

theorem rice_in_each_container 
  (total_weight : ℚ) 
  (num_containers : ℕ)
  (conversion_factor : ℚ) 
  (equal_division : total_weight = 29 / 4 ∧ num_containers = 4 ∧ conversion_factor = 16) : 
  (total_weight / num_containers) * conversion_factor = 29 := 
by 
  sorry

end rice_in_each_container_l145_145181


namespace correct_properties_l145_145868

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem correct_properties :
  (∀ x, f x = 4 * Real.cos (2 * x - Real.pi / 6)) ∧
  (f (-Real.pi / 6) = 0) :=
by
  sorry

end correct_properties_l145_145868


namespace smallest_positive_integer_l145_145494

theorem smallest_positive_integer (k : ℕ) :
  (∃ k : ℕ, ((2^4 ∣ 1452 * k) ∧ (3^3 ∣ 1452 * k) ∧ (13^3 ∣ 1452 * k))) → 
  k = 676 := 
sorry

end smallest_positive_integer_l145_145494


namespace toys_ratio_l145_145781

-- Definitions of given conditions
variables (rabbits : ℕ) (toys_monday toys_wednesday toys_friday toys_saturday total_toys : ℕ)
variables (h_rabbits : rabbits = 16)
variables (h_toys_monday : toys_monday = 6)
variables (h_toys_friday : toys_friday = 4 * toys_monday)
variables (h_toys_saturday : toys_saturday = toys_wednesday / 2)
variables (h_total_toys : total_toys = rabbits * 3)

-- Define the Lean theorem to state the problem conditions and prove the ratio
theorem toys_ratio (h : toys_monday + toys_wednesday + toys_friday + toys_saturday = total_toys) :
  (if (2 * toys_wednesday = 12) then 2 else 1) = 2 :=
by 
  sorry

end toys_ratio_l145_145781


namespace multiply_powers_same_base_l145_145261

theorem multiply_powers_same_base (a : ℝ) : a^3 * a = a^4 :=
by
  sorry

end multiply_powers_same_base_l145_145261


namespace angle_between_strips_l145_145584

theorem angle_between_strips (w : ℝ) (a : ℝ) (angle : ℝ) (h_w : w = 1) (h_area : a = 2) :
  ∃ θ : ℝ, θ = 30 ∧ angle = θ :=
by
  sorry

end angle_between_strips_l145_145584


namespace quadratic_union_nonempty_l145_145247

theorem quadratic_union_nonempty (a : ℝ) :
  (∃ x : ℝ, x^2 - (a-2)*x - 2*a + 4 = 0) ∨ (∃ y : ℝ, y^2 + (2*a-3)*y + 2*a^2 - a - 3 = 0) ↔
    a ≤ -6 ∨ (-7/2) ≤ a ∧ a ≤ (3/2) ∨ a ≥ 2 :=
sorry

end quadratic_union_nonempty_l145_145247


namespace no_high_quality_triangle_exist_high_quality_quadrilateral_l145_145912

-- Define the necessary predicate for a number being a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the property of being a high-quality triangle
def high_quality_triangle (a b c : ℕ) : Prop :=
  is_perfect_square (a + b) ∧ is_perfect_square (b + c) ∧ is_perfect_square (c + a)

-- Define the property of non-existence of a high-quality triangle
theorem no_high_quality_triangle (a b c : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c) : 
  ¬high_quality_triangle a b c := by sorry

-- Define the property of being a high-quality quadrilateral
def high_quality_quadrilateral (a b c d : ℕ) : Prop :=
  is_perfect_square (a + b) ∧ is_perfect_square (b + c) ∧ is_perfect_square (c + d) ∧ is_perfect_square (d + a)

-- Define the property of existence of a high-quality quadrilateral
theorem exist_high_quality_quadrilateral (a b c d : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c) (hd : Prime d) : 
  high_quality_quadrilateral a b c d := by sorry

end no_high_quality_triangle_exist_high_quality_quadrilateral_l145_145912


namespace unreasonable_inference_l145_145574

theorem unreasonable_inference:
  (∀ (S T : Type) (P : S → Prop) (Q : T → Prop), (∀ x y, P x → ¬ Q y) → ¬ (∀ x, P x) → (∃ y, ¬ Q y))
  ∧ ¬ (∀ s : ℝ, (s = 100) → ∀ t : ℝ, t = 100) :=
sorry

end unreasonable_inference_l145_145574


namespace examination_students_total_l145_145193

/-
  Problem Statement:
  Given:
  - 35% of the students passed the examination.
  - 546 students failed the examination.

  Prove:
  - The total number of students who appeared for the examination is 840.
-/

theorem examination_students_total (T : ℝ) (h1 : 0.35 * T + 0.65 * T = T) (h2 : 0.65 * T = 546) : T = 840 :=
by
  -- skipped proof part
  sorry

end examination_students_total_l145_145193


namespace cube_root_neg_frac_l145_145126

theorem cube_root_neg_frac : (-(1/3 : ℝ))^3 = - 1 / 27 := by
  sorry

end cube_root_neg_frac_l145_145126


namespace range_of_k_l145_145811

noncomputable def triangle_range (A B C : ℝ) (a b c k : ℝ) : Prop :=
  A + B + C = Real.pi ∧
  (B = Real.pi / 3) ∧       -- From arithmetic sequence and solving for B
  a^2 + c^2 = k * b^2 ∧
  (1 < k ∧ k <= 2)

theorem range_of_k (A B C a b c k : ℝ) :
  A + B + C = Real.pi →
  (B = Real.pi - (A + C)) →
  (B = Real.pi / 3) →
  a^2 + c^2 = k * b^2 →
  0 < A ∧ A < 2*Real.pi/3 →
  1 < k ∧ k <= 2 :=
by
  sorry

end range_of_k_l145_145811


namespace trapezoid_problem_case1_trapezoid_problem_case2_trapezoid_problem_case3_l145_145467

variable {a b : ℝ}
variable {M N : ℝ}

/-- Trapezoid problem statements -/
theorem trapezoid_problem_case1 (h : a < 2 * b) : M - N = a - 2 * b := 
sorry

theorem trapezoid_problem_case2 (h : a = 2 * b) : M - N = 0 := 
sorry

theorem trapezoid_problem_case3 (h : a > 2 * b) : M - N = 2 * b - a := 
sorry

end trapezoid_problem_case1_trapezoid_problem_case2_trapezoid_problem_case3_l145_145467


namespace probability_of_at_least_one_die_shows_2_is_correct_l145_145806

-- Definitions for the conditions
def total_outcomes : ℕ := 64
def neither_die_shows_2_outcomes : ℕ := 49
def favorability (total : ℕ) (exclusion : ℕ) : ℕ := total - exclusion
def favorable_outcomes : ℕ := favorability total_outcomes neither_die_shows_2_outcomes
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- Mathematically equivalent proof problem statement
theorem probability_of_at_least_one_die_shows_2_is_correct : 
  probability favorable_outcomes total_outcomes = 15 / 64 :=
sorry

end probability_of_at_least_one_die_shows_2_is_correct_l145_145806


namespace proposal_spreading_problem_l145_145112

theorem proposal_spreading_problem (n : ℕ) : 1 + n + n^2 = 1641 := 
sorry

end proposal_spreading_problem_l145_145112


namespace pizza_slices_with_all_three_toppings_l145_145582

theorem pizza_slices_with_all_three_toppings : 
  ∀ (a b c d e f g : ℕ), 
  a + b + c + d + e + f + g = 24 ∧ 
  a + d + e + g = 12 ∧ 
  b + d + f + g = 15 ∧ 
  c + e + f + g = 10 → 
  g = 5 := 
by {
  sorry
}

end pizza_slices_with_all_three_toppings_l145_145582


namespace sum_divided_among_xyz_l145_145934

noncomputable def total_amount (x_share y_share z_share : ℝ) : ℝ :=
  x_share + y_share + z_share

theorem sum_divided_among_xyz
    (x_share : ℝ) (y_share : ℝ) (z_share : ℝ)
    (y_gets_45_paisa : y_share = 0.45 * x_share)
    (z_gets_50_paisa : z_share = 0.50 * x_share)
    (y_share_is_18 : y_share = 18) :
    total_amount x_share y_share z_share = 78 := by
  sorry

end sum_divided_among_xyz_l145_145934


namespace maximize_product_numbers_l145_145928

theorem maximize_product_numbers (a b : ℕ) (ha : a = 96420) (hb : b = 87531) (cond: a * b = 96420 * 87531):
  b = 87531 := 
by sorry

end maximize_product_numbers_l145_145928


namespace population_approx_10000_2090_l145_145044

def population (initial_population : ℕ) (years : ℕ) : ℕ :=
  initial_population * 2 ^ (years / 20)

theorem population_approx_10000_2090 :
  ∃ y, y = 2090 ∧ population 500 (2090 - 2010) = 500 * 2 ^ (80 / 20) :=
by
  sorry

end population_approx_10000_2090_l145_145044


namespace quadratic_roots_expression_l145_145866

theorem quadratic_roots_expression (x1 x2 : ℝ) (h1 : x1^2 + x1 - 2023 = 0) (h2 : x2^2 + x2 - 2023 = 0) :
  x1^2 + 2*x1 + x2 = 2022 :=
by
  sorry

end quadratic_roots_expression_l145_145866


namespace min_value_expr_l145_145748

theorem min_value_expr (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
  ∃ x : ℝ, x = 6 ∧ x = (2 * a + b) / c + (2 * a + c) / b + (2 * b + c) / a :=
by
  sorry

end min_value_expr_l145_145748


namespace opposite_of_neg_two_l145_145737

theorem opposite_of_neg_two : ∃ x : Int, (-2 + x = 0) ∧ x = 2 :=
by
  use 2
  constructor
  . simp
  . rfl

end opposite_of_neg_two_l145_145737


namespace sum_of_dice_less_than_10_probability_l145_145389

/-
  Given:
  - A fair die with faces labeled 1, 2, 3, 4, 5, 6.
  - The die is rolled twice.

  Prove that the probability that the sum of the face values is less than 10 is 5/6.
-/

noncomputable def probability_sum_less_than_10 : ℚ :=
  let total_outcomes := 36
  let favorable_outcomes := 30
  favorable_outcomes / total_outcomes

theorem sum_of_dice_less_than_10_probability :
  probability_sum_less_than_10 = 5 / 6 :=
by
  sorry

end sum_of_dice_less_than_10_probability_l145_145389


namespace train_cross_time_in_seconds_l145_145272

-- Definitions based on conditions
def train_speed_kph : ℚ := 60
def train_length_m : ℚ := 450

-- Statement: prove that the time to cross the pole is 27 seconds
theorem train_cross_time_in_seconds (train_speed_kph train_length_m : ℚ) :
  train_speed_kph = 60 →
  train_length_m = 450 →
  (train_length_m / (train_speed_kph * 1000 / 3600)) = 27 :=
by
  intros h_speed h_length
  rw [h_speed, h_length]
  sorry

end train_cross_time_in_seconds_l145_145272


namespace no_intersection_of_asymptotes_l145_145056

noncomputable def given_function (x : ℝ) : ℝ :=
  (x^2 - 9 * x + 20) / (x^2 - 9 * x + 18)

theorem no_intersection_of_asymptotes : 
  (∀ x, x = 3 → ¬ ∃ y, y = given_function x) ∧ 
  (∀ x, x = 6 → ¬ ∃ y, y = given_function x) ∧ 
  ¬ ∃ x, (x = 3 ∨ x = 6) ∧ given_function x = 1 := 
by
  sorry

end no_intersection_of_asymptotes_l145_145056


namespace eccentricity_range_of_ellipse_l145_145632

theorem eccentricity_range_of_ellipse
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (e : ℝ) (he1 : e > 0) (he2 : e < 1)
  (h_directrix : 2 * (a / e) ≤ 3 * (2 * a)) :
  (1 / 3) ≤ e ∧ e < 1 := 
sorry

end eccentricity_range_of_ellipse_l145_145632


namespace theta_in_third_quadrant_l145_145762

theorem theta_in_third_quadrant (θ : ℝ) (h1 : Real.tan θ > 0) (h2 : Real.sin θ < 0) : 
  ∃ q : ℕ, q = 3 := 
sorry

end theta_in_third_quadrant_l145_145762


namespace eleven_y_minus_x_l145_145600

theorem eleven_y_minus_x (x y : ℤ) (h1 : x = 7 * y + 3) (h2 : 2 * x = 18 * y + 2) : 11 * y - x = 1 := by
  sorry

end eleven_y_minus_x_l145_145600


namespace sqrt_7_estimate_l145_145557

theorem sqrt_7_estimate (h1 : 4 < 7) (h2 : 7 < 9) (h3 : Nat.sqrt 4 = 2) (h4 : Nat.sqrt 9 = 3) : 2 < Real.sqrt 7 ∧ Real.sqrt 7 < 3 :=
  by {
    -- the proof would go here, but use 'sorry' to omit it
    sorry
  }

end sqrt_7_estimate_l145_145557


namespace shaded_area_represents_correct_set_l145_145930

theorem shaded_area_represents_correct_set :
  ∀ (U A B : Set ℕ), 
    U = {0, 1, 2, 3, 4} → 
    A = {1, 2, 3} → 
    B = {2, 4} → 
    (U \ (A ∪ B)) ∪ (A ∩ B) = {0, 2} :=
by
  intros U A B hU hA hB
  -- The rest of the proof would go here
  sorry

end shaded_area_represents_correct_set_l145_145930


namespace daily_earnings_from_oil_refining_l145_145264

-- Definitions based on conditions
def daily_earnings_from_mining : ℝ := 3000000
def monthly_expenses : ℝ := 30000000
def fine : ℝ := 25600000
def profit_percentage : ℝ := 0.01
def months_in_year : ℝ := 12
def days_in_month : ℝ := 30

-- The question translated as a Lean theorem statement
theorem daily_earnings_from_oil_refining : ∃ O : ℝ, O = 5111111.11 ∧ 
  fine = profit_percentage * months_in_year * 
    (days_in_month * (daily_earnings_from_mining + O) - monthly_expenses) :=
sorry

end daily_earnings_from_oil_refining_l145_145264


namespace students_on_couch_per_room_l145_145505

def total_students : ℕ := 30
def total_rooms : ℕ := 6
def students_per_bed : ℕ := 2
def beds_per_room : ℕ := 2
def students_in_beds_per_room : ℕ := beds_per_room * students_per_bed

theorem students_on_couch_per_room :
  (total_students / total_rooms) - students_in_beds_per_room = 1 := by
  sorry

end students_on_couch_per_room_l145_145505


namespace range_for_k_solutions_when_k_eq_1_l145_145851

noncomputable section

-- Part (1): Range for k
theorem range_for_k (k : ℝ) :
  (∀ x : ℝ, k * x^2 - (2 * k + 4) * x + k - 6 = 0 → (∃ x1 x2 : ℝ, x1 ≠ x2)) ↔ (k > -2/5 ∧ k ≠ 0) :=
sorry

-- Part (2): Completing the square for k = 1
theorem solutions_when_k_eq_1 :
  (∀ x : ℝ, x^2 - 6 * x - 5 = 0 → (x = 3 + Real.sqrt 14 ∨ x = 3 - Real.sqrt 14)) :=
sorry

end range_for_k_solutions_when_k_eq_1_l145_145851


namespace mean_score_is_74_l145_145763

theorem mean_score_is_74 (σ q : ℝ)
  (h1 : 58 = q - 2 * σ)
  (h2 : 98 = q + 3 * σ) :
  q = 74 :=
by
  sorry

end mean_score_is_74_l145_145763


namespace simplify_frac_and_find_cd_l145_145220

theorem simplify_frac_and_find_cd :
  ∀ (m : ℤ), ∃ (c d : ℤ), 
    (c * m + d = (6 * m + 12) / 3) ∧ (c = 2) ∧ (d = 4) ∧ (c / d = 1 / 2) :=
by
  sorry

end simplify_frac_and_find_cd_l145_145220


namespace locus_of_p_ratio_distances_l145_145743

theorem locus_of_p_ratio_distances :
  (∀ (P : ℝ × ℝ), (dist P (1, 0) = (1 / 3) * abs (P.1 - 9)) →
  (P.1^2 / 9 + P.2^2 / 8 = 1)) :=
by
  sorry

end locus_of_p_ratio_distances_l145_145743


namespace total_ticket_cost_l145_145522

theorem total_ticket_cost :
  ∀ (A : ℝ), 
  -- Conditions
  (6 : ℝ) * (5 : ℝ) + (2 : ℝ) * A = 50 :=
by
  sorry

end total_ticket_cost_l145_145522


namespace common_ratio_l145_145134

variable {G : Type} [LinearOrderedField G]

-- Definitions based on conditions
def geometric_seq (a₁ q : G) (n : ℕ) : G := a₁ * q^(n-1)
def sum_geometric_seq (a₁ q : G) (n : ℕ) : G :=
  if q = 1 then a₁ * n else a₁ * (1 - q^n) / (1 - q)

-- Hypotheses from conditions
variable {a₁ q : G}
variable (h1 : sum_geometric_seq a₁ q 3 = 7)
variable (h2 : sum_geometric_seq a₁ q 6 = 63)

theorem common_ratio (a₁ q : G) (h1 : sum_geometric_seq a₁ q 3 = 7)
  (h2 : sum_geometric_seq a₁ q 6 = 63) : q = 2 :=
by
  -- Proof to be completed
  sorry

end common_ratio_l145_145134


namespace minimum_stool_height_l145_145937

def ceiling_height : ℤ := 280
def alice_height : ℤ := 150
def reach : ℤ := alice_height + 30
def light_bulb_height : ℤ := ceiling_height - 15

theorem minimum_stool_height : 
  ∃ h : ℤ, reach + h = light_bulb_height ∧ h = 85 :=
by
  sorry

end minimum_stool_height_l145_145937


namespace unique_solution_l145_145980

def s (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem unique_solution (m n : ℕ) (h : n * (n + 1) = 3 ^ m + s n + 1182) : (m, n) = (0, 34) :=
by
  sorry

end unique_solution_l145_145980


namespace train_speed_l145_145461

-- Define the conditions in terms of distance and time
def train_length : ℕ := 160
def crossing_time : ℕ := 8

-- Define the expected speed
def expected_speed : ℕ := 20

-- The theorem stating the speed of the train given the conditions
theorem train_speed : (train_length / crossing_time) = expected_speed :=
by
  -- Note: The proof is omitted
  sorry

end train_speed_l145_145461


namespace greatest_drop_is_third_quarter_l145_145624

def priceStart (quarter : ℕ) : ℕ :=
  match quarter with
  | 1 => 10
  | 2 => 7
  | 3 => 9
  | 4 => 5
  | _ => 0 -- default case for invalid quarters

def priceEnd (quarter : ℕ) : ℕ :=
  match quarter with
  | 1 => 7
  | 2 => 9
  | 3 => 5
  | 4 => 6
  | _ => 0 -- default case for invalid quarters

def priceChange (quarter : ℕ) : ℤ :=
  priceStart quarter - priceEnd quarter

def greatestDropInQuarter : ℕ :=
  if priceChange 1 > priceChange 3 then 1
  else if priceChange 2 > priceChange 1 then 2
  else if priceChange 3 > priceChange 4 then 3
  else 4

theorem greatest_drop_is_third_quarter :
  greatestDropInQuarter = 3 :=
by
  -- proof goes here
  sorry

end greatest_drop_is_third_quarter_l145_145624


namespace rojas_speed_l145_145343

theorem rojas_speed (P R : ℝ) (h1 : P = 3) (h2 : 4 * (R + P) = 28) : R = 4 :=
by
  sorry

end rojas_speed_l145_145343


namespace digits_count_of_special_numbers_l145_145514

theorem digits_count_of_special_numbers
  (n : ℕ)
  (h1 : 8^n = 28672) : n = 5 := 
by
  sorry

end digits_count_of_special_numbers_l145_145514


namespace icosahedron_path_count_l145_145406

-- Definitions from the conditions
def vertices := 12
def edges := 30
def top_adjacent := 5
def bottom_adjacent := 5

-- Define the total paths calculation based on the given structural conditions
theorem icosahedron_path_count (v e ta ba : ℕ) (hv : v = 12) (he : e = 30) (hta : ta = 5) (hba : ba = 5) : 
  (ta * (ta - 1) * (ba - 1)) * 2 = 810 :=
by
-- Insert calculation logic here if needed or detailed structure definitions
  sorry

end icosahedron_path_count_l145_145406


namespace find_base_l145_145824

noncomputable def base_satisfies_first_transaction (s : ℕ) : Prop :=
  5 * s^2 + 3 * s + 460 = s^3 + s^2 + 1

noncomputable def base_satisfies_second_transaction (s : ℕ) : Prop :=
  s^2 + 2 * s + 2 * s^2 + 6 * s = 5 * s^2

theorem find_base (s : ℕ) (h1 : base_satisfies_first_transaction s) (h2 : base_satisfies_second_transaction s) :
  s = 4 :=
sorry

end find_base_l145_145824


namespace find_largest_angle_l145_145158

noncomputable def largest_angle_in_convex_pentagon (x : ℝ) : Prop :=
  let angle1 := 2 * x + 2
  let angle2 := 3 * x - 3
  let angle3 := 4 * x + 4
  let angle4 := 6 * x - 6
  let angle5 := x + 5
  angle1 + angle2 + angle3 + angle4 + angle5 = 540 ∧
  max (max angle1 (max angle2 (max angle3 angle4))) angle5 = angle4 ∧
  angle4 = 195.75

theorem find_largest_angle (x : ℝ) : largest_angle_in_convex_pentagon x := by
  sorry

end find_largest_angle_l145_145158


namespace fixed_point_of_transformed_logarithmic_function_l145_145355

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

noncomputable def f_a (a : ℝ) (x : ℝ) : ℝ := 1 + log_a a (x - 1)

theorem fixed_point_of_transformed_logarithmic_function
  (a : ℝ) (ha : 0 < a ∧ a ≠ 1) : f_a a 2 = 1 :=
by
  -- Prove the theorem using given conditions
  sorry

end fixed_point_of_transformed_logarithmic_function_l145_145355


namespace first_term_of_geometric_series_l145_145723

/-- An infinite geometric series with common ratio -1/3 has a sum of 24.
    Prove that the first term of the series is 32. -/
theorem first_term_of_geometric_series (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 24) 
  (h3 : S = a / (1 - r)) : 
  a = 32 := 
sorry

end first_term_of_geometric_series_l145_145723


namespace luca_lost_more_weight_l145_145336

theorem luca_lost_more_weight (barbi_kg_month : ℝ) (luca_kg_year : ℝ) (months_in_year : ℕ) (years : ℕ) 
(h_barbi : barbi_kg_month = 1.5) (h_luca : luca_kg_year = 9) (h_months_in_year : months_in_year = 12) (h_years : years = 11) : 
  (luca_kg_year * years) - (barbi_kg_month * months_in_year * (years / 11)) = 81 := 
by 
  sorry

end luca_lost_more_weight_l145_145336


namespace tangent_line_eq_at_x_is_1_range_of_sum_extreme_values_l145_145944

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * Real.log x - x
noncomputable def g (x m : ℝ) : ℝ := f x + m * x^2
noncomputable def tangentLineEq (x y : ℝ) : Prop := x + 2 * y + 1 = 0
noncomputable def rangeCondition (x₁ x₂ m : ℝ) : Prop := g x₁ m + g x₂ m < -3 / 2

theorem tangent_line_eq_at_x_is_1 :
  tangentLineEq 1 (f 1) := 
sorry

theorem range_of_sum_extreme_values (h : 0 < m ∧ m < 1 / 4) (x₁ x₂ : ℝ) :
  rangeCondition x₁ x₂ m := 
sorry

end tangent_line_eq_at_x_is_1_range_of_sum_extreme_values_l145_145944


namespace cube_difference_l145_145628

variables (a b : ℝ)  -- Specify the variables a and b are real numbers

theorem cube_difference (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a^3 - b^3 = 108 :=
by
  -- Skip the proof as requested.
  sorry

end cube_difference_l145_145628


namespace fraction_of_rotten_is_one_third_l145_145716

def total_berries (blueberries cranberries raspberries : Nat) : Nat :=
  blueberries + cranberries + raspberries

def fresh_berries (berries_to_sell berries_to_keep : Nat) : Nat :=
  berries_to_sell + berries_to_keep

def rotten_berries (total fresh : Nat) : Nat :=
  total - fresh

def fraction_rot (rotten total : Nat) : Rat :=
  (rotten : Rat) / (total : Rat)

theorem fraction_of_rotten_is_one_third :
  ∀ (blueberries cranberries raspberries berries_to_sell : Nat),
    blueberries = 30 →
    cranberries = 20 →
    raspberries = 10 →
    berries_to_sell = 20 →
    fraction_rot (rotten_berries (total_berries blueberries cranberries raspberries) 
                  (fresh_berries berries_to_sell berries_to_sell))
                  (total_berries blueberries cranberries raspberries) = 1 / 3 :=
by
  intros blueberries cranberries raspberries berries_to_sell
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end fraction_of_rotten_is_one_third_l145_145716


namespace solutions_diff_l145_145068

theorem solutions_diff (a b : ℝ) (h1: (a-5)*(a+5) = 26*a - 130) (h2: (b-5)*(b+5) = 26*b - 130) (h3 : a ≠ b) (h4: a > b) : a - b = 16 := 
by
  sorry 

end solutions_diff_l145_145068


namespace sequence_polynomial_degree_l145_145607

theorem sequence_polynomial_degree
  (k : ℕ)
  (v : ℕ → ℤ)
  (u : ℕ → ℤ)
  (h_diff_poly : ∃ p : Polynomial ℤ, ∀ n, v n = Polynomial.eval (n : ℤ) p)
  (h_diff_seq : ∀ n, v n = (u (n + 1) - u n)) :
  ∃ q : Polynomial ℤ, ∀ n, u n = Polynomial.eval (n : ℤ) q := 
sorry

end sequence_polynomial_degree_l145_145607


namespace typist_salary_proof_l145_145823

noncomputable def original_salary (x : ℝ) : Prop :=
  1.10 * x * 0.95 = 1045

theorem typist_salary_proof (x : ℝ) (H : original_salary x) : x = 1000 :=
sorry

end typist_salary_proof_l145_145823


namespace stephen_total_distance_l145_145816

noncomputable def total_distance : ℝ :=
let speed1 : ℝ := 16
let time1 : ℝ := 10 / 60
let distance1 : ℝ := speed1 * time1

let speed2 : ℝ := 12 - 2 -- headwind reduction
let time2 : ℝ := 20 / 60
let distance2 : ℝ := speed2 * time2

let speed3 : ℝ := 20 + 4 -- tailwind increase
let time3 : ℝ := 15 / 60
let distance3 : ℝ := speed3 * time3

distance1 + distance2 + distance3

theorem stephen_total_distance :
  total_distance = 12 :=
by sorry

end stephen_total_distance_l145_145816


namespace range_of_a_l145_145081

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ 1 → x * |x - a| - 2 < 0) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l145_145081


namespace exponent_multiplication_correct_l145_145923

theorem exponent_multiplication_correct (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end exponent_multiplication_correct_l145_145923


namespace amoeba_growth_one_week_l145_145163

theorem amoeba_growth_one_week :
  (3 ^ 7 = 2187) :=
by
  sorry

end amoeba_growth_one_week_l145_145163


namespace sufficient_but_not_necessary_l145_145680

theorem sufficient_but_not_necessary (x : ℝ) : (x^2 = 9 → x = 3) ∧ (¬(x^2 = 9 → x = 3 ∨ x = -3)) :=
by
  sorry

end sufficient_but_not_necessary_l145_145680


namespace mod_exponent_problem_l145_145818

theorem mod_exponent_problem : (11 ^ 2023) % 100 = 31 := by
  sorry

end mod_exponent_problem_l145_145818


namespace arith_to_geom_l145_145932

noncomputable def a (n : ℕ) (d : ℝ) : ℝ := 1 + (n - 1) * d

theorem arith_to_geom (m n : ℕ) (d : ℝ) 
  (h_pos : d > 0)
  (h_arith_seq : ∀ k : ℕ, a k d > 0)
  (h_geo_seq : (a 4 d + 5 / 2)^2 = (a 3 d) * (a 11 d))
  (h_mn : m - n = 8) : 
  a m d - a n d = 12 := 
sorry

end arith_to_geom_l145_145932


namespace simplify_expression_l145_145821

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

theorem simplify_expression (ha : a > 0) (hb : b > 0) (hab : a^3 + b^3 = a + b) :
  (a / b) + (b / a) - (1 / (a * b)) = 1 :=
by sorry

end simplify_expression_l145_145821


namespace determine_m_l145_145278

theorem determine_m (x y m : ℝ) 
  (h1 : 3 * x + 2 * y = 4 * m - 5) 
  (h2 : 2 * x + 3 * y = m) 
  (h3 : x + y = 2) : 
  m = 3 :=
sorry

end determine_m_l145_145278


namespace arithmetic_sequence_problem_l145_145635

variable (d a1 : ℝ)
variable (h1 : a1 ≠ d)
variable (h2 : d ≠ 0)

theorem arithmetic_sequence_problem (S20 M : ℝ)
  (h3 : S20 = 10 * M)
  (x y : ℝ)
  (h4 : M = x * (a1 + 9 * d) + y * d) :
  x = 2 ∧ y = 1 := 
by 
  sorry

end arithmetic_sequence_problem_l145_145635


namespace range_of_a_l145_145065

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 - 2 * a

theorem range_of_a (a : ℝ) :
  (∃ (x₀ : ℝ), x₀ ≤ a ∧ f x₀ a ≥ 0) ↔ (a ∈ Set.Icc (-1 : ℝ) 0 ∪ Set.Ici 2) := by
  sorry

end range_of_a_l145_145065


namespace non_negative_combined_quadratic_l145_145039

theorem non_negative_combined_quadratic (a b c A B C : ℝ) (h1 : a ≥ 0) (h2 : b^2 ≤ a * c) (h3 : A ≥ 0) (h4 : B^2 ≤ A * C) :
  ∀ x : ℝ, a * A * x^2 + 2 * b * B * x + c * C ≥ 0 :=
by
  sorry

end non_negative_combined_quadratic_l145_145039


namespace feasible_test_for_rhombus_l145_145512

def is_rhombus (paper : Type) : Prop :=
  true -- Placeholder for the actual definition of a rhombus

def method_A (paper : Type) : Prop :=
  -- Placeholder for the condition "Measure if the four internal angles are equal"
  true

def method_B (paper : Type) : Prop :=
  -- Placeholder for the condition "Measure if the two diagonals are equal"
  true

def method_C (paper : Type) : Prop :=
  -- Placeholder for the condition "Measure if the distance from the intersection of the two diagonals to the four vertices is equal"
  true

def method_D (paper : Type) : Prop :=
  -- Placeholder for the condition "Fold the paper along the two diagonals separately and see if the parts on both sides of the diagonals coincide completely each time"
  true

theorem feasible_test_for_rhombus (paper : Type) : is_rhombus paper → method_D paper :=
by
  intro h_rhombus
  sorry

end feasible_test_for_rhombus_l145_145512


namespace smallest_k_base_representation_l145_145132

theorem smallest_k_base_representation :
  ∃ k : ℕ, (k > 0) ∧ (∀ n k, 0 = (42 * (1 - k^(n+1))/(1 - k))) ∧ (0 = (4 * (53 * (1 - k^(n+1))/(1 - k)))) →
  (k = 11) := sorry

end smallest_k_base_representation_l145_145132


namespace initial_deadline_l145_145229

theorem initial_deadline (D : ℝ) :
  (∀ (n : ℝ), (10 * 20) / 4 = n / 1) → 
  (∀ (m : ℝ), 8 * 75 = m * 3) →
  (∀ (d1 d2 : ℝ), d1 = 20 ∧ d2 = 93.75 → D = d1 + d2) →
  D = 113.75 :=
by {
  sorry
}

end initial_deadline_l145_145229


namespace transformed_area_l145_145516

noncomputable def area_transformation (f : ℝ → ℝ) (x1 x2 x3 : ℝ)
  (h : (1 / 2 * ((x2 - x1) * ((3 * f x3) - (3 * f x1))) - 1 / 2 * ((x3 - x2) * ((3 * f x1) - (3 * f x2)))) = 27) : Prop :=
  1 / 2 * ((0.5 * x2 - 0.5 * x1) * (3 * f (2 * x3) - 3 * f (2 * x1)) - 1 / 2 * (0.5 * x3 - 0.5 * x2) * (3 * f (2 * x1) - 3 * f (2 * x2))) = 40.5

theorem transformed_area
  (f : ℝ → ℝ) (x1 x2 x3 : ℝ)
  (h : 1 / 2 * ((x2 - x1) * (f x3 - f x1) - (x3 - x2) * (f x1 - f x2)) = 27) :
  1 / 2 * ((0.5 * x2 - 0.5 * x1) * (3 * f (2 * x3) - 3 * f (2 * x1)) - 1 / 2 * (0.5 * x3 - 0.5 * x2) * (3 * f (2 * x1) - 3 * f (2 * x2))) = 40.5 := sorry

end transformed_area_l145_145516


namespace calculate_f_zero_l145_145654

noncomputable def f (ω φ x : ℝ) := Real.sin (ω * x + φ)

theorem calculate_f_zero
  (ω φ : ℝ)
  (h_inc : ∀ x y : ℝ, (π / 6 < x ∧ x < y ∧ y < 2 * π / 3) → f ω φ x < f ω φ y)
  (h_symmetry1 : ∀ x : ℝ, f ω φ (π / 6 - x) = f ω φ (π / 6 + x))
  (h_symmetry2 : ∀ x : ℝ, f ω φ (2 * π / 3 - x) = f ω φ (2 * π / 3 + x)) :
  f ω φ 0 = -1 / 2 :=
sorry

end calculate_f_zero_l145_145654


namespace length_of_second_train_l145_145377

theorem length_of_second_train
  (length_first_train : ℝ)
  (speed_first_train : ℝ)
  (speed_second_train : ℝ)
  (time_to_cross : ℝ)
  (h1 : length_first_train = 270)
  (h2 : speed_first_train = 120)
  (h3 : speed_second_train = 80)
  (h4 : time_to_cross = 9) :
  ∃ length_second_train : ℝ, length_second_train = 229.95 :=
by
  sorry

end length_of_second_train_l145_145377


namespace three_digit_divisible_by_11_l145_145285

theorem three_digit_divisible_by_11
  (x y z : ℕ) (h1 : y = x + z) : (100 * x + 10 * y + z) % 11 = 0 :=
by
  sorry

end three_digit_divisible_by_11_l145_145285


namespace expression_positive_l145_145369

theorem expression_positive (x y z : ℝ) (h : x^2 + y^2 + z^2 ≠ 0) : 
  5 * x^2 + 5 * y^2 + 5 * z^2 + 6 * x * y - 8 * x * z - 8 * y * z > 0 := 
sorry

end expression_positive_l145_145369


namespace annika_hike_distance_l145_145424

-- Define the conditions as definitions
def hiking_rate : ℝ := 10  -- rate of 10 minutes per kilometer
def total_minutes : ℝ := 35 -- total available time in minutes
def total_distance_east : ℝ := 3 -- total distance hiked east

-- Define the statement to prove
theorem annika_hike_distance : ∃ (x : ℝ), (x / hiking_rate) + ((total_distance_east - x) / hiking_rate) = (total_minutes - 30) / hiking_rate :=
by
  sorry

end annika_hike_distance_l145_145424


namespace number_divisors_l145_145381

theorem number_divisors (p : ℕ) (h : p = 2^56 - 1) : ∃ x y : ℕ, 95 ≤ x ∧ x ≤ 105 ∧ 95 ≤ y ∧ y ≤ 105 ∧ p % x = 0 ∧ p % y = 0 ∧ x = 101 ∧ y = 127 :=
by {
  sorry
}

end number_divisors_l145_145381


namespace max_value_of_y_no_min_value_l145_145705

noncomputable def function_y (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

theorem max_value_of_y_no_min_value :
  (∃ x, -2 < x ∧ x < 2 ∧ function_y x = 5) ∧
  (∀ y, ∃ x, -2 < x ∧ x < 2 ∧ function_y x >= y) :=
by
  sorry

end max_value_of_y_no_min_value_l145_145705


namespace monkey_ladder_min_rungs_l145_145836

/-- 
  Proof that the minimum number of rungs n that allows the monkey to climb 
  to the top of the ladder and return to the ground, given that the monkey 
  ascends 16 rungs or descends 9 rungs at a time, is 24. 
-/
theorem monkey_ladder_min_rungs (n : ℕ) (ascend descend : ℕ) 
  (h1 : ascend = 16) (h2 : descend = 9) 
  (h3 : (∃ x y : ℤ, 16 * x - 9 * y = n) ∧ 
        (∃ x' y' : ℤ, 16 * x' - 9 * y' = 0)) : 
  n = 24 :=
sorry

end monkey_ladder_min_rungs_l145_145836


namespace geometric_sequence_k_value_l145_145376

theorem geometric_sequence_k_value (a : ℕ → ℝ) (S : ℕ → ℝ) (a1_pos : 0 < a 1)
  (geometric_seq : ∀ n, a (n + 2) = a n * (a 3 / a 1)) (h_a1 : a 1 = 1)
  (h_a3 : a 3 = 4) (h_Sk : S k = 63) :
  k = 6 := 
by
  sorry

end geometric_sequence_k_value_l145_145376


namespace tangent_line_at_point_A_l145_145674

noncomputable def curve (x : ℝ) : ℝ := Real.exp x

def point : ℝ × ℝ := (0, 1)

theorem tangent_line_at_point_A :
  ∃ m b : ℝ, (∀ x : ℝ, (curve x - (m * x + b))^2 = 0) ∧  
  m = 1 ∧ b = 1 :=
by
  sorry

end tangent_line_at_point_A_l145_145674


namespace suitable_b_values_l145_145882

theorem suitable_b_values (b : ℤ) :
  (∃ (c d e f : ℤ), 35 * c * d + (c * f + d * e) * b + 35 = 0 ∧
    c * e = 35 ∧ d * f = 35) →
  (∃ (k : ℤ), b = 2 * k) :=
by
  intro h
  sorry

end suitable_b_values_l145_145882


namespace sample_size_calculation_l145_145592

theorem sample_size_calculation (n : ℕ) (ratio_A_B_C q_A q_B q_C : ℕ) 
  (ratio_condition : ratio_A_B_C = 2 ∧ ratio_A_B_C * q_A = 2 ∧ ratio_A_B_C * q_B = 3 ∧ ratio_A_B_C * q_C = 5)
  (sample_A_units : q_A = 16) : n = 80 :=
sorry

end sample_size_calculation_l145_145592


namespace solve_system_l145_145150

theorem solve_system (X Y Z : ℝ)
  (h1 : 0.15 * 40 = 0.25 * X + 2)
  (h2 : 0.30 * 60 = 0.20 * Y + 3)
  (h3 : 0.10 * Z = X - Y) :
  X = 16 ∧ Y = 75 ∧ Z = -590 :=
by
  sorry

end solve_system_l145_145150


namespace total_chocolate_bars_l145_145449

theorem total_chocolate_bars (small_boxes : ℕ) (bars_per_box : ℕ) 
  (h1 : small_boxes = 17) (h2 : bars_per_box = 26) 
  : small_boxes * bars_per_box = 442 :=
by sorry

end total_chocolate_bars_l145_145449


namespace product_value_4_l145_145106

noncomputable def product_of_values (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 5) : ℝ :=
(x - 1) * (y - 1)

theorem product_value_4 (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 5) : ∃ v : ℝ, product_of_values x y h = v ∧ v = 4 :=
sorry

end product_value_4_l145_145106


namespace players_count_l145_145192

theorem players_count (total_socks : ℕ) (socks_per_player : ℕ) (h1 : total_socks = 16) (h2 : socks_per_player = 2) :
  total_socks / socks_per_player = 8 :=
by
  sorry

end players_count_l145_145192


namespace circle_center_and_radius_locus_of_midpoint_l145_145255

-- Part 1: Prove the equation of the circle C:
theorem circle_center_and_radius (a b r: ℝ) (hc: a + b = 2):
  (4 - a)^2 + b^2 = r^2 →
  (2 - a)^2 + (2 - b)^2 = r^2 →
  a = 2 ∧ b = 0 ∧ r = 2 := by
  sorry

-- Part 2: Prove the locus of the midpoint M:
theorem locus_of_midpoint (x y : ℝ) :
  ∃ (x1 y1 : ℝ), (x1 - 2)^2 + y1^2 = 4 ∧ x = (x1 + 5) / 2 ∧ y = y1 / 2 →
  x^2 - 7*x + y^2 + 45/4 = 0 := by
  sorry

end circle_center_and_radius_locus_of_midpoint_l145_145255


namespace operations_correctness_l145_145344

theorem operations_correctness (a b : ℝ) : 
  ((-ab)^2 ≠ -a^2 * b^2)
  ∧ (a^3 * a^2 ≠ a^6)
  ∧ ((a^3)^4 ≠ a^7)
  ∧ (b^2 + b^2 = 2 * b^2) :=
by
  sorry

end operations_correctness_l145_145344


namespace find_m_n_l145_145498

theorem find_m_n : ∃ (m n : ℕ), 2^n + 1 = m^2 ∧ m = 3 ∧ n = 3 :=
by {
  sorry
}

end find_m_n_l145_145498


namespace max_value_xyz_l145_145054

theorem max_value_xyz 
  (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_sum : x + y + z = 3) : 
  ∃ M, M = 243 ∧ (x + y^4 + z^5) ≤ M := 
  by sorry

end max_value_xyz_l145_145054


namespace first_term_geometric_progression_l145_145302

theorem first_term_geometric_progression (S : ℝ) (sum_first_two_terms : ℝ) (a : ℝ) (r : ℝ) :
  S = 8 → sum_first_two_terms = 5 →
  (a = 8 * (1 - (Real.sqrt 6) / 4)) ∨ (a = 8 * (1 + (Real.sqrt 6) / 4)) :=
by
  sorry

end first_term_geometric_progression_l145_145302


namespace find_divisor_for_multiple_l145_145102

theorem find_divisor_for_multiple (d : ℕ) :
  (∃ k : ℕ, k * d % 1821 = 710 ∧ k * d % 24 = 13 ∧ k * d = 3024) →
  d = 23 :=
by
  intros h
  sorry

end find_divisor_for_multiple_l145_145102


namespace tan_product_pi_8_l145_145289

theorem tan_product_pi_8 :
  (Real.tan (π / 8)) * (Real.tan (3 * π / 8)) * (Real.tan (5 * π / 8)) * (Real.tan (7 * π / 8)) = 1 :=
sorry

end tan_product_pi_8_l145_145289


namespace solve_for_A_l145_145040

def hash (A B : ℝ) : ℝ := A^2 + B^2

theorem solve_for_A (A : ℝ) (h : hash A 7 = 200) : A = Real.sqrt 151 :=
by
  sorry

end solve_for_A_l145_145040


namespace seq_nonzero_l145_145115

def seq (a : ℕ → ℤ) : Prop :=
  (a 1 = 1) ∧ (a 2 = 2) ∧ (∀ n, n ≥ 3 → 
    (if (a (n - 2) * a (n - 1)) % 2 = 0 
     then a n = 5 * a (n - 1) - 3 * a (n - 2) 
     else a n = a (n - 1) - a (n - 2)))

theorem seq_nonzero (a : ℕ → ℤ) (h : seq a) : ∀ n, n > 0 → a n ≠ 0 :=
  sorry

end seq_nonzero_l145_145115


namespace div_decimal_l145_145819

theorem div_decimal (a b : ℝ)  (h₁ : a = 0.45) (h₂ : b = 0.005):
  a / b = 90 :=
by {
  sorry
}

end div_decimal_l145_145819


namespace sector_area_eq_three_halves_l145_145832

theorem sector_area_eq_three_halves (θ R S : ℝ) (hθ : θ = 3) (h₁ : 2 * R + θ * R = 5) :
  S = 3 / 2 :=
by
  sorry

end sector_area_eq_three_halves_l145_145832


namespace competition_results_correct_l145_145156

theorem competition_results_correct :
  ∃ (first second third fourth : String), 
    (first = "Oleg" ∧ second = "Olya" ∧ third = "Polya" ∧ fourth = "Pasha") ∧
    ∀ (claims : String → String → Prop),
      (claims "Olya" "all_odd_places_boys") ∧ 
      (claims "Oleg" "consecutive_places_with_olya") ∧
      (claims "Pasha" "all_odd_places_names_start_O") ∧
      ∃ (truth_teller : String), 
        truth_teller = "Oleg" ∧ 
        (claims "Oleg" "first_place") ∧ 
        ¬ (claims "Olya" "first_place") ∧ 
        ¬ (claims "Pasha" "first_place") ∧ 
        ¬ (claims "Polya" "first_place") :=
sorry

end competition_results_correct_l145_145156


namespace product_of_ratios_eq_l145_145878

theorem product_of_ratios_eq :
  (∃ x_1 y_1 x_2 y_2 x_3 y_3 : ℝ,
    (x_1^3 - 3 * x_1 * y_1^2 = 2006) ∧
    (y_1^3 - 3 * x_1^2 * y_1 = 2007) ∧
    (x_2^3 - 3 * x_2 * y_2^2 = 2006) ∧
    (y_2^3 - 3 * x_2^2 * y_2 = 2007) ∧
    (x_3^3 - 3 * x_3 * y_3^2 = 2006) ∧
    (y_3^3 - 3 * x_3^2 * y_3 = 2007)) →
    (1 - x_1 / y_1) * (1 - x_2 / y_2) * (1 - x_3 / y_3) = 1 / 1003 :=
by
  sorry

end product_of_ratios_eq_l145_145878


namespace value_of_f_neg_a_l145_145412

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = -2 := 
by 
  sorry

end value_of_f_neg_a_l145_145412


namespace inequality_holds_l145_145463

theorem inequality_holds : ∀ (n : ℕ), (n - 1)^(n + 1) * (n + 1)^(n - 1) < n^(2 * n) :=
by sorry

end inequality_holds_l145_145463


namespace tomatoes_picked_second_week_l145_145386

-- Define the constants
def initial_tomatoes : Nat := 100
def fraction_picked_first_week : Nat := 1 / 4
def remaining_tomatoes : Nat := 15

-- Theorem to prove the number of tomatoes Jane picked in the second week
theorem tomatoes_picked_second_week (x : Nat) :
  let T := initial_tomatoes
  let p := fraction_picked_first_week
  let r := remaining_tomatoes
  let first_week_pick := T * p
  let remaining_after_first := T - first_week_pick
  let total_picked := remaining_after_first - r
  let second_week_pick := total_picked / 3
  second_week_pick = 20 := 
sorry

end tomatoes_picked_second_week_l145_145386


namespace range_of_x_for_direct_above_inverse_l145_145259

-- The conditions
def is_intersection_point (p : ℝ × ℝ) (k1 k2 : ℝ) : Prop :=
  let (x, y) := p
  y = k1 * x ∧ y = k2 / x

-- The main proof that we need to show
theorem range_of_x_for_direct_above_inverse :
  (∃ k1 k2 : ℝ, is_intersection_point (2, -1/3) k1 k2) →
  {x : ℝ | -1/6 * x > -2/(3 * x)} = {x : ℝ | x < -2 ∨ (0 < x ∧ x < 2)} :=
by
  intros
  sorry

end range_of_x_for_direct_above_inverse_l145_145259


namespace union_complement_eq_l145_145772

open Set

variable (U A B : Set ℕ)

def complement (U A : Set ℕ) : Set ℕ := { x | x ∈ U ∧ x ∉ A }

theorem union_complement_eq (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hA : A = {1, 2, 3}) (hB : B = {2, 4}) :
  (complement U A) ∪ B = {0, 2, 4} :=
by
  rw [hU, hA, hB]
  sorry

end union_complement_eq_l145_145772


namespace angle_B_equiv_60_l145_145462

noncomputable def triangle_condition (a b c : ℝ) (A B C : ℝ) : Prop :=
  2 * b * Real.cos B = a * Real.cos C + c * Real.cos A

theorem angle_B_equiv_60 
  (a b c A B C : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 0 < A) (h5 : A < π)
  (h6 : 0 < B) (h7 : B < π)
  (h8 : 0 < C) (h9 : C < π)
  (h_triangle : A + B + C = π)
  (h_arith : triangle_condition a b c A B C) : 
  B = π / 3 :=
by
  sorry

end angle_B_equiv_60_l145_145462


namespace geom_seq_a1_l145_145361

-- Define a geometric sequence.
def geom_seq (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 * q ^ n

-- Given conditions
def a2 (a : ℕ → ℝ) : Prop := a 1 = 2 -- because a2 = a(1) in zero-indexed
def a5 (a : ℕ → ℝ) : Prop := a 4 = -54 -- because a5 = a(4) in zero-indexed

-- Prove that a1 = -2/3
theorem geom_seq_a1 (a : ℕ → ℝ) (a1 q : ℝ) (h_geom : geom_seq a a1 q)
  (h_a2 : a2 a) (h_a5 : a5 a) : a1 = -2 / 3 :=
by
  sorry

end geom_seq_a1_l145_145361


namespace second_group_work_days_l145_145151

theorem second_group_work_days (M B : ℕ) (d1 d2 : ℕ) (H1 : M = 2 * B) 
  (H2 : (12 * M + 16 * B) * 5 = d1) (H3 : (13 * M + 24 * B) * d2 = d1) : 
  d2 = 4 :=
by
  sorry

end second_group_work_days_l145_145151


namespace smallest_integer_in_set_l145_145464

theorem smallest_integer_in_set (n : ℤ) (h : n+4 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) / 5)) : n ≥ 0 :=
by sorry

end smallest_integer_in_set_l145_145464


namespace find_circle_equation_l145_145855

noncomputable def center (m : ℝ) := (3 * m, m)

def radius (m : ℝ) : ℝ := 3 * m

def circle_eq (m : ℝ) (x y : ℝ) : Prop :=
  (x - 3 * m)^2 + (y - m)^2 = (radius m)^2

def point_A : ℝ × ℝ := (6, 1)

theorem find_circle_equation (m : ℝ) :
  (radius m = 3 * m ∧ center m = (3 * m, m) ∧ 
   point_A = (6, 1) ∧
   circle_eq m 6 1) →
  (circle_eq 1 x y ∨ circle_eq 37 x y) :=
by
  sorry

end find_circle_equation_l145_145855


namespace min_value_a4b3c2_l145_145391

theorem min_value_a4b3c2 {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 1/a + 1/b + 1/c = 9) :
  a ^ 4 * b ^ 3 * c ^ 2 ≥ 1 / 5184 := 
sorry

end min_value_a4b3c2_l145_145391


namespace find_a9_l145_145771

theorem find_a9 (a_1 a_2 : ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n ≥ 1, a (n + 2) = a (n + 1) + a n)
  (h2 : a 7 = 210)
  (h3 : a 1 = a_1)
  (h4 : a 2 = a_2) : 
  a 9 = 550 := by
  sorry

end find_a9_l145_145771


namespace total_slices_sold_l145_145063

theorem total_slices_sold (sold_yesterday served_today : ℕ) (h1 : sold_yesterday = 5) (h2 : served_today = 2) :
  sold_yesterday + served_today = 7 :=
by
  -- Proof skipped
  exact sorry

end total_slices_sold_l145_145063


namespace total_animals_to_spay_l145_145159

theorem total_animals_to_spay : 
  ∀ (c d : ℕ), c = 7 → d = 2 * c → c + d = 21 :=
by
  intros c d h1 h2
  sorry

end total_animals_to_spay_l145_145159


namespace count_three_digit_integers_with_remainder_3_div_7_l145_145300

theorem count_three_digit_integers_with_remainder_3_div_7 :
  ∃ n, (100 ≤ 7 * n + 3 ∧ 7 * n + 3 < 1000) ∧
  ∀ m, (100 ≤ 7 * m + 3 ∧ 7 * m + 3 < 1000) → m - n < 142 - 14 + 1 :=
by
  sorry

end count_three_digit_integers_with_remainder_3_div_7_l145_145300


namespace range_of_a_l145_145623

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 4 → (1/3 * x^3 - (a/2) * x^2 + (a-1) * x + 1) < 0) ∧
  (∀ x : ℝ, x > 6 → (1/3 * x^3 - (a/2) * x^2 + (a-1) * x + 1) > 0)
  ↔ (5 < a ∧ a < 7) :=
sorry

end range_of_a_l145_145623


namespace mary_earns_per_home_l145_145793

theorem mary_earns_per_home :
  let total_earned := 12696
  let homes_cleaned := 276.0
  total_earned / homes_cleaned = 46 :=
by
  sorry

end mary_earns_per_home_l145_145793


namespace x_squared_plus_y_squared_l145_145348

theorem x_squared_plus_y_squared (x y : ℝ) (h₁ : x - y = 18) (h₂ : x * y = 9) : x^2 + y^2 = 342 := by
  sorry

end x_squared_plus_y_squared_l145_145348


namespace ensure_two_of_each_l145_145753

theorem ensure_two_of_each {A B : ℕ} (hA : A = 10) (hB : B = 10) :
  ∃ n : ℕ, n = 12 ∧
  ∀ (extracted : ℕ → ℕ),
    (extracted 0 + extracted 1 = n) →
    (extracted 0 ≥ 2 ∧ extracted 1 ≥ 2) :=
by
  sorry

end ensure_two_of_each_l145_145753


namespace smallest_k_for_positive_roots_5_l145_145351

noncomputable def smallest_k_for_positive_roots : ℕ := 5

theorem smallest_k_for_positive_roots_5
  (k p q : ℕ) 
  (hk : k = smallest_k_for_positive_roots)
  (hq_pos : 0 < q)
  (h_distinct_pos_roots : ∃ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧ 
    k * x₁ * x₂ = q ∧ k * x₁ + k * x₂ > p ∧ k * x₁ * x₂ < q * ( 1 / (x₁*(1 - x₁) * x₂ * (1 - x₂)))) :
  k = 5 :=
by
  sorry

end smallest_k_for_positive_roots_5_l145_145351


namespace chickens_pigs_legs_l145_145263

variable (x : ℕ)

-- Define the conditions
def sum_chickens_pigs (x : ℕ) : Prop := x + (70 - x) = 70
def total_legs (x : ℕ) : Prop := 2 * x + 4 * (70 - x) = 196

-- Main theorem to prove the given mathematical statement
theorem chickens_pigs_legs (x : ℕ) (h1 : sum_chickens_pigs x) (h2 : total_legs x) : (2 * x + 4 * (70 - x) = 196) :=
by sorry

end chickens_pigs_legs_l145_145263


namespace escher_probability_l145_145427

def num_arrangements (n : ℕ) : ℕ := Nat.factorial n

def favorable_arrangements (total_art : ℕ) (escher_prints : ℕ) : ℕ :=
  num_arrangements (total_art - escher_prints + 1) * num_arrangements escher_prints

def total_arrangements (total_art : ℕ) : ℕ :=
  num_arrangements total_art

def prob_all_escher_consecutive (total_art : ℕ) (escher_prints : ℕ) : ℚ :=
  favorable_arrangements total_art escher_prints / total_arrangements total_art

theorem escher_probability :
  prob_all_escher_consecutive 12 4 = 1/55 :=
by
  sorry

end escher_probability_l145_145427


namespace leftmost_digit_base9_l145_145962

theorem leftmost_digit_base9 (x : ℕ) (h : x = 3^19 + 2*3^18 + 1*3^17 + 1*3^16 + 2*3^15 + 2*3^14 + 1*3^13 + 1*3^12 + 1*3^11 + 2*3^10 + 2*3^9 + 2*3^8 + 1*3^7 + 1*3^6 + 1*3^5 + 1*3^4 + 2*3^3 + 2*3^2 + 2*3^1 + 2) : ℕ :=
by
  sorry

end leftmost_digit_base9_l145_145962


namespace solve_exp_l145_145760

theorem solve_exp (x : ℕ) : 8^x = 2^9 → x = 3 :=
by
  sorry

end solve_exp_l145_145760


namespace min_workers_for_profit_l145_145213

def revenue (n : ℕ) : ℕ := 240 * n
def cost (n : ℕ) : ℕ := 600 + 200 * n

theorem min_workers_for_profit (n : ℕ) (h : 240 * n > 600 + 200 * n) : n >= 16 :=
by {
  -- Placeholder for the proof steps (which are not required per instructions)
  sorry
}

end min_workers_for_profit_l145_145213


namespace avg_calculation_l145_145827

-- Define averages
def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem avg_calculation : avg3 (avg3 2 2 0) (avg2 0 2) 0 = 7 / 9 :=
  by
    sorry

end avg_calculation_l145_145827


namespace polynomial_coef_sum_l145_145060

theorem polynomial_coef_sum :
  ∃ (a b c d : ℝ), (∀ x : ℝ, (4 * x^2 - 6 * x + 3) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) ∧ (8 * a + 4 * b + 2 * c + d = 14) :=
by
  sorry

end polynomial_coef_sum_l145_145060


namespace value_of_g_at_3_l145_145019

def g (x : ℝ) := x^2 + 1

theorem value_of_g_at_3 : g 3 = 10 := by
  sorry

end value_of_g_at_3_l145_145019


namespace painter_time_remaining_l145_145614

theorem painter_time_remaining (total_rooms : ℕ) (time_per_room : ℕ) (rooms_painted : ℕ) (remaining_hours : ℕ)
  (h1 : total_rooms = 12) (h2 : time_per_room = 7) (h3 : rooms_painted = 5) 
  (h4 : remaining_hours = (total_rooms - rooms_painted) * time_per_room) : 
  remaining_hours = 49 :=
by
  sorry

end painter_time_remaining_l145_145614


namespace distance_between_foci_l145_145330

theorem distance_between_foci (x y : ℝ)
    (h : 2 * x^2 - 12 * x - 8 * y^2 + 16 * y = 100) :
    2 * Real.sqrt 68.75 =
    2 * Real.sqrt (55 + 13.75) :=
by
  sorry

end distance_between_foci_l145_145330


namespace a7_plus_a11_l145_145223

variable {a : ℕ → ℤ} (d : ℤ) (a₁ : ℤ)

-- Definitions based on given conditions
def S_n (n : ℕ) := (n * (2 * a₁ + (n - 1) * d)) / 2
def a_n (n : ℕ) := a₁ + (n - 1) * d

-- Condition: S_17 = 51
axiom h : S_n 17 = 51

-- Theorem to prove the question is equivalent to the answer
theorem a7_plus_a11 (h : S_n 17 = 51) : a_n 7 + a_n 11 = 6 :=
by
  -- This is where you'd fill in the actual proof, but we'll use sorry for now
  sorry

end a7_plus_a11_l145_145223


namespace max_ab_l145_145528

theorem max_ab (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a + 4 * b = 8) :
  ab ≤ 4 :=
sorry

end max_ab_l145_145528


namespace distance_around_track_l145_145483

-- Define the conditions
def total_mileage : ℝ := 10
def distance_to_high_school : ℝ := 3
def round_trip_distance : ℝ := 2 * distance_to_high_school

-- State the question and the desired proof problem
theorem distance_around_track : 
  total_mileage - round_trip_distance = 4 := 
by
  sorry

end distance_around_track_l145_145483


namespace bruno_initial_books_l145_145573

theorem bruno_initial_books :
  ∃ (B : ℕ), B - 4 + 10 = 39 → B = 33 :=
by
  use 33
  intro h
  linarith [h]

end bruno_initial_books_l145_145573


namespace shire_total_population_l145_145886

theorem shire_total_population :
  let n := 25
  let avg_pop_min := 5400
  let avg_pop_max := 5700
  let avg_pop := (avg_pop_min + avg_pop_max) / 2
  n * avg_pop = 138750 :=
by
  let n := 25
  let avg_pop_min := 5400
  let avg_pop_max := 5700
  let avg_pop := (avg_pop_min + avg_pop_max) / 2
  show n * avg_pop = 138750
  sorry

end shire_total_population_l145_145886


namespace solve_for_y_l145_145859

theorem solve_for_y (y : ℚ) (h : |(4 : ℚ) * y - 6| = 0) : y = 3 / 2 :=
sorry

end solve_for_y_l145_145859


namespace find_third_term_l145_145010

theorem find_third_term :
  ∃ (a : ℕ → ℝ), a 0 = 5 ∧ a 4 = 2025 ∧ (∀ n, a (n + 1) = a n * r) ∧ a 2 = 225 :=
by
  sorry

end find_third_term_l145_145010


namespace triangular_prism_sliced_faces_l145_145813

noncomputable def resulting_faces_count : ℕ :=
  let initial_faces := 5 -- 2 bases + 3 lateral faces
  let additional_faces := 3 -- from the slices
  initial_faces + additional_faces

theorem triangular_prism_sliced_faces :
  resulting_faces_count = 8 := by
  sorry

end triangular_prism_sliced_faces_l145_145813


namespace factorize_expression_l145_145700

theorem factorize_expression (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_expression_l145_145700


namespace candies_leftover_l145_145666

theorem candies_leftover (n : ℕ) : 31254389 % 6 = 5 :=
by {
  sorry
}

end candies_leftover_l145_145666


namespace reflection_line_slope_l145_145104

theorem reflection_line_slope (m b : ℝ)
  (h_reflection : ∀ (x1 y1 x2 y2 : ℝ), 
    x1 = 2 ∧ y1 = 3 ∧ x2 = 10 ∧ y2 = 7 → 
    (x1 + x2) / 2 = (10 - 2) / 2 ∧ (y1 + y2) / 2 = (7 - 3) / 2 ∧ 
    y1 = m * x1 + b ∧ y2 = m * x2 + b) :
  m + b = 15 :=
sorry

end reflection_line_slope_l145_145104


namespace find_standard_equation_of_ellipse_l145_145459

noncomputable def ellipse_equation (a c b : ℝ) : Prop :=
  ∃ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ∨ (y^2 / a^2 + x^2 / b^2 = 1)

theorem find_standard_equation_of_ellipse (h1 : 2 * a = 12) (h2 : c / a = 1 / 3) :
  ellipse_equation 6 2 4 :=
by
  -- We are proving that given the conditions, the standard equation of the ellipse is as stated
  sorry

end find_standard_equation_of_ellipse_l145_145459


namespace proof_line_eq_l145_145141

variable (a T : ℝ) (line : ℝ × ℝ → Prop)

def line_eq (point : ℝ × ℝ) : Prop := 
  point.2 = (-2 * T / a^2) * point.1 + (2 * T / a)

def correct_line_eq (point : ℝ × ℝ) : Prop :=
  -2 * T * point.1 + a^2 * point.2 + 2 * a * T = 0

theorem proof_line_eq :
  ∀ point : ℝ × ℝ, line_eq a T point ↔ correct_line_eq a T point :=
by
  sorry

end proof_line_eq_l145_145141


namespace native_answer_l145_145517

-- Define properties to represent native types
inductive NativeType
| normal
| zombie
| half_zombie

-- Define the function that determines the response of a native
def response (native : NativeType) : String :=
  match native with
  | NativeType.normal => "да"
  | NativeType.zombie => "да"
  | NativeType.half_zombie => "да"

-- Define the main theorem
theorem native_answer (native : NativeType) : response native = "да" :=
by sorry

end native_answer_l145_145517


namespace algebraic_expression_value_zero_l145_145491

theorem algebraic_expression_value_zero (a b : ℝ) (h : a - b = 2) : (a^3 - 2 * a^2 * b + a * b^2 - 4 * a = 0) :=
sorry

end algebraic_expression_value_zero_l145_145491


namespace andrey_stamps_count_l145_145804

theorem andrey_stamps_count (x : ℕ) : 
  (x % 3 = 1) ∧ (x % 5 = 3) ∧ (x % 7 = 5) ∧ (150 < x ∧ x ≤ 300) → x = 208 := 
by 
  sorry

end andrey_stamps_count_l145_145804


namespace minimum_sticks_broken_n12_can_form_square_n15_l145_145275

-- Define the total length function
def total_length (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- For n = 12, prove that at least 2 sticks need to be broken to form a square
theorem minimum_sticks_broken_n12 : ∀ (n : ℕ), n = 12 → total_length n % 4 ≠ 0 → 2 = 2 := 
by 
  intros n h1 h2
  sorry

-- For n = 15, prove that a square can be directly formed
theorem can_form_square_n15 : ∀ (n : ℕ), n = 15 → total_length n % 4 = 0 := 
by 
  intros n h1
  sorry

end minimum_sticks_broken_n12_can_form_square_n15_l145_145275


namespace correct_relationship_5_25_l145_145333

theorem correct_relationship_5_25 : 5^2 = 25 :=
by
  sorry

end correct_relationship_5_25_l145_145333


namespace general_term_min_value_S_n_l145_145258

-- Definitions and conditions according to the problem statement
variable (d : ℤ) (a₁ : ℤ) (n : ℕ)

def a_n (n : ℕ) : ℤ := a₁ + (n - 1) * d
def S_n (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

-- Given conditions
axiom positive_common_difference : 0 < d
axiom a3_a4_product : a_n 3 * a_n 4 = 117
axiom a2_a5_sum : a_n 2 + a_n 5 = -22

-- Proof 1: General term of the arithmetic sequence
theorem general_term : a_n n = 4 * (n : ℤ) - 25 :=
  by sorry

-- Proof 2: Minimum value of the sum of the first n terms
theorem min_value_S_n : S_n 6 = -66 :=
  by sorry

end general_term_min_value_S_n_l145_145258


namespace number_eq_180_l145_145800

theorem number_eq_180 (x : ℝ) (h : 64 + 5 * 12 / (x / 3) = 65) : x = 180 :=
sorry

end number_eq_180_l145_145800


namespace jade_living_expenses_l145_145012

-- Definitions from the conditions
variable (income : ℝ) (insurance_fraction : ℝ) (savings : ℝ) (P : ℝ)

-- Constants from the given problem
noncomputable def jadeIncome : income = 1600 := by sorry
noncomputable def jadeInsuranceFraction : insurance_fraction = 1 / 5 := by sorry
noncomputable def jadeSavings : savings = 80 := by sorry

-- The proof problem statement
theorem jade_living_expenses :
    (P * 1600 + (1 / 5) * 1600 + 80 = 1600) → P = 3 / 4 := by
    intros h
    sorry

end jade_living_expenses_l145_145012


namespace midpoint_coords_l145_145273

noncomputable def F1 : (ℝ × ℝ) := (-2 * Real.sqrt 2, 0)
noncomputable def F2 : (ℝ × ℝ) := (2 * Real.sqrt 2, 0)
def major_axis_length : ℝ := 6
def line_eq (x y : ℝ) : Prop := x - y + 2 = 0

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  let a := 3
  let b := 1
  (x^2) / (a^2) + y^2 / (b^2) = 1

theorem midpoint_coords :
  ∃ (A B : ℝ × ℝ), ellipse_eq A.1 A.2 ∧ ellipse_eq B.1 B.2 ∧ line_eq A.1 A.2 ∧ line_eq B.1 B.2 →
  (A.1 + B.1) / 2 = -9 / 5 ∧ (A.2 + B.2) / 2 = 1 / 5 :=
by
  sorry

end midpoint_coords_l145_145273


namespace surface_area_of_rectangular_solid_l145_145313

-- Conditions
variables {a b c : ℕ}
variables (h_a_prime : Nat.Prime a) (h_b_prime : Nat.Prime b) (h_c_prime : Nat.Prime c)
variables (h_volume : a * b * c = 308)

-- Question and Proof Problem
theorem surface_area_of_rectangular_solid :
  2 * (a * b + b * c + c * a) = 226 :=
sorry

end surface_area_of_rectangular_solid_l145_145313


namespace product_of_two_numbers_l145_145940

-- Define HCF function
def HCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Define LCM function
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- Define the conditions for the problem
def problem_conditions (x y : ℕ) : Prop :=
  HCF x y = 55 ∧ LCM x y = 1500

-- State the theorem that should be proven
theorem product_of_two_numbers (x y : ℕ) (h_conditions : problem_conditions x y) :
  x * y = 82500 :=
by
  sorry

end product_of_two_numbers_l145_145940


namespace lance_hourly_earnings_l145_145926

theorem lance_hourly_earnings
  (hours_per_week : ℕ)
  (workdays_per_week : ℕ)
  (daily_earnings : ℕ)
  (total_weekly_earnings : ℕ)
  (hourly_wage : ℕ)
  (h1 : hours_per_week = 35)
  (h2 : workdays_per_week = 5)
  (h3 : daily_earnings = 63)
  (h4 : total_weekly_earnings = daily_earnings * workdays_per_week)
  (h5 : total_weekly_earnings = hourly_wage * hours_per_week)
  : hourly_wage = 9 :=
sorry

end lance_hourly_earnings_l145_145926


namespace solve_for_y_l145_145925

theorem solve_for_y (x y : ℤ) (h1 : x + y = 260) (h2 : x - y = 200) : y = 30 := by
  sorry

end solve_for_y_l145_145925


namespace avg_age_adults_l145_145460

-- Given conditions
def num_members : ℕ := 50
def avg_age_members : ℕ := 20
def num_girls : ℕ := 25
def num_boys : ℕ := 20
def num_adults : ℕ := 5
def avg_age_girls : ℕ := 18
def avg_age_boys : ℕ := 22

-- Prove that the average age of the adults is 22 years
theorem avg_age_adults :
  (num_members * avg_age_members - num_girls * avg_age_girls - num_boys * avg_age_boys) / num_adults = 22 :=
by 
  sorry

end avg_age_adults_l145_145460


namespace relationship_between_ys_l145_145604

theorem relationship_between_ys :
  ∀ (y1 y2 y3 : ℝ),
    (y1 = - (6 / (-2))) ∧ (y2 = - (6 / (-1))) ∧ (y3 = - (6 / 3)) →
    y2 > y1 ∧ y1 > y3 :=
by sorry

end relationship_between_ys_l145_145604


namespace balloons_kept_by_Andrew_l145_145967

theorem balloons_kept_by_Andrew :
  let blue := 303
  let purple := 453
  let red := 165
  let yellow := 324
  let blue_kept := (2/3 : ℚ) * blue
  let purple_kept := (3/5 : ℚ) * purple
  let red_kept := (4/7 : ℚ) * red
  let yellow_kept := (1/3 : ℚ) * yellow
  let total_kept := blue_kept.floor + purple_kept.floor + red_kept.floor + yellow_kept
  total_kept = 675 := by
  sorry

end balloons_kept_by_Andrew_l145_145967


namespace find_ordered_pairs_l145_145606

theorem find_ordered_pairs :
  {p : ℝ × ℝ | p.1 > p.2 ∧ (p.1 - p.2 = 2 * p.1 / p.2 ∨ p.1 - p.2 = 2 * p.2 / p.1)} = 
  {(8, 4), (9, 3), (2, 1)} :=
sorry

end find_ordered_pairs_l145_145606


namespace brenda_age_l145_145418

variable (A B J : ℕ)

theorem brenda_age :
  (A = 3 * B) →
  (J = B + 6) →
  (A = J) →
  (B = 3) :=
by
  intros h1 h2 h3
  -- condition: A = 3 * B
  -- condition: J = B + 6
  -- condition: A = J
  -- prove B = 3
  sorry

end brenda_age_l145_145418


namespace polynomial_irreducible_segment_intersect_l145_145902

-- Part (a)
theorem polynomial_irreducible 
  (f : Polynomial ℤ) 
  (h_def : f = Polynomial.C 12 + Polynomial.X * Polynomial.C 9 + Polynomial.X^2 * Polynomial.C 6 + Polynomial.X^3 * Polynomial.C 3 + Polynomial.X^4) : 
  ¬ ∃ (p q : Polynomial ℤ), (Polynomial.degree p = 2) ∧ (Polynomial.degree q = 2) ∧ (f = p * q) :=
sorry

-- Part (b)
theorem segment_intersect 
  (n : ℕ) 
  (segments : Fin (2*n+1) → Set (ℝ × ℝ)) 
  (h_intersect : ∀ i, ∃ n_indices : Finset (Fin (2*n+1)), n_indices.card = n ∧ ∀ j ∈ n_indices, (segments i ∩ segments j).Nonempty) :
  ∃ i, ∀ j, i ≠ j → (segments i ∩ segments j).Nonempty :=
sorry


end polynomial_irreducible_segment_intersect_l145_145902


namespace initial_non_electrified_part_l145_145658

variables (x y : ℝ)

def electrified_fraction : Prop :=
  x + y = 1 ∧ 2 * x + 0.75 * y = 1

theorem initial_non_electrified_part (h : electrified_fraction x y) : y = 4 / 5 :=
by {
  sorry
}

end initial_non_electrified_part_l145_145658


namespace no_solution_in_natural_numbers_l145_145828

theorem no_solution_in_natural_numbers (x y z : ℕ) (hxy : x ≠ 0) (hyz : y ≠ 0) (hzx : z ≠ 0) :
  ¬ (x / y + y / z + z / x = 1) :=
by sorry

end no_solution_in_natural_numbers_l145_145828


namespace man_reaches_home_at_11_pm_l145_145148

theorem man_reaches_home_at_11_pm :
  let start_time := 15 -- represents 3 pm in 24-hour format
  let level_speed := 4 -- km/hr
  let uphill_speed := 3 -- km/hr
  let downhill_speed := 6 -- km/hr
  let total_distance := 12 -- km
  let level_distance := 4 -- km
  let uphill_distance := 4 -- km
  let downhill_distance := 4 -- km
  let level_time := level_distance / level_speed -- time for 4 km on level ground
  let uphill_time := uphill_distance / uphill_speed -- time for 4 km uphill
  let downhill_time := downhill_distance / downhill_speed -- time for 4 km downhill
  let total_time_one_way := level_time + uphill_time + downhill_time + level_time
  let destination_time := start_time + total_time_one_way
  let return_time := destination_time + total_time_one_way
  return_time = 23 := -- represents 11 pm in 24-hour format
by
  sorry

end man_reaches_home_at_11_pm_l145_145148


namespace fraction_value_l145_145371

variable (x y : ℚ)

theorem fraction_value (h₁ : x = 4 / 6) (h₂ : y = 8 / 12) : 
  (6 * x + 8 * y) / (48 * x * y) = 7 / 16 :=
by
  sorry

end fraction_value_l145_145371


namespace triangle_inequality_l145_145082

theorem triangle_inequality
  (a b c : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_ineq : 5 * (a^2 + b^2 + c^2) < 6 * (a * b + b * c + c * a)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
by
  sorry

end triangle_inequality_l145_145082


namespace arithmetic_seq_sum_ratio_l145_145328

theorem arithmetic_seq_sum_ratio
  (a : ℕ → ℤ)
  (S : ℕ → ℤ)
  (h1 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h2 : S 25 / a 23 = 5)
  (h3 : S 45 / a 33 = 25) :
  S 65 / a 43 = 45 :=
by sorry

end arithmetic_seq_sum_ratio_l145_145328


namespace smaller_square_perimeter_l145_145625

theorem smaller_square_perimeter (s : ℕ) (h1 : 4 * s = 144) : 
  let smaller_s := s / 3 
  let smaller_perimeter := 4 * smaller_s 
  smaller_perimeter = 48 :=
by
  let smaller_s := s / 3
  let smaller_perimeter := 4 * smaller_s 
  sorry

end smaller_square_perimeter_l145_145625


namespace range_of_m_l145_145560

def A (m : ℝ) : Set ℝ := {x | x^2 - 4*m*x + 2*m + 6 = 0}
def B : Set ℝ := {x | x < 0}

theorem range_of_m (m : ℝ) (h : (A m) ∩ B ≠ ∅) : m ≤ -1 :=
sorry

end range_of_m_l145_145560


namespace parallelogram_opposite_sides_equal_l145_145797

-- Given definitions and properties of a parallelogram
structure Parallelogram (α : Type*) [Add α] [AddCommGroup α] [Module ℝ α] :=
(a b c d : α) 
(parallel_a : a + b = c + d)
(parallel_b : b + c = d + a)
(parallel_c : c + d = a + b)
(parallel_d : d + a = b + c)

open Parallelogram

-- Define problem statement to prove opposite sides are equal
theorem parallelogram_opposite_sides_equal {α : Type*} [Add α] [AddCommGroup α] [Module ℝ α] 
  (p : Parallelogram α) : 
  p.a = p.c ∧ p.b = p.d :=
sorry -- Proof goes here

end parallelogram_opposite_sides_equal_l145_145797


namespace running_time_15mph_l145_145227

theorem running_time_15mph (x y z : ℝ) (h1 : x + y + z = 14) (h2 : 15 * x + 10 * y + 8 * z = 164) :
  x = 3 :=
sorry

end running_time_15mph_l145_145227


namespace line_equation_l145_145286

-- Given conditions
variables (k x x0 y y0 : ℝ)
variable (line_passes_through : ∀ x0 y0, y0 = k * x0 + l)
variable (M0 : (ℝ × ℝ))

-- Main statement we need to prove
theorem line_equation (k x x0 y y0 : ℝ) (M0 : (ℝ × ℝ)) (line_passes_through : ∀ x0 y0, y0 = k * x0 + l) :
  y - y0 = k * (x - x0) :=
sorry

end line_equation_l145_145286


namespace num_ways_distribute_plants_correct_l145_145480

def num_ways_to_distribute_plants : Nat :=
  let basil := 2
  let aloe := 1
  let cactus := 1
  let white_lamps := 2
  let red_lamp := 1
  let blue_lamp := 1
  let plants := basil + aloe + cactus
  let lamps := white_lamps + red_lamp + blue_lamp
  4
  
theorem num_ways_distribute_plants_correct :
  num_ways_to_distribute_plants = 4 :=
by
  sorry -- Proof of the correctness of the distribution

end num_ways_distribute_plants_correct_l145_145480


namespace original_lettuce_cost_l145_145880

theorem original_lettuce_cost
  (original_cost: ℝ) (tomatoes_original: ℝ) (tomatoes_new: ℝ) (celery_original: ℝ) (celery_new: ℝ) (lettuce_new: ℝ)
  (delivery_tip: ℝ) (new_bill: ℝ)
  (H1: original_cost = 25)
  (H2: tomatoes_original = 0.99) (H3: tomatoes_new = 2.20)
  (H4: celery_original = 1.96) (H5: celery_new = 2.00)
  (H6: lettuce_new = 1.75)
  (H7: delivery_tip = 8.00)
  (H8: new_bill = 35) :
  ∃ (lettuce_original: ℝ), lettuce_original = 1.00 :=
by
  let tomatoes_diff := tomatoes_new - tomatoes_original
  let celery_diff := celery_new - celery_original
  let new_cost_without_lettuce := original_cost + tomatoes_diff + celery_diff
  let new_cost_excl_delivery := new_bill - delivery_tip
  have lettuce_diff := new_cost_excl_delivery - new_cost_without_lettuce
  let lettuce_original := lettuce_new - lettuce_diff
  exists lettuce_original
  sorry

end original_lettuce_cost_l145_145880


namespace caleb_apples_less_than_kayla_l145_145045

theorem caleb_apples_less_than_kayla :
  ∀ (Kayla Suraya Caleb : ℕ),
  (Kayla = 20) →
  (Suraya = Kayla + 7) →
  (Suraya = Caleb + 12) →
  (Suraya = 27) →
  (Kayla - Caleb = 5) :=
by
  intros Kayla Suraya Caleb hKayla hSuraya1 hSuraya2 hSuraya3
  sorry

end caleb_apples_less_than_kayla_l145_145045


namespace min_value_of_reciprocal_sum_l145_145079

theorem min_value_of_reciprocal_sum (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a + 2 * b = 1) (h2 : c + 2 * d = 1) :
  16 ≤ (1 / a) + 1 / (b * c * d) :=
by
  sorry

end min_value_of_reciprocal_sum_l145_145079


namespace power_of_negative_base_l145_145708

theorem power_of_negative_base : (-64 : ℤ)^(7 / 6) = -128 := by
  sorry

end power_of_negative_base_l145_145708


namespace shaded_area_l145_145433

open Real

theorem shaded_area (AH HF GF : ℝ) (AH_eq : AH = 12) (HF_eq : HF = 16) (GF_eq : GF = 4) 
  (DG : ℝ) (DG_eq : DG = 3) (area_triangle_DGF : ℝ) (area_triangle_DGF_eq : area_triangle_DGF = 6) :
  let area_square : ℝ := 4 * 4
  let shaded_area : ℝ := area_square - area_triangle_DGF
  shaded_area = 10 := by
    sorry

end shaded_area_l145_145433


namespace exists_six_distinct_naturals_l145_145747

theorem exists_six_distinct_naturals :
  ∃ (a b c d e f : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
    d ≠ e ∧ d ≠ f ∧ 
    e ≠ f ∧ 
    a + b + c + d + e + f = 3528 ∧
    (1/a + 1/b + 1/c + 1/d + 1/e + 1/f : ℝ) = 3528 / 2012 :=
sorry

end exists_six_distinct_naturals_l145_145747


namespace find_acute_angle_l145_145595

theorem find_acute_angle (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) 
    (h3 : Real.sin α = 1 - Real.sqrt 3 * Real.tan (π / 18) * Real.sin α) : 
    α = π / 3 * 5 / 9 :=
by
  sorry

end find_acute_angle_l145_145595


namespace factorial_expression_l145_145304

open Nat

theorem factorial_expression :
  7 * (6!) + 6 * (5!) + 2 * (5!) = 6000 :=
by
  sorry

end factorial_expression_l145_145304


namespace arithmetic_sequence_a5_value_l145_145436

theorem arithmetic_sequence_a5_value 
  (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a n = a 1 + (n - 1) * d)
  (h_a2 : a 2 = 1)
  (h_a8 : a 8 = 2 * a 6 + a 4) : 
  a 5 = -1 / 2 :=
by
  sorry

end arithmetic_sequence_a5_value_l145_145436


namespace find_equation_of_line_l145_145246

theorem find_equation_of_line 
  (l : ℝ → ℝ → Prop)
  (h_intersect : ∃ x y : ℝ, 2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0 ∧ l x y)
  (h_parallel : ∀ x y : ℝ, l x y → 4 * x - 3 * y - 6 = 0) :
  ∀ x y : ℝ, l x y ↔ 4 * x - 3 * y - 6 = 0 :=
by
  sorry

end find_equation_of_line_l145_145246


namespace max_difference_and_max_value_of_multiple_of_5_l145_145549

theorem max_difference_and_max_value_of_multiple_of_5:
  ∀ (N : ℕ), 
  (∃ (d : ℕ), d = 0 ∨ d = 5 ∧ N = 740 + d) →
  (∃ (diff : ℕ), diff = 5) ∧ (∃ (max_num : ℕ), max_num = 745) :=
by
  intro N
  rintro ⟨d, (rfl | rfl), rfl⟩
  apply And.intro
  use 5
  use 745
  sorry

end max_difference_and_max_value_of_multiple_of_5_l145_145549


namespace larger_cookie_sugar_l145_145979

theorem larger_cookie_sugar :
  let initial_cookies := 40
  let initial_sugar_per_cookie := 1 / 8
  let total_sugar := initial_cookies * initial_sugar_per_cookie
  let larger_cookies := 25
  let sugar_per_larger_cookie := total_sugar / larger_cookies
  sugar_per_larger_cookie = 1 / 5 := by
sorry

end larger_cookie_sugar_l145_145979


namespace domain_ln_2_minus_x_is_interval_l145_145244

noncomputable def domain_ln_2_minus_x : Set Real := { x : Real | 2 - x > 0 }

theorem domain_ln_2_minus_x_is_interval : domain_ln_2_minus_x = Set.Iio 2 :=
by
  sorry

end domain_ln_2_minus_x_is_interval_l145_145244


namespace jacks_speed_l145_145116

-- Define the initial distance between Jack and Christina.
def initial_distance : ℝ := 360

-- Define Christina's speed.
def christina_speed : ℝ := 7

-- Define Lindy's speed.
def lindy_speed : ℝ := 12

-- Define the total distance Lindy travels.
def lindy_total_distance : ℝ := 360

-- Prove Jack's speed given the conditions.
theorem jacks_speed : ∃ v : ℝ, (initial_distance - christina_speed * (lindy_total_distance / lindy_speed)) / (lindy_total_distance / lindy_speed) = v ∧ v = 5 :=
by {
  sorry
}

end jacks_speed_l145_145116


namespace angle_movement_condition_l145_145031

noncomputable def angle_can_reach_bottom_right (m n : ℕ) (h1 : 2 ≤ m) (h2 : 2 ≤ n) : Prop :=
  (m % 2 = 1) ∧ (n % 2 = 1)

theorem angle_movement_condition (m n : ℕ) (h1 : 2 ≤ m) (h2 : 2 ≤ n) :
  angle_can_reach_bottom_right m n h1 h2 ↔ (m % 2 = 1 ∧ n % 2 = 1) :=
sorry

end angle_movement_condition_l145_145031


namespace employee_salary_l145_145208

theorem employee_salary (X Y : ℝ) (h1 : X = 1.2 * Y) (h2 : X + Y = 528) : Y = 240 :=
by
  sorry

end employee_salary_l145_145208


namespace maximize_box_volume_l145_145074

noncomputable def volume (x : ℝ) := (16 - 2 * x) * (10 - 2 * x) * x

theorem maximize_box_volume :
  (∃ x : ℝ, volume x = 144 ∧ ∀ y : ℝ, 0 < y ∧ y < 5 → volume y ≤ volume 2) := 
by
  sorry

end maximize_box_volume_l145_145074


namespace relay_team_average_time_l145_145009

theorem relay_team_average_time :
  let d1 := 200
  let t1 := 38
  let d2 := 300
  let t2 := 56
  let d3 := 250
  let t3 := 47
  let d4 := 400
  let t4 := 80
  let total_distance := d1 + d2 + d3 + d4
  let total_time := t1 + t2 + t3 + t4
  let average_time_per_meter := total_time / total_distance
  average_time_per_meter = 0.1922 := by
  sorry

end relay_team_average_time_l145_145009


namespace sum_of_factors_636405_l145_145933

theorem sum_of_factors_636405 :
  ∃ (a b c : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ 10 ≤ c ∧ c < 100 ∧
    a * b * c = 636405 ∧ a + b + c = 259 :=
sorry

end sum_of_factors_636405_l145_145933


namespace train_crosses_signal_pole_in_12_seconds_l145_145690

noncomputable def time_to_cross_signal_pole (length_train : ℕ) (time_to_cross_platform : ℕ) (length_platform : ℕ) : ℕ :=
  let distance_train_platform := length_train + length_platform
  let speed_train := distance_train_platform / time_to_cross_platform
  let time_to_cross_pole := length_train / speed_train
  time_to_cross_pole

theorem train_crosses_signal_pole_in_12_seconds :
  time_to_cross_signal_pole 300 39 675 = 12 :=
by
  -- expected proof in the interactive mode
  sorry

end train_crosses_signal_pole_in_12_seconds_l145_145690


namespace sum_opposite_abs_val_eq_neg_nine_l145_145425

theorem sum_opposite_abs_val_eq_neg_nine (a b : ℤ) (h1 : a = -15) (h2 : b = 6) : a + b = -9 := 
by
  -- conditions given
  rw [h1, h2]
  -- skip the proof
  sorry

end sum_opposite_abs_val_eq_neg_nine_l145_145425


namespace log_equation_l145_145817

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_equation (x : ℝ) (h1 : x > 1) (h2 : (log_base_10 x)^2 - log_base_10 (x^4) = 32) :
  (log_base_10 x)^4 - log_base_10 (x^4) = 4064 :=
by
  sorry

end log_equation_l145_145817


namespace race_winner_laps_l145_145167

/-- Given:
  * A lap equals 100 meters.
  * Award per hundred meters is $3.5.
  * The winner earned $7 per minute.
  * The race lasted 12 minutes.
  Prove that the number of laps run by the winner is 24.
-/ 
theorem race_winner_laps :
  let lap_distance := 100 -- meters
  let award_per_100meters := 3.5 -- dollars per 100 meters
  let earnings_per_minute := 7 -- dollars per minute
  let race_duration := 12 -- minutes
  let total_earnings := earnings_per_minute * race_duration
  let total_100meters := total_earnings / award_per_100meters
  let laps := total_100meters
  laps = 24 := by
  sorry

end race_winner_laps_l145_145167


namespace find_ab_sum_l145_145718

theorem find_ab_sum
  (a b : ℝ)
  (h₁ : a^3 - 3 * a^2 + 5 * a - 1 = 0)
  (h₂ : b^3 - 3 * b^2 + 5 * b - 5 = 0) :
  a + b = 2 := by
  sorry

end find_ab_sum_l145_145718


namespace correct_proposition_l145_145701

theorem correct_proposition :
  (∃ x₀ : ℤ, x₀^2 = 1) ∧ ¬(∃ x₀ : ℤ, x₀^2 < 0) ∧ ¬(∀ x : ℤ, x^2 ≤ 0) ∧ ¬(∀ x : ℤ, x^2 ≥ 1) :=
by
  sorry

end correct_proposition_l145_145701


namespace largest_n_l145_145270

theorem largest_n (n x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 6 * x + 6 * y + 6 * z - 18 →
  n ≤ 3 := 
by 
  sorry

end largest_n_l145_145270


namespace rate_per_kg_grapes_l145_145678

/-- Define the conditions for the problem -/
def rate_per_kg_mangoes : ℕ := 55
def kg_grapes_purchased : ℕ := 3
def kg_mangoes_purchased : ℕ := 9
def total_paid : ℕ := 705

/-- The theorem statement to prove the rate per kg for grapes -/
theorem rate_per_kg_grapes (G : ℕ) :
  kg_grapes_purchased * G + kg_mangoes_purchased * rate_per_kg_mangoes = total_paid →
  G = 70 :=
by
  sorry -- Proof will go here

end rate_per_kg_grapes_l145_145678


namespace geometric_sequence_a_eq_one_l145_145374

theorem geometric_sequence_a_eq_one (a : ℝ) 
  (h₁ : ∃ (r : ℝ), a = 1 / (1 - r) ∧ r = a - 1/2 ∧ r ≠ 0) : 
  a = 1 := 
sorry

end geometric_sequence_a_eq_one_l145_145374


namespace jayden_current_age_l145_145566

def current_age_of_Jayden (e : ℕ) (j_in_3_years : ℕ) : ℕ :=
  j_in_3_years - 3

theorem jayden_current_age (e : ℕ) (h1 : e = 11) (h2 : ∃ j : ℕ, j = ((e + 3) / 2) ∧ j_in_3_years = j) : 
  current_age_of_Jayden e j_in_3_years = 4 :=
by
  sorry

end jayden_current_age_l145_145566


namespace race_distance_l145_145417

-- Definitions for the conditions
def A_time : ℕ := 20
def B_time : ℕ := 25
def A_beats_B_by : ℕ := 14

-- Definition of the function to calculate whether the total distance D is correct
def total_distance : ℕ := 56

-- The theorem statement without proof
theorem race_distance (D : ℕ) (A_time B_time A_beats_B_by : ℕ)
  (hA : A_time = 20)
  (hB : B_time = 25)
  (hAB : A_beats_B_by = 14)
  (h_eq : (D / A_time) * B_time = D + A_beats_B_by) : 
  D = total_distance :=
sorry

end race_distance_l145_145417


namespace find_a_l145_145655

theorem find_a (a : ℝ) :
  (∀ (x y : ℝ), (a * x + 2 * y + 3 * a = 0) → (3 * x + (a - 1) * y = a - 7)) → 
  a = 3 :=
by
  sorry

end find_a_l145_145655


namespace find_b_find_perimeter_b_plus_c_l145_145826

noncomputable def triangle_condition_1
  (a b c : ℝ) (A B C : ℝ) : Prop :=
  a * Real.cos B = (3 * c - b) * Real.cos A

noncomputable def triangle_condition_2
  (a b : ℝ) (C : ℝ) : Prop :=
  a * Real.sin C = 2 * Real.sqrt 2

noncomputable def triangle_condition_3
  (a b c : ℝ) (A : ℝ) : Prop :=
  (1 / 2) * b * c * Real.sin A = Real.sqrt 2

noncomputable def given_a
  (a : ℝ) : Prop :=
  a = 2 * Real.sqrt 2

theorem find_b
  (a b c A B C : ℝ)
  (h1 : triangle_condition_1 a b c A B C)
  (h2 : triangle_condition_2 a b B)
  (h3 : triangle_condition_3 a b c A)
  (h4 : given_a a) :
  b = 3 :=
sorry

theorem find_perimeter_b_plus_c
  (a b c A B C : ℝ)
  (h1 : triangle_condition_1 a b c A B C)
  (h2 : triangle_condition_2 a b B)
  (h3 : triangle_condition_3 a b c A)
  (h4 : given_a a) :
  b + c = 2 * Real.sqrt 3 :=
sorry

end find_b_find_perimeter_b_plus_c_l145_145826


namespace mary_characters_initial_D_l145_145168

theorem mary_characters_initial_D (total_characters initial_A initial_C initial_D initial_E : ℕ)
  (h1 : total_characters = 60)
  (h2 : initial_A = total_characters / 2)
  (h3 : initial_C = initial_A / 2)
  (remaining := total_characters - initial_A - initial_C)
  (h4 : remaining = initial_D + initial_E)
  (h5 : initial_D = 2 * initial_E) : initial_D = 10 := by
  sorry

end mary_characters_initial_D_l145_145168


namespace smaller_angle_measure_l145_145777

theorem smaller_angle_measure (α β : ℝ) (h1 : α + β = 90) (h2 : α = 4 * β) : β = 18 :=
by
  sorry

end smaller_angle_measure_l145_145777


namespace initial_jellybeans_l145_145260

theorem initial_jellybeans (J : ℕ) :
    (∀ x y : ℕ, x = 24 → y = 12 →
    (J - x - y + ((x + y) / 2) = 72) → J = 90) :=
by
  intros x y hx hy h
  rw [hx, hy] at h
  sorry

end initial_jellybeans_l145_145260


namespace inequality_multiplication_l145_145776

theorem inequality_multiplication (a b : ℝ) (h : a > b) : 3 * a > 3 * b :=
sorry

end inequality_multiplication_l145_145776


namespace new_weight_is_77_l145_145919

theorem new_weight_is_77 (weight_increase_per_person : ℝ) (number_of_persons : ℕ) (old_weight : ℝ) 
  (total_weight_increase : ℝ) (new_weight : ℝ) 
  (h1 : weight_increase_per_person = 1.5)
  (h2 : number_of_persons = 8)
  (h3 : old_weight = 65)
  (h4 : total_weight_increase = number_of_persons * weight_increase_per_person)
  (h5 : new_weight = old_weight + total_weight_increase) :
  new_weight = 77 :=
sorry

end new_weight_is_77_l145_145919


namespace ball_bounce_height_l145_145585

theorem ball_bounce_height (a : ℝ) (r : ℝ) (threshold : ℝ) (k : ℕ) 
  (h_a : a = 20) (h_r : r = 1/2) (h_threshold : threshold = 0.5) :
  20 * (r^k) < threshold ↔ k = 5 :=
by sorry

end ball_bounce_height_l145_145585


namespace sample_size_is_fifteen_l145_145656

variable (total_employees : ℕ) (young_employees : ℕ) (middle_aged_employees : ℕ)
variable (elderly_employees : ℕ) (young_sample_count : ℕ) (sample_size : ℕ)

theorem sample_size_is_fifteen
  (h1 : total_employees = 750)
  (h2 : young_employees = 350)
  (h3 : middle_aged_employees = 250)
  (h4 : elderly_employees = 150)
  (h5 : 7 = young_sample_count)
  : sample_size = 15 := 
sorry

end sample_size_is_fifteen_l145_145656


namespace problem_solution_l145_145267

noncomputable def circle_constant : ℝ := Real.pi
noncomputable def natural_base : ℝ := Real.exp 1

theorem problem_solution (π : ℝ) (e : ℝ) (h₁ : π = Real.pi) (h₂ : e = Real.exp 1) :
  π * Real.log e / Real.log 3 > 3 * Real.log e / Real.log π := by
  sorry

end problem_solution_l145_145267


namespace hole_digging_problem_l145_145276

theorem hole_digging_problem
  (total_distance : ℕ)
  (original_interval : ℕ)
  (new_interval : ℕ)
  (original_holes : ℕ)
  (new_holes : ℕ)
  (lcm_interval : ℕ)
  (common_holes : ℕ)
  (new_holes_to_be_dug : ℕ)
  (original_holes_discarded : ℕ)
  (h1 : total_distance = 3000)
  (h2 : original_interval = 50)
  (h3 : new_interval = 60)
  (h4 : original_holes = total_distance / original_interval + 1)
  (h5 : new_holes = total_distance / new_interval + 1)
  (h6 : lcm_interval = Nat.lcm original_interval new_interval)
  (h7 : common_holes = total_distance / lcm_interval + 1)
  (h8 : new_holes_to_be_dug = new_holes - common_holes)
  (h9 : original_holes_discarded = original_holes - common_holes) :
  new_holes_to_be_dug = 40 ∧ original_holes_discarded = 50 :=
sorry

end hole_digging_problem_l145_145276


namespace volume_of_inscribed_cube_l145_145581

theorem volume_of_inscribed_cube (S : ℝ) (π : ℝ) (V : ℝ) (r : ℝ) (s : ℝ) :
    S = 12 * π → 4 * π * r^2 = 12 * π → s = 2 * r → V = s^3 → V = 8 :=
by
  sorry

end volume_of_inscribed_cube_l145_145581


namespace no_determinable_cost_of_2_pans_l145_145091

def pots_and_pans_problem : Prop :=
  ∀ (P Q : ℕ), 3 * P + 4 * Q = 100 → ¬∃ Q_cost : ℕ, Q_cost = 2 * Q

theorem no_determinable_cost_of_2_pans : pots_and_pans_problem :=
by
  sorry

end no_determinable_cost_of_2_pans_l145_145091


namespace sequence_ninth_term_l145_145572

theorem sequence_ninth_term (a b : ℚ) :
  ∀ n : ℕ, n = 9 → (-1 : ℚ) ^ n * (n * b ^ n) / ((n + 1) * a ^ (n + 2)) = -9 * b^9 / (10 * a^11) :=
by
  sorry

end sequence_ninth_term_l145_145572


namespace least_number_divisible_by_12_leaves_remainder_4_is_40_l145_145539

theorem least_number_divisible_by_12_leaves_remainder_4_is_40 :
  ∃ n : ℕ, (∀ k : ℕ, n = 12 * k + 4) ∧ (∀ m : ℕ, (∀ k : ℕ, m = 12 * k + 4) → n ≤ m) ∧ n = 40 :=
by
  sorry

end least_number_divisible_by_12_leaves_remainder_4_is_40_l145_145539


namespace sin_315_equals_minus_sqrt2_div_2_l145_145672

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l145_145672


namespace find_a_given_solution_set_l145_145157

theorem find_a_given_solution_set :
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 ↔ x^2 + a * x + 6 ≤ 0) → a = -5 :=
by
  sorry

end find_a_given_solution_set_l145_145157


namespace infinite_series_sum_l145_145131

theorem infinite_series_sum :
  (∑' n : ℕ, (n + 1) / 4^(n + 1)) + (∑' n : ℕ, 1 / 2^(n + 1)) = 13 / 9 := 
sorry

end infinite_series_sum_l145_145131


namespace parker_added_dumbbells_l145_145288

def initial_dumbbells : Nat := 4
def weight_per_dumbbell : Nat := 20
def total_weight_used : Nat := 120

theorem parker_added_dumbbells :
  (total_weight_used - (initial_dumbbells * weight_per_dumbbell)) / weight_per_dumbbell = 2 := by
  sorry

end parker_added_dumbbells_l145_145288


namespace mushroom_collection_l145_145309

variable (a b v g : ℕ)

theorem mushroom_collection : 
  (a / 2 + 2 * b = v + g) ∧ (a + b = v / 2 + 2 * g) → (v = 2 * b) ∧ (a = 2 * g) :=
by
  sorry

end mushroom_collection_l145_145309


namespace fraction_inequality_l145_145921

theorem fraction_inequality (a b c : ℝ) : 
  (a / (a + 2 * b + c)) + (b / (a + b + 2 * c)) + (c / (2 * a + b + c)) ≥ 3 / 4 := 
by
  sorry

end fraction_inequality_l145_145921


namespace sum_remainder_l145_145482

theorem sum_remainder (a b c d : ℤ) (h1 : a % 53 = 33) (h2 : b % 53 = 11) 
                       (h3 : c % 53 = 49) (h4 : d % 53 = 2) :
  (a + b + c + d) % 53 = 42 :=
sorry

end sum_remainder_l145_145482


namespace compound_analysis_l145_145726

noncomputable def molecular_weight : ℝ := 18
noncomputable def atomic_weight_nitrogen : ℝ := 14.01
noncomputable def atomic_weight_hydrogen : ℝ := 1.01

theorem compound_analysis :
  ∃ (n : ℕ) (element : String), element = "hydrogen" ∧ n = 4 ∧
  (∃ remaining_weight : ℝ, remaining_weight = molecular_weight - atomic_weight_nitrogen ∧
   ∃ k, remaining_weight / atomic_weight_hydrogen = k ∧ k = n) :=
by
  sorry

end compound_analysis_l145_145726


namespace vertical_asymptotes_sum_l145_145146

theorem vertical_asymptotes_sum : 
  (∀ x : ℝ, 4 * x^2 + 7 * x + 3 = 0 → x = -3 / 4 ∨ x = -1) →
  (-3 / 4) + (-1) = -7 / 4 :=
by
  intro h
  sorry

end vertical_asymptotes_sum_l145_145146


namespace polynomial_coefficients_sum_l145_145865

theorem polynomial_coefficients_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (2*x - 3)^5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 160 :=
by
  sorry

end polynomial_coefficients_sum_l145_145865


namespace union_of_M_and_N_l145_145523

namespace SetOperations

def M : Set ℕ := {1, 2, 4}
def N : Set ℕ := {1, 3, 4}

theorem union_of_M_and_N :
  M ∪ N = {1, 2, 3, 4} :=
sorry

end SetOperations

end union_of_M_and_N_l145_145523


namespace triangle_side_lengths_l145_145365

theorem triangle_side_lengths (A B C : ℝ) (a b c : ℝ) 
  (hcosA : Real.cos A = 1/4)
  (ha : a = 4)
  (hbc_sum : b + c = 6)
  (hbc_order : b < c) :
  b = 2 ∧ c = 4 := by
  sorry

end triangle_side_lengths_l145_145365


namespace solve_for_y_l145_145987

theorem solve_for_y (y : ℤ) : (2 / 3 - 3 / 5 : ℚ) = 5 / y → y = 75 :=
by
  sorry

end solve_for_y_l145_145987


namespace conditional_probability_l145_145948

variables (A B : Prop)
variables (P : Prop → ℚ)
variables (h₁ : P A = 8 / 30) (h₂ : P (A ∧ B) = 7 / 30)

theorem conditional_probability : P (A → B) = 7 / 8 :=
by sorry

end conditional_probability_l145_145948


namespace hike_up_time_eq_l145_145531

variable (t : ℝ)
variable (h_rate_up : ℝ := 4)
variable (h_rate_down : ℝ := 6)
variable (total_time : ℝ := 3)

theorem hike_up_time_eq (h_rate_up_eq : h_rate_up = 4) 
                        (h_rate_down_eq : h_rate_down = 6) 
                        (total_time_eq : total_time = 3) 
                        (dist_eq : h_rate_up * t = h_rate_down * (total_time - t)) :
  t = 9 / 5 := by
  sorry

end hike_up_time_eq_l145_145531


namespace maciek_total_cost_l145_145894

-- Define the cost of pretzels and the additional cost percentage for chips
def cost_pretzel : ℝ := 4
def cost_chip := cost_pretzel + (cost_pretzel * 0.75)

-- Number of packets Maciek bought for pretzels and chips
def num_pretzels : ℕ := 2
def num_chips : ℕ := 2

-- Total cost calculation
def total_cost := (cost_pretzel * num_pretzels) + (cost_chip * num_chips)

-- The final theorem statement
theorem maciek_total_cost :
  total_cost = 22 := by
  sorry

end maciek_total_cost_l145_145894


namespace total_spent_on_computer_l145_145250

def initial_cost_of_pc : ℕ := 1200
def sale_price_old_card : ℕ := 300
def cost_new_card : ℕ := 500

theorem total_spent_on_computer : 
  (initial_cost_of_pc + (cost_new_card - sale_price_old_card)) = 1400 :=
by
  sorry

end total_spent_on_computer_l145_145250


namespace part1_part2_l145_145735

-- Part (1)
theorem part1 (m n : ℝ) (hm : |m| = 1) (hn : |n| = 4) (opposite : m * n < 0) :
  m + n = -3 ∨ m + n = 3 :=
sorry

-- Part (2)
theorem part2 (m n : ℝ) (hm : |m| = 1) (hn : |n| = 4) :
  (m - n) ≤ 5 :=
sorry

end part1_part2_l145_145735


namespace one_leg_divisible_by_3_l145_145108

theorem one_leg_divisible_by_3 (a b c : ℕ) (h : a^2 + b^2 = c^2) : (3 ∣ a) ∨ (3 ∣ b) :=
by sorry

end one_leg_divisible_by_3_l145_145108


namespace solve_fraction_l145_145909

theorem solve_fraction (x : ℚ) : (x^2 + 3*x + 5) / (x + 6) = x + 7 ↔ x = -37 / 10 :=
by
  sorry

end solve_fraction_l145_145909


namespace vehicles_sent_l145_145841

theorem vehicles_sent (x y : ℕ) (h1 : x + y < 18) (h2 : y < 2 * x) (h3 : x + 4 < y) :
  x = 6 ∧ y = 11 := by
  sorry

end vehicles_sent_l145_145841


namespace min_disks_required_l145_145999

/-- A structure to hold information about the file storage problem -/
structure FileStorageConditions where
  total_files : ℕ
  disk_capacity : ℝ
  num_files_1_6MB : ℕ
  num_files_1MB : ℕ
  num_files_0_5MB : ℕ

/-- Define specific conditions given in the problem -/
def storage_conditions : FileStorageConditions := {
  total_files := 42,
  disk_capacity := 2.88,
  num_files_1_6MB := 8,
  num_files_1MB := 16,
  num_files_0_5MB := 18 -- Derived from total_files - num_files_1_6MB - num_files_1MB
}

/-- Theorem stating the minimum number of disks required to store all files is 16 -/
theorem min_disks_required (c : FileStorageConditions)
  (h1 : c.total_files = 42)
  (h2 : c.disk_capacity = 2.88)
  (h3 : c.num_files_1_6MB = 8)
  (h4 : c.num_files_1MB = 16)
  (h5 : c.num_files_0_5MB = 18) :
  ∃ n : ℕ, n = 16 := by
  sorry

end min_disks_required_l145_145999


namespace palabras_bookstore_workers_l145_145664

theorem palabras_bookstore_workers (W : ℕ) (h1 : W / 2 = (W / 2)) (h2 : W / 6 = (W / 6)) (h3 : 12 = 12) (h4 : W - (W / 2 + W / 6 - 12 + 1) = 35) : W = 210 := 
sorry

end palabras_bookstore_workers_l145_145664


namespace find_z_l145_145960

theorem find_z (x y z : ℝ) 
  (h1 : y = 2 * x + 3) 
  (h2 : x + 1 / x = 3.5 + (Real.sin (z * Real.exp (-z)))) :
  z = x^2 + 1 / x^2 := 
sorry

end find_z_l145_145960


namespace angle_between_a_and_b_is_2pi_over_3_l145_145228

open Real

variables (a b c : ℝ × ℝ)

-- Given conditions
def condition1 := a.1^2 + a.2^2 = 2  -- |a| = sqrt(2)
def condition2 := b = (-1, 1)        -- b = (-1, 1)
def condition3 := c = (2, -2)        -- c = (2, -2)
def condition4 := a.1 * (b.1 + c.1) + a.2 * (b.2 + c.2) = 1  -- a · (b + c) = 1

-- Prove the angle θ between a and b is 2π/3
theorem angle_between_a_and_b_is_2pi_over_3 :
  condition1 a → condition2 b → condition3 c → condition4 a b c →
  ∃ θ, 0 ≤ θ ∧ θ ≤ π ∧ cos θ = -(1/2) ∧ θ = 2 * π / 3 :=
by
  sorry

end angle_between_a_and_b_is_2pi_over_3_l145_145228


namespace system_of_equations_solution_l145_145691

theorem system_of_equations_solution
  (a b c d e f g : ℝ)
  (x y z : ℝ)
  (h1 : a * x = b * y)
  (h2 : b * y = c * z)
  (h3 : d * x + e * y + f * z = g) :
  (x = g * b * c / (d * b * c + e * a * c + f * a * b)) ∧
  (y = g * a * c / (d * b * c + e * a * c + f * a * b)) ∧
  (z = g * a * b / (d * b * c + e * a * c + f * a * b)) :=
by
  sorry

end system_of_equations_solution_l145_145691


namespace height_of_fourth_person_l145_145950

theorem height_of_fourth_person
  (h : ℝ)
  (cond : (h + (h + 2) + (h + 4) + (h + 10)) / 4 = 79) :
  (h + 10) = 85 :=
by 
  sorry

end height_of_fourth_person_l145_145950


namespace seats_shortage_l145_145546

-- Definitions of the conditions
def children := 52
def adults := 29
def seniors := 15
def pets := 3
def total_seats := 95

-- Theorem statement to prove the number of people and pets without seats
theorem seats_shortage : children + adults + seniors + pets - total_seats = 4 :=
by
  sorry

end seats_shortage_l145_145546


namespace complement_union_A_B_complement_A_intersection_B_l145_145402

open Set

-- Definitions of A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- Proving the complement of A ∪ B
theorem complement_union_A_B : (A ∪ B)ᶜ = {x : ℝ | x ≤ 2 ∨ 10 ≤ x} :=
by sorry

-- Proving the intersection of the complement of A with B
theorem complement_A_intersection_B : (Aᶜ ∩ B) = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} :=
by sorry

end complement_union_A_B_complement_A_intersection_B_l145_145402


namespace possible_values_of_a_l145_145992

def A (a : ℝ) : Set ℝ := { x | 0 < x ∧ x < a }
def B : Set ℝ := { x | 1 < x ∧ x < 2 }
def complement_R (s : Set ℝ) : Set ℝ := { x | x ∉ s }

theorem possible_values_of_a (a : ℝ) :
  (∃ x, x ∈ A a) →
  B ⊆ complement_R (A a) →
  0 < a ∧ a ≤ 1 :=
by 
  sorry

end possible_values_of_a_l145_145992


namespace frac_series_simplification_l145_145802

theorem frac_series_simplification :
  (1 - 2 + 4 - 8 + 16 - 32 + 64 - 128 : ℚ) / (1^2 + 2^2 - 4^2 + 8^2 + 16^2 - 32^2 + 64^2 - 128^2 : ℚ) = 1 / 113 := 
by
  sorry

end frac_series_simplification_l145_145802


namespace number_of_biscuits_l145_145613

theorem number_of_biscuits (dough_length dough_width biscuit_length biscuit_width : ℕ)
    (h_dough : dough_length = 12) (h_dough_width : dough_width = 12)
    (h_biscuit_length : biscuit_length = 3) (h_biscuit_width : biscuit_width = 3)
    (dough_area : ℕ := dough_length * dough_width)
    (biscuit_area : ℕ := biscuit_length * biscuit_width) :
    dough_area / biscuit_area = 16 :=
by
  -- assume dough_area and biscuit_area are calculated from the given conditions
  -- dough_area = 144 and biscuit_area = 9
  sorry

end number_of_biscuits_l145_145613


namespace largest_prime_factor_8250_l145_145070

-- Define a function to check if a number is prime (using an existing library function)
def is_prime (n: ℕ) : Prop := Nat.Prime n

-- Define the given problem statement as a Lean theorem
theorem largest_prime_factor_8250 :
  ∃ p, is_prime p ∧ p ∣ 8250 ∧ 
    ∀ q, is_prime q ∧ q ∣ 8250 → q ≤ p :=
sorry -- The proof will be filled in later

end largest_prime_factor_8250_l145_145070


namespace polynomial_transformable_l145_145307

theorem polynomial_transformable (a b c d : ℝ) :
  (∃ A B : ℝ, ∀ z : ℝ, z^4 + A * z^2 + B = (z + a/4)^4 + a * (z + a/4)^3 + b * (z + a/4)^2 + c * (z + a/4) + d) ↔ a^3 - 4 * a * b + 8 * c = 0 :=
by
  sorry

end polynomial_transformable_l145_145307


namespace solve_fraction_equation_l145_145644

theorem solve_fraction_equation (x : ℝ) (h : x ≠ 1) : (3 * x - 1) / (4 * x - 4) = 2 / 3 → x = -5 :=
by
  intro h_eq
  sorry

end solve_fraction_equation_l145_145644


namespace negation_proof_l145_145638

theorem negation_proof (a b : ℝ) (h : a^2 + b^2 = 0) : ¬(a = 0 ∧ b = 0) :=
sorry

end negation_proof_l145_145638


namespace function_at_neg_one_zero_l145_145334

-- Define the function f with the given conditions
variable {f : ℝ → ℝ}

-- Declare the conditions as hypotheses
def domain_condition : ∀ x : ℝ, true := by sorry
def non_zero_condition : ∃ x : ℝ, f x ≠ 0 := by sorry
def even_function_condition : ∀ x : ℝ, f (x + 2) = f (2 - x) := by sorry
def odd_function_condition : ∀ x : ℝ, f (1 - 2 * x) = -f (2 * x + 1) := by sorry

-- The main theorem to be proved
theorem function_at_neg_one_zero :
  f (-1) = 0 :=
by
  -- Use the conditions to derive the result
  sorry

end function_at_neg_one_zero_l145_145334


namespace cos_B_find_b_l145_145515

theorem cos_B (a b c : ℝ) (h1 : 2 * b = a + c) (h2 : 7 * a = 3 * c) :
  Real.cos (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) = 11 / 14 := by
  sorry

theorem find_b (a b c : ℝ) (h1 : 2 * b = a + c) (h2 : 7 * a = 3 * c)
  (area : ℝ := 15 * Real.sqrt 3 / 4)
  (h3 : (1/2) * a * c * Real.sin (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) = area) :
  b = 5 := by
  sorry

end cos_B_find_b_l145_145515


namespace triangle_least_perimeter_l145_145997

noncomputable def least_perimeter_of_triangle : ℕ :=
  let a := 7
  let b := 17
  let c := 13
  a + b + c

theorem triangle_least_perimeter :
  let a := 7
  let b := 17
  let c := 13
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  4 ∣ (a^2 + b^2 + c^2) - 2 * c^2 ∧
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) →
  least_perimeter_of_triangle = 37 :=
by
  intros _ _ _ h
  sorry

end triangle_least_perimeter_l145_145997


namespace min_value_z_l145_145810

theorem min_value_z (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 1) :
  (x + 1/x) * (y + 1/y) ≥ 25/4 := 
sorry

end min_value_z_l145_145810


namespace expression_may_not_hold_l145_145920

theorem expression_may_not_hold (a b c : ℝ) (h : a = b) (hc : c = 0) :
  a = b → ¬ (a / c = b / c) := 
by
  intro hab
  intro h_div
  sorry

end expression_may_not_hold_l145_145920


namespace parallel_planes_of_skew_lines_l145_145884

variables {Plane : Type*} {Line : Type*}
variables (α β : Plane)
variables (a b : Line)

-- Conditions
def is_parallel (p1 p2 : Plane) : Prop := sorry -- Parallel planes relation
def line_in_plane (l : Line) (p : Plane) : Prop := sorry -- Line in plane relation
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry -- Line parallel to plane relation
def is_skew_lines (l1 l2 : Line) : Prop := sorry -- Skew lines relation

-- Theorem to prove
theorem parallel_planes_of_skew_lines 
  (h1 : line_in_plane a α)
  (h2 : line_in_plane b β)
  (h3 : line_parallel_plane a β)
  (h4 : line_parallel_plane b α)
  (h5 : is_skew_lines a b) :
  is_parallel α β :=
sorry

end parallel_planes_of_skew_lines_l145_145884


namespace truth_probability_l145_145488

theorem truth_probability (P_A : ℝ) (P_A_and_B : ℝ) (P_B : ℝ) 
  (hA : P_A = 0.70) (hA_and_B : P_A_and_B = 0.42) : 
  P_A * P_B = P_A_and_B → P_B = 0.6 :=
by
  sorry

end truth_probability_l145_145488


namespace augmented_wedge_volume_proof_l145_145946

open Real

noncomputable def sphere_radius (circumference : ℝ) : ℝ :=
  circumference / (2 * π)

noncomputable def sphere_volume (r : ℝ) : ℝ :=
  (4/3) * π * r^3

noncomputable def wedge_volume (volume_sphere : ℝ) (number_of_wedges : ℕ) : ℝ :=
  volume_sphere / number_of_wedges

noncomputable def augmented_wedge_volume (original_wedge_volume : ℝ) : ℝ :=
  2 * original_wedge_volume

theorem augmented_wedge_volume_proof (circumference : ℝ) (number_of_wedges : ℕ) 
  (volume : ℝ) (augmented_volume : ℝ) :
  circumference = 18 * π →
  number_of_wedges = 6 →
  volume = sphere_volume (sphere_radius circumference) →
  augmented_volume = augmented_wedge_volume (wedge_volume volume number_of_wedges) →
  augmented_volume = 324 * π :=
by
  intros h_circ h_wedges h_vol h_aug_vol
  -- This is where the proof steps would go
  sorry

end augmented_wedge_volume_proof_l145_145946


namespace digit_product_equality_l145_145093

theorem digit_product_equality (x y z : ℕ) (hx : x = 3) (hy : y = 7) (hz : z = 1) :
  x * (10 * x + y) = 111 * z :=
by
  -- Using hx, hy, and hz, the proof can proceed from here
  sorry

end digit_product_equality_l145_145093


namespace count_perfect_fourth_powers_l145_145789

theorem count_perfect_fourth_powers: 
  ∃ n_count: ℕ, n_count = 4 ∧ ∀ n: ℕ, (50 ≤ n^4 ∧ n^4 ≤ 2000) → (n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6) :=
by {
  sorry
}

end count_perfect_fourth_powers_l145_145789


namespace area_triangle_formed_by_line_l145_145100

theorem area_triangle_formed_by_line (b : ℝ) (h : (1 / 2) * |b * (-b / 2)| > 1) : b < -2 ∨ b > 2 :=
by 
  sorry

end area_triangle_formed_by_line_l145_145100


namespace original_number_is_0_02_l145_145383

theorem original_number_is_0_02 (x : ℝ) (h : 10000 * x = 4 / x) : x = 0.02 :=
by
  sorry

end original_number_is_0_02_l145_145383


namespace range_of_z_l145_145956

theorem range_of_z (x y : ℝ) (h1 : -4 ≤ x - y ∧ x - y ≤ -1) (h2 : -1 ≤ 4 * x - y ∧ 4 * x - y ≤ 5) :
  ∃ (z : ℝ), z = 9 * x - y ∧ -1 ≤ z ∧ z ≤ 20 :=
sorry

end range_of_z_l145_145956


namespace fraction_of_coins_in_decade_1800_through_1809_l145_145825

theorem fraction_of_coins_in_decade_1800_through_1809 (total_coins : ℕ) (coins_in_decade : ℕ) (c : total_coins = 30) (d : coins_in_decade = 5) : coins_in_decade / (total_coins : ℚ) = 1 / 6 :=
by
  sorry

end fraction_of_coins_in_decade_1800_through_1809_l145_145825


namespace probability_of_D_l145_145305

theorem probability_of_D (pA pB pC pD : ℚ)
  (hA : pA = 1/4)
  (hB : pB = 1/3)
  (hC : pC = 1/6)
  (hTotal : pA + pB + pC + pD = 1) : pD = 1/4 :=
by
  have hTotal_before_D : pD = 1 - (pA + pB + pC) := by sorry
  sorry

end probability_of_D_l145_145305


namespace union_set_subset_range_intersection_empty_l145_145685

-- Define the sets A and B
def A : Set ℝ := { x | 1 < x ∧ x < 3 }
def B (m : ℝ) : Set ℝ := { x | 2 * m < x ∧ x < 1 - m }

-- Question 1: When m = -1, prove A ∪ B = { x | -2 < x < 3 }
theorem union_set (m : ℝ) (h : m = -1) : A ∪ B m = { x | -2 < x ∧ x < 3 } := by
  sorry

-- Question 2: If A ⊆ B, prove m ∈ (-∞, -2]
theorem subset_range (m : ℝ) (h : A ⊆ B m) : m ∈ Set.Iic (-2) := by
  sorry

-- Question 3: If A ∩ B = ∅, prove m ∈ [0, +∞)
theorem intersection_empty (m : ℝ) (h : A ∩ B m = ∅) : m ∈ Set.Ici 0 := by
  sorry

end union_set_subset_range_intersection_empty_l145_145685


namespace total_cups_needed_l145_145568

theorem total_cups_needed (cereal_servings : ℝ) (milk_servings : ℝ) (nuts_servings : ℝ) 
  (cereal_cups_per_serving : ℝ) (milk_cups_per_serving : ℝ) (nuts_cups_per_serving : ℝ) : 
  cereal_servings = 18.0 ∧ milk_servings = 12.0 ∧ nuts_servings = 6.0 ∧ 
  cereal_cups_per_serving = 2.0 ∧ milk_cups_per_serving = 1.5 ∧ nuts_cups_per_serving = 0.5 → 
  (cereal_servings * cereal_cups_per_serving + milk_servings * milk_cups_per_serving + 
   nuts_servings * nuts_cups_per_serving) = 57.0 :=
by
  sorry

end total_cups_needed_l145_145568


namespace negation_of_universal_to_existential_l145_145138

theorem negation_of_universal_to_existential :
  (¬(∀ x : ℝ, x^2 > 0)) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
sorry

end negation_of_universal_to_existential_l145_145138


namespace total_money_divided_l145_145451

theorem total_money_divided (A B C T : ℝ) 
    (h1 : A = (2/5) * (B + C)) 
    (h2 : B = (1/5) * (A + C)) 
    (h3 : A = 600) :
    T = A + B + C →
    T = 2100 :=
by 
  sorry

end total_money_divided_l145_145451


namespace intersection_PQ_l145_145446

def P := {x : ℝ | x < 1}
def Q := {x : ℝ | x^2 < 4}
def PQ_intersection := {x : ℝ | -2 < x ∧ x < 1}

theorem intersection_PQ : P ∩ Q = PQ_intersection := by
  sorry

end intersection_PQ_l145_145446


namespace diameter_of_lake_l145_145659

theorem diameter_of_lake (d : ℝ) (pi : ℝ) (h1 : pi = 3.14) 
  (h2 : 3.14 * d - d = 1.14) : d = 0.5327 :=
by
  sorry

end diameter_of_lake_l145_145659


namespace discount_problem_l145_145230

theorem discount_problem (m : ℝ) (h : (200 * (1 - m / 100)^2 = 162)) : m = 10 :=
sorry

end discount_problem_l145_145230


namespace find_product_l145_145620

-- Define the variables used in the problem statement
variables (A P D B E C F : Type) (AP PD BP PE CP PF : ℝ)

-- The condition given in the problem
def condition (x y z : ℝ) : Prop := 
  x + y + z = 90

-- The theorem to prove
theorem find_product (x y z : ℝ) (h : condition x y z) : 
  x * y * z = 94 :=
sorry

end find_product_l145_145620


namespace probability_of_roots_condition_l145_145873

theorem probability_of_roots_condition :
  let k := 6 -- Lower bound of the interval
  let k' := 10 -- Upper bound of the interval
  let interval_length := k' - k
  let satisfying_interval_length := (22 / 3) - 6
  -- The probability that the roots of the quadratic equation satisfy x₁ ≤ 2x₂
  (satisfying_interval_length / interval_length) = (1 / 3) := by
    sorry

end probability_of_roots_condition_l145_145873


namespace mean_of_six_numbers_l145_145993

theorem mean_of_six_numbers (sum : ℚ) (h : sum = 1/3) : (sum / 6 = 1/18) :=
by
  sorry

end mean_of_six_numbers_l145_145993


namespace honda_cars_in_city_l145_145805

variable (H N : ℕ)

theorem honda_cars_in_city (total_cars : ℕ)
                         (total_red_car_ratio : ℚ)
                         (honda_red_car_ratio : ℚ)
                         (non_honda_red_car_ratio : ℚ)
                         (total_red_cars : ℕ)
                         (h : total_cars = 9000)
                         (h1 : total_red_car_ratio = 0.6)
                         (h2 : honda_red_car_ratio = 0.9)
                         (h3 : non_honda_red_car_ratio = 0.225)
                         (h4 : total_red_cars = 5400)
                         (h5 : H + N = total_cars)
                         (h6 : honda_red_car_ratio * H + non_honda_red_car_ratio * N = total_red_cars) :
  H = 5000 := by
  -- Proof goes here
  sorry

end honda_cars_in_city_l145_145805


namespace diane_head_start_l145_145587

theorem diane_head_start (x : ℝ) :
  (100 - 11.91) / (88.09 + x) = 99.25 / 100 ->
  abs (x - 12.68) < 0.01 := 
by
  sorry

end diane_head_start_l145_145587


namespace amplitude_of_cosine_wave_l145_145145

theorem amplitude_of_cosine_wave 
  (a b c d : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_pos_d : 0 < d) 
  (h_max_min : ∀ x : ℝ, d + a = 5 ∧ d - a = 1) 
  : a = 2 :=
by
  sorry

end amplitude_of_cosine_wave_l145_145145


namespace sum_nat_numbers_l145_145439

/-- 
If S is the set of all natural numbers n such that 0 ≤ n ≤ 200, n ≡ 7 [MOD 11], 
and n ≡ 5 [MOD 7], then the sum of elements in S is 351.
-/
theorem sum_nat_numbers (S : Finset ℕ) 
  (hs : ∀ n, n ∈ S ↔ n ≤ 200 ∧ n % 11 = 7 ∧ n % 7 = 5) 
  : S.sum id = 351 := 
sorry 

end sum_nat_numbers_l145_145439


namespace area_of_triangle_l145_145742

theorem area_of_triangle:
  let line1 := λ x => 3 * x - 6
  let line2 := λ x => -2 * x + 18
  let y_axis: ℝ → ℝ := λ _ => 0
  let intersection := (4.8, line1 4.8)
  let y_intercept1 := (0, -6)
  let y_intercept2 := (0, 18)
  (1/2) * 24 * 4.8 = 57.6 := by
  sorry

end area_of_triangle_l145_145742


namespace g_at_negative_two_l145_145885

-- Function definition
def g (x : ℝ) : ℝ := 3*x^5 - 4*x^4 + 2*x^3 - 5*x^2 - x + 8

-- Theorem statement
theorem g_at_negative_two : g (-2) = -186 :=
by
  -- Proof will go here, but it is skipped with sorry
  sorry

end g_at_negative_two_l145_145885


namespace number_of_boys_l145_145863

-- Definitions for the given conditions
def total_children := 60
def happy_children := 30
def sad_children := 10
def neither_happy_nor_sad_children := 20
def total_girls := 41
def happy_boys := 6
def sad_girls := 4
def neither_happy_nor_sad_boys := 7

-- Define the total number of boys
def total_boys := total_children - total_girls

-- Proof statement
theorem number_of_boys : total_boys = 19 :=
  by
    sorry

end number_of_boys_l145_145863


namespace rectangle_perimeter_l145_145007

variable (L W : ℝ) 

theorem rectangle_perimeter (h1 : L > 4) (h2 : W > 4) (h3 : (L * W) - ((L - 4) * (W - 4)) = 168) : 
  2 * (L + W) = 92 := 
  sorry

end rectangle_perimeter_l145_145007


namespace first_term_of_geometric_series_l145_145466

theorem first_term_of_geometric_series (a r S : ℝ) (h₁ : r = 1/4) (h₂ : S = 80) (h₃ : S = a / (1 - r)) : a = 60 :=
by
  sorry

end first_term_of_geometric_series_l145_145466


namespace range_of_m_l145_145067

noncomputable def f (x : ℝ) : ℝ := -x^2
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 2^x - m

theorem range_of_m (m : ℝ) :
  (∀ x1 ∈ Set.Icc (-1 : ℝ) 3, ∃ x2 ∈ Set.Icc (0 : ℝ) 2, f x1 ≥ g x2 m) ↔ m ≥ 10 := 
by
  sorry

end range_of_m_l145_145067


namespace probability_same_color_opposite_feet_l145_145050

/-- Define the initial conditions: number of pairs of each color. -/
def num_black_pairs : ℕ := 8
def num_brown_pairs : ℕ := 4
def num_gray_pairs : ℕ := 3
def num_red_pairs : ℕ := 1

/-- The total number of shoes. -/
def total_shoes : ℕ := 2 * (num_black_pairs + num_brown_pairs + num_gray_pairs + num_red_pairs)

theorem probability_same_color_opposite_feet :
  ((num_black_pairs * (num_black_pairs - 1)) + 
   (num_brown_pairs * (num_brown_pairs - 1)) + 
   (num_gray_pairs * (num_gray_pairs - 1)) + 
   (num_red_pairs * (num_red_pairs - 1))) * 2 / (total_shoes * (total_shoes - 1)) = 45 / 248 :=
by sorry

end probability_same_color_opposite_feet_l145_145050


namespace x_cubed_plus_y_cubed_l145_145340

theorem x_cubed_plus_y_cubed (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 14) : x^3 + y^3 = 176 :=
sorry

end x_cubed_plus_y_cubed_l145_145340


namespace area_of_306090_triangle_l145_145053

-- Conditions
def is_306090_triangle (a b c : ℝ) : Prop :=
  a / b = 1 / Real.sqrt 3 ∧ a / c = 1 / 2

-- Given values
def hypotenuse : ℝ := 6

-- To prove
theorem area_of_306090_triangle :
  ∃ (a b c : ℝ), is_306090_triangle a b c ∧ c = hypotenuse ∧ (1 / 2) * a * b = (9 * Real.sqrt 3) / 2 :=
by
  sorry

end area_of_306090_triangle_l145_145053


namespace cannot_determine_x_l145_145721

theorem cannot_determine_x
  (n m : ℝ) (x : ℝ)
  (h1 : n + m = 8) 
  (h2 : n * x + m * (1/5) = 1) : true :=
by {
  sorry
}

end cannot_determine_x_l145_145721


namespace day_crew_fraction_l145_145395

theorem day_crew_fraction (D W : ℕ) (h1 : ∀ n, n = D / 4) (h2 : ∀ w, w = 4 * W / 5) :
  (D * W) / ((D * W) + ((D / 4) * (4 * W / 5))) = 5 / 6 :=
by 
  sorry

end day_crew_fraction_l145_145395


namespace paint_area_correct_l145_145096

-- Definitions for the conditions of the problem
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def door_height : ℕ := 3
def door_length : ℕ := 5

-- Define the total area of the wall (without considering the door)
def wall_area : ℕ := wall_height * wall_length

-- Define the area of the door
def door_area : ℕ := door_height * door_length

-- Define the area that needs to be painted
def area_to_paint : ℕ := wall_area - door_area

-- The proof problem: Prove that Sandy needs to paint 135 square feet
theorem paint_area_correct : area_to_paint = 135 := 
by
  -- Sorry will be replaced with an actual proof
  sorry

end paint_area_correct_l145_145096


namespace largest_hexagon_angle_l145_145787

-- We define the conditions first
def angle_ratios (x : ℝ) := [3*x, 3*x, 3*x, 4*x, 5*x, 6*x]
def sum_of_angles (angles : List ℝ) := angles.sum = 720

-- Now we state our proof goal
theorem largest_hexagon_angle :
  ∀ (x : ℝ), sum_of_angles (angle_ratios x) → 6 * x = 180 :=
by
  intro x
  intro h
  sorry

end largest_hexagon_angle_l145_145787


namespace combined_work_time_l145_145034

theorem combined_work_time (A B C D : ℕ) (hA : A = 10) (hB : B = 15) (hC : C = 20) (hD : D = 30) :
  1 / (1 / A + 1 / B + 1 / C + 1 / D) = 4 := by
  -- Replace the following "sorry" with your proof.
  sorry

end combined_work_time_l145_145034


namespace minimum_focal_chord_length_l145_145991

theorem minimum_focal_chord_length (p : ℝ) (hp : p > 0) :
  ∃ l, (l = 2 * p) ∧ (∀ y x1 x2, y^2 = 2 * p * x1 ∧ y^2 = 2 * p * x2 → l = x2 - x1) := 
sorry

end minimum_focal_chord_length_l145_145991


namespace sin_240_eq_neg_sqrt3_over_2_l145_145437

theorem sin_240_eq_neg_sqrt3_over_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_over_2_l145_145437


namespace mike_chocolate_squares_l145_145032

theorem mike_chocolate_squares (M : ℕ) (h1 : 65 = 3 * M + 5) : M = 20 :=
by {
  -- proof of the theorem (not included as per instructions)
  sorry
}

end mike_chocolate_squares_l145_145032


namespace ball_bounces_l145_145807

theorem ball_bounces (k : ℕ) :
  1500 * (2 / 3 : ℝ)^k < 2 ↔ k ≥ 19 :=
sorry

end ball_bounces_l145_145807


namespace smallest_base_power_l145_145576

theorem smallest_base_power (x y z : ℝ) (hx : 1 < x) (hy : 1 < y) (hz : 1 < z)
  (h_log_eq : Real.log x / Real.log 2 = Real.log y / Real.log 3 ∧ Real.log y / Real.log 3 = Real.log z / Real.log 5) :
  z ^ (1 / 5) < x ^ (1 / 2) ∧ z ^ (1 / 5) < y ^ (1 / 3) :=
by
  -- required proof here
  sorry

end smallest_base_power_l145_145576


namespace max_entanglements_l145_145120

theorem max_entanglements (a b : ℕ) (h1 : a < b) (h2 : a < 1000) (h3 : b < 1000) :
  ∃ n ≤ 9, ∀ k, k ≤ n → ∃ a' b' : ℕ, (b' - a' = b - a - 2^k) :=
by sorry

end max_entanglements_l145_145120


namespace pineapple_total_cost_correct_l145_145502

-- Define the conditions
def pineapple_cost : ℝ := 1.25
def num_pineapples : ℕ := 12
def shipping_cost : ℝ := 21.00

-- Calculate total cost
noncomputable def total_pineapple_cost : ℝ := pineapple_cost * num_pineapples
noncomputable def total_cost : ℝ := total_pineapple_cost + shipping_cost
noncomputable def cost_per_pineapple : ℝ := total_cost / num_pineapples

-- The proof problem
theorem pineapple_total_cost_correct : cost_per_pineapple = 3 := by
  -- The proof will be filled in here
  sorry

end pineapple_total_cost_correct_l145_145502


namespace sum_of_acute_angles_l145_145621

open Real

theorem sum_of_acute_angles (θ₁ θ₂ : ℝ)
  (h1 : 0 < θ₁ ∧ θ₁ < π / 2)
  (h2 : 0 < θ₂ ∧ θ₂ < π / 2)
  (h_eq : (sin θ₁) ^ 2020 / (cos θ₂) ^ 2018 + (cos θ₁) ^ 2020 / (sin θ₂) ^ 2018 = 1) :
  θ₁ + θ₂ = π / 2 := sorry

end sum_of_acute_angles_l145_145621


namespace samara_oil_spent_l145_145397

theorem samara_oil_spent (O : ℕ) (A_total : ℕ) (S_tires : ℕ) (S_detailing : ℕ) (diff : ℕ) (S_total : ℕ) :
  A_total = 2457 →
  S_tires = 467 →
  S_detailing = 79 →
  diff = 1886 →
  S_total = O + S_tires + S_detailing →
  A_total = S_total + diff →
  O = 25 :=
by
  sorry

end samara_oil_spent_l145_145397


namespace reflection_line_slope_intercept_l145_145945

theorem reflection_line_slope_intercept (m b : ℝ) :
  let P1 := (2, 3)
  let P2 := (10, 7)
  let midpoint := ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)
  midpoint = (6, 5) ∧
  ∃(m b : ℝ), 
    m = -2 ∧
    b = 17 ∧
    P2 = (2 * midpoint.1 - P1.1, 2 * midpoint.2 - P1.2)
→ m + b = 15 := by
  intros
  sorry

end reflection_line_slope_intercept_l145_145945


namespace part1_part2_l145_145225

theorem part1 (p : ℝ) (h : p = 2 / 5) : 
  (p^2 + 2 * (3 / 5) * p^2) = 0.352 :=
by 
  rw [h]
  sorry

theorem part2 (p : ℝ) (h : p = 2 / 5) : 
  (4 * (1 / (11.32 * p^4)) + 5 * (2.4 / (11.32 * p^4)) + 6 * (3.6 / (11.32 * p^4)) + 7 * (2.16 / (11.32 * p^4))) = 4.834 :=
by 
  rw [h]
  sorry

end part1_part2_l145_145225


namespace find_sum_of_squares_l145_145119

theorem find_sum_of_squares (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : x * y + x + y = 119) (h2 : x^2 * y + x * y^2 = 1680) :
  x^2 + y^2 = 1057 := 
by
  sorry

end find_sum_of_squares_l145_145119


namespace sandy_paid_for_pants_l145_145555

-- Define the costs and change as constants
def cost_of_shirt : ℝ := 8.25
def amount_paid_with : ℝ := 20.00
def change_received : ℝ := 2.51

-- Define the amount paid for pants
def amount_paid_for_pants : ℝ := 9.24

-- The theorem stating the problem
theorem sandy_paid_for_pants : 
  amount_paid_with - (cost_of_shirt + change_received) = amount_paid_for_pants := 
by 
  -- proof is required here
  sorry

end sandy_paid_for_pants_l145_145555


namespace largest_common_number_in_range_l145_145740

theorem largest_common_number_in_range (n1 d1 n2 d2 : ℕ) (h1 : n1 = 2) (h2 : d1 = 4) (h3 : n2 = 5) (h4 : d2 = 6) :
  ∃ k : ℕ, k ≤ 200 ∧ (∀ n3 : ℕ, n3 = n1 + d1 * k) ∧ (∀ n4 : ℕ, n4 = n2 + d2 * k) ∧ n3 = 190 ∧ n4 = 190 := 
by {
  sorry
}

end largest_common_number_in_range_l145_145740


namespace range_of_a_iff_l145_145455

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, |x| + |x - 1| ≤ a → a ≥ 1

theorem range_of_a_iff (a : ℝ) :
  (∃ x : ℝ, |x| + |x - 1| ≤ a) ↔ (a ≥ 1) :=
by sorry

end range_of_a_iff_l145_145455


namespace complement_of_A_l145_145199

-- Definition of the universal set U and the set A
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | (x - 1) * (x + 2) > 0}

-- Theorem statement for the complement of A in U
theorem complement_of_A:
  (U \ A) = {x | -2 ≤ x ∧ x ≤ 1} :=
  sorry

end complement_of_A_l145_145199


namespace kendall_tau_correct_l145_145869

-- Base Lean setup and list of dependencies might go here

structure TestScores :=
  (A : List ℚ)
  (B : List ℚ)

-- Constants from the problem
def scores : TestScores :=
  { A := [95, 90, 86, 84, 75, 70, 62, 60, 57, 50]
  , B := [92, 93, 83, 80, 55, 60, 45, 72, 62, 70] }

-- Function to calculate the Kendall rank correlation coefficient
noncomputable def kendall_tau (scores : TestScores) : ℚ :=
  -- the method of calculating Kendall tau could be very complex
  -- hence we assume the correct coefficient directly for the example
  0.51

-- The proof problem
theorem kendall_tau_correct : kendall_tau scores = 0.51 :=
by
  sorry

end kendall_tau_correct_l145_145869


namespace daria_multiple_pizzas_l145_145877

variable (m : ℝ)
variable (don_pizzas : ℝ) (total_pizzas : ℝ)

axiom don_pizzas_def : don_pizzas = 80
axiom total_pizzas_def : total_pizzas = 280

theorem daria_multiple_pizzas (m : ℝ) (don_pizzas : ℝ) (total_pizzas : ℝ) 
    (h1 : don_pizzas = 80) (h2 : total_pizzas = 280) 
    (h3 : total_pizzas = don_pizzas + m * don_pizzas) : 
    m = 2.5 :=
by sorry

end daria_multiple_pizzas_l145_145877


namespace value_of_expression_l145_145266

theorem value_of_expression (a b c : ℝ) (h : a * (-2)^5 + b * (-2)^3 + c * (-2) - 5 = 7) :
  a * 2^5 + b * 2^3 + c * 2 - 5 = -17 :=
by sorry

end value_of_expression_l145_145266


namespace equal_powers_equal_elements_l145_145306

theorem equal_powers_equal_elements
  (a : Fin 17 → ℕ)
  (h : ∀ i : Fin 17, a i ^ a (i + 1) % 17 = a ((i + 1) % 17) ^ a ((i + 2) % 17) % 17)
  : ∀ i j : Fin 17, a i = a j :=
by
  sorry

end equal_powers_equal_elements_l145_145306


namespace non_empty_subsets_count_l145_145785

def odd_set : Finset ℕ := {1, 3, 5, 7, 9}
def even_set : Finset ℕ := {2, 4, 6, 8}

noncomputable def num_non_empty_subsets_odd : ℕ := 2 ^ odd_set.card - 1
noncomputable def num_non_empty_subsets_even : ℕ := 2 ^ even_set.card - 1

theorem non_empty_subsets_count :
  num_non_empty_subsets_odd + num_non_empty_subsets_even = 46 :=
by sorry

end non_empty_subsets_count_l145_145785


namespace weight_of_gravel_l145_145251

theorem weight_of_gravel (total_weight : ℝ) (weight_sand : ℝ) (weight_water : ℝ) (weight_gravel : ℝ) 
  (h1 : total_weight = 48)
  (h2 : weight_sand = (1/3) * total_weight)
  (h3 : weight_water = (1/2) * total_weight)
  (h4 : weight_gravel = total_weight - (weight_sand + weight_water)) :
  weight_gravel = 8 :=
sorry

end weight_of_gravel_l145_145251


namespace apples_more_than_oranges_l145_145283

-- Definitions based on conditions
def total_fruits : ℕ := 301
def apples : ℕ := 164

-- Statement to prove
theorem apples_more_than_oranges : (apples - (total_fruits - apples)) = 27 :=
by
  sorry

end apples_more_than_oranges_l145_145283


namespace longest_playing_time_l145_145059

theorem longest_playing_time (total_playtime : ℕ) (n : ℕ) (k : ℕ) (standard_time : ℚ) (long_time : ℚ) :
  total_playtime = 120 ∧ n = 6 ∧ k = 2 ∧ long_time = k * standard_time →
  5 * standard_time + long_time = 240 →
  long_time = 68 :=
by
  sorry

end longest_playing_time_l145_145059


namespace max_area_of_triangle_l145_145532

-- Define the problem conditions and the maximum area S
theorem max_area_of_triangle
  (A B C : ℝ)
  (a b c S : ℝ)
  (h1 : 4 * S = a^2 - (b - c)^2)
  (h2 : b + c = 8) :
  S ≤ 8 :=
sorry

end max_area_of_triangle_l145_145532


namespace store_loss_l145_145072

theorem store_loss (x y : ℝ) (hx : x + 0.25 * x = 135) (hy : y - 0.25 * y = 135) : 
  (135 * 2) - (x + y) = -18 := 
by
  sorry

end store_loss_l145_145072


namespace sum_of_numbers_l145_145561

noncomputable def sum_two_numbers (x y : ℝ) : ℝ :=
  x + y

theorem sum_of_numbers (x y : ℝ) (h1 : x * y = 16) (h2 : (1 / x) = 3 * (1 / y)) : 
  sum_two_numbers x y = (16 * Real.sqrt 3) / 3 := 
by 
  sorry

end sum_of_numbers_l145_145561


namespace hexagon_perimeter_l145_145408

theorem hexagon_perimeter (side_length : ℕ) (num_sides : ℕ) (h1 : side_length = 5) (h2 : num_sides = 6) : 
  num_sides * side_length = 30 := by
  sorry

end hexagon_perimeter_l145_145408


namespace unique_solution_triple_l145_145536

theorem unique_solution_triple (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (xy / z : ℚ) + (yz / x) + (zx / y) = 3 → (x = 1 ∧ y = 1 ∧ z = 1) := 
by 
  sorry

end unique_solution_triple_l145_145536


namespace fraction_not_exist_implies_x_neg_one_l145_145861

theorem fraction_not_exist_implies_x_neg_one {x : ℝ} :
  ¬(∃ y : ℝ, y = 1 / (x + 1)) → x = -1 :=
by
  intro h
  have : x + 1 = 0 :=
    by
      contrapose! h
      exact ⟨1 / (x + 1), rfl⟩
  linarith

end fraction_not_exist_implies_x_neg_one_l145_145861


namespace sequence_98th_term_l145_145982

-- Definitions of the rules
def rule1 (n : ℕ) : ℕ := n * 9
def rule2 (n : ℕ) : ℕ := n / 2
def rule3 (n : ℕ) : ℕ := n - 5

-- Function to compute the next term in the sequence based on the current term
def next_term (n : ℕ) : ℕ :=
  if n < 10 then rule1 n
  else if n % 2 = 0 then rule2 n
  else rule3 n

-- Function to compute the nth term of the sequence starting with the initial term
def nth_term (start : ℕ) (n : ℕ) : ℕ :=
  Nat.iterate next_term n start

-- Theorem to prove that the 98th term of the sequence starting at 98 is 27
theorem sequence_98th_term : nth_term 98 98 = 27 := by
  sorry

end sequence_98th_term_l145_145982


namespace min_Sn_value_l145_145373

noncomputable def a (n : ℕ) (d : ℤ) : ℤ := -11 + (n - 1) * d

def Sn (n : ℕ) (d : ℤ) : ℤ := n * -11 + n * (n - 1) * d / 2

theorem min_Sn_value {d : ℤ} (h5_6 : a 5 d + a 6 d = -4) : 
  ∃ n, Sn n d = (n - 6)^2 - 36 ∧ n = 6 :=
by
  sorry

end min_Sn_value_l145_145373


namespace function_passes_through_point_l145_145509

noncomputable def func_graph (a : ℝ) (x : ℝ) : ℝ := a ^ (x - 1) + 2

theorem function_passes_through_point (a : ℝ) (h0 : a > 0) (h1 : a ≠ 1) :
  func_graph a 1 = 3 :=
by
  -- Proof logic is omitted
  sorry

end function_passes_through_point_l145_145509


namespace other_girl_age_l145_145790

theorem other_girl_age (x : ℕ) (h1 : 13 + x = 27) : x = 14 := by
  sorry

end other_girl_age_l145_145790


namespace solution_set_of_inequality_l145_145822

theorem solution_set_of_inequality (x : ℝ) : x^2 < -2 * x + 15 ↔ -5 < x ∧ x < 3 := 
sorry

end solution_set_of_inequality_l145_145822


namespace ratio_of_speeds_l145_145153

theorem ratio_of_speeds (D H : ℕ) (h1 : D = 2 * H) (h2 : 10 * D = 20 * H) :
  10 * D = 2 * (10 * H) :=
by
  sorry

example (D H : ℕ) (h1 : D = 2 * H) (h2 : 10 * D = 20 * H) :
  10 = 10 :=
by
  sorry

end ratio_of_speeds_l145_145153


namespace satisfying_integers_l145_145057

theorem satisfying_integers (a b : ℤ) :
  a^4 + (a + b)^4 + b^4 = x^2 → a = 0 ∧ b = 0 :=
by
  -- Proof is required to be filled in here.
  sorry

end satisfying_integers_l145_145057


namespace find_number_l145_145015

-- Definitions used in the given problem conditions
def condition (x : ℝ) : Prop := (3.242 * x) / 100 = 0.04863

-- Statement of the problem
theorem find_number (x : ℝ) (h : condition x) : x = 1.5 :=
by
  sorry
 
end find_number_l145_145015


namespace points_not_all_odd_distance_l145_145876

open Real

theorem points_not_all_odd_distance (p : Fin 4 → ℝ × ℝ) : ∃ i j : Fin 4, i ≠ j ∧ ¬ Odd (dist (p i) (p j)) := 
by
  sorry

end points_not_all_odd_distance_l145_145876


namespace additional_machines_needed_l145_145087

theorem additional_machines_needed
  (machines : ℕ)
  (days : ℕ)
  (one_fourth_less_days : ℕ)
  (machine_days_total : ℕ)
  (machines_needed : ℕ)
  (additional_machines : ℕ) 
  (h1 : machines = 15) 
  (h2 : days = 36)
  (h3 : one_fourth_less_days = 27)
  (h4 : machine_days_total = machines * days)
  (h5 : machines_needed = machine_days_total / one_fourth_less_days) :
  additional_machines = machines_needed - machines → additional_machines = 5 :=
by
  admit -- sorry

end additional_machines_needed_l145_145087


namespace distinct_remainders_sum_quotient_l145_145547

theorem distinct_remainders_sum_quotient :
  let sq_mod_7 (n : Nat) := (n * n) % 7
  let distinct_remainders := List.eraseDup ([sq_mod_7 1, sq_mod_7 2, sq_mod_7 3, sq_mod_7 4, sq_mod_7 5])
  let s := List.sum distinct_remainders
  s / 7 = 1 :=
by
  sorry

end distinct_remainders_sum_quotient_l145_145547


namespace max_radius_of_circle_in_triangle_inscribed_l145_145083

theorem max_radius_of_circle_in_triangle_inscribed (ω : Set (ℝ × ℝ)) (hω : ∀ (P : ℝ × ℝ), P ∈ ω → P.1^2 + P.2^2 = 1)
  (O : ℝ × ℝ) (hO : O = (0, 0)) (P : ℝ × ℝ) (hP : P ∈ ω) (A : ℝ × ℝ) 
  (hA : A = (P.1, 0)) : 
  (∃ r : ℝ, r = (Real.sqrt 2 - 1) / 2) :=
by
  sorry

end max_radius_of_circle_in_triangle_inscribed_l145_145083


namespace probability_same_group_l145_145005

noncomputable def calcProbability : ℚ := 
  let totalOutcomes := 18 * 17
  let favorableCase1 := 6 * 5
  let favorableCase2 := 4 * 3
  let totalFavorableOutcomes := favorableCase1 + favorableCase2
  totalFavorableOutcomes / totalOutcomes

theorem probability_same_group (cards : Finset ℕ) (draws : Finset ℕ) (number1 number2 : ℕ) (condition_cardinality : cards.card = 20) 
  (condition_draws : draws.card = 4) (condition_numbers : number1 = 5 ∧ number2 = 14 ∧ number1 ∈ cards ∧ number2 ∈ cards) 
  : calcProbability = 7 / 51 :=
sorry

end probability_same_group_l145_145005


namespace max_blocks_l145_145544

theorem max_blocks (box_height box_width box_length : ℝ) 
  (typeA_height typeA_width typeA_length typeB_height typeB_width typeB_length : ℝ) 
  (h_box : box_height = 8) (w_box : box_width = 10) (l_box : box_length = 12) 
  (h_typeA : typeA_height = 3) (w_typeA : typeA_width = 2) (l_typeA : typeA_length = 4) 
  (h_typeB : typeB_height = 4) (w_typeB : typeB_width = 3) (l_typeB : typeB_length = 5) : 
  max (⌊box_height / typeA_height⌋ * ⌊box_width / typeA_width⌋ * ⌊box_length / typeA_length⌋)
      (⌊box_height / typeB_height⌋ * ⌊box_width / typeB_width⌋ * ⌊box_length / typeB_length⌋) = 30 := 
  by
  sorry

end max_blocks_l145_145544


namespace cookie_contest_l145_145415

theorem cookie_contest (A B : ℚ) (hA : A = 5/6) (hB : B = 2/3) :
  A - B = 1/6 :=
by 
  sorry

end cookie_contest_l145_145415


namespace find_reflection_line_l145_145359

-- Definition of the original and reflected vertices
structure Point :=
  (x : ℝ)
  (y : ℝ)

def D : Point := {x := 1, y := 2}
def E : Point := {x := 6, y := 7}
def F : Point := {x := -5, y := 5}
def D' : Point := {x := 1, y := -4}
def E' : Point := {x := 6, y := -9}
def F' : Point := {x := -5, y := -7}

theorem find_reflection_line (M : ℝ) :
  (D.y + D'.y) / 2 = M ∧ (E.y + E'.y) / 2 = M ∧ (F.y + F'.y) / 2 = M → M = -1 :=
by
  intros
  sorry

end find_reflection_line_l145_145359


namespace monotonic_increasing_intervals_l145_145696

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 + 2*x + 1)
noncomputable def f' (x : ℝ) : ℝ := Real.exp x * (x^2 + 4*x + 3)

theorem monotonic_increasing_intervals :
  ∀ x, f' x > 0 ↔ (x < -3 ∨ x > -1) :=
by
  intro x
  -- proof omitted
  sorry

end monotonic_increasing_intervals_l145_145696


namespace sum_of_roots_of_equation_l145_145989

theorem sum_of_roots_of_equation : 
  (∀ x, 5 = (x^3 - 2*x^2 - 8*x) / (x + 2)) → 
  (∃ x1 x2, (5 = x1) ∧ (5 = x2) ∧ (x1 + x2 = 4)) := 
by
  sorry

end sum_of_roots_of_equation_l145_145989


namespace least_trees_l145_145953

theorem least_trees (N : ℕ) (h1 : N % 7 = 0) (h2 : N % 6 = 0) (h3 : N % 4 = 0) (h4 : N ≥ 100) : N = 168 :=
sorry

end least_trees_l145_145953


namespace zach_cookies_left_l145_145457

/- Defining the initial conditions on cookies baked each day -/
def cookies_monday : ℕ := 32
def cookies_tuesday : ℕ := cookies_monday / 2
def cookies_wednesday : ℕ := 3 * cookies_tuesday - 4 - 3
def cookies_thursday : ℕ := 2 * cookies_monday - 10 + 5
def cookies_friday : ℕ := cookies_wednesday - 6 - 4
def cookies_saturday : ℕ := cookies_monday + cookies_friday - 10

/- Aggregating total cookies baked throughout the week -/
def total_baked : ℕ := cookies_monday + cookies_tuesday + cookies_wednesday +
                      cookies_thursday + cookies_friday + cookies_saturday

/- Defining cookies lost each day -/
def daily_parents_eat : ℕ := 2 * 6
def neighbor_friday_eat : ℕ := 8
def friends_thursday_eat : ℕ := 3 * 2

def total_lost : ℕ := 4 + 3 + 10 + 6 + 4 + 10 + daily_parents_eat + neighbor_friday_eat + friends_thursday_eat

/- Calculating cookies left at end of six days -/
def cookies_left : ℕ := total_baked - total_lost

/- Proof objective -/
theorem zach_cookies_left : cookies_left = 200 := by
  sorry

end zach_cookies_left_l145_145457


namespace pratt_certificate_space_bound_l145_145780

-- Define the Pratt certificate space function λ(p)
noncomputable def pratt_space (p : ℕ) : ℝ := sorry

-- Define the log_2 function (if not already available in Mathlib)
noncomputable def log2 (x : ℝ) : ℝ := sorry

-- Assuming that p is a prime number
variable {p : ℕ} (hp : Nat.Prime p)

-- The proof problem
theorem pratt_certificate_space_bound (hp : Nat.Prime p) :
  pratt_space p ≤ 6 * (log2 p) ^ 2 := 
sorry

end pratt_certificate_space_bound_l145_145780


namespace sequence_of_arrows_from_425_to_427_l145_145479

theorem sequence_of_arrows_from_425_to_427 :
  ∀ (arrows : ℕ → ℕ), (∀ n, arrows (n + 4) = arrows n) →
  (arrows 425, arrows 426, arrows 427) = (arrows 1, arrows 2, arrows 3) :=
by
  intros arrows h_period
  have h1 : arrows 425 = arrows 1 := by 
    sorry
  have h2 : arrows 426 = arrows 2 := by 
    sorry
  have h3 : arrows 427 = arrows 3 := by 
    sorry
  sorry

end sequence_of_arrows_from_425_to_427_l145_145479


namespace confidence_relationship_l145_145671
noncomputable def K_squared : ℝ := 3.918
noncomputable def critical_value : ℝ := 3.841
noncomputable def p_val : ℝ := 0.05

theorem confidence_relationship (K_squared : ℝ) (critical_value : ℝ) (p_val : ℝ) :
  K_squared ≥ critical_value -> p_val = 0.05 ->
  1 - p_val = 0.95 :=
by
  sorry

end confidence_relationship_l145_145671


namespace root_of_unity_product_l145_145176

theorem root_of_unity_product (ω : ℂ) (h1 : ω^3 = 1) (h2 : ω ≠ 1) :
  (1 - ω + ω^2) * (1 + ω - ω^2) = 1 :=
  sorry

end root_of_unity_product_l145_145176


namespace books_on_shelf_l145_145245

-- Step definitions based on the conditions
def initial_books := 38
def marta_books_removed := 10
def tom_books_removed := 5
def tom_books_added := 12

-- Final number of books on the shelf
def final_books : ℕ := initial_books - marta_books_removed - tom_books_removed + tom_books_added

-- Theorem statement to prove the final number of books
theorem books_on_shelf : final_books = 35 :=
by 
  -- Proof for the statement goes here
  sorry

end books_on_shelf_l145_145245


namespace speed_second_half_l145_145092

theorem speed_second_half (H : ℝ) (S1 S2 : ℝ) (T : ℝ) : T = 11 → S1 = 30 → S1 * T1 = 150 → S1 * T1 + S2 * T2 = 300 → S2 = 25 :=
by
  intro hT hS1 hD1 hTotal
  sorry

end speed_second_half_l145_145092


namespace symmetrical_point_of_P_is_correct_l145_145350

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the function to get the symmetric point with respect to the origin
def symmetrical_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Prove that the symmetrical point of P with respect to the origin is (1, -2)
theorem symmetrical_point_of_P_is_correct : symmetrical_point P = (1, -2) :=
  sorry

end symmetrical_point_of_P_is_correct_l145_145350


namespace inequality_condition_l145_145650

theorem inequality_condition {x : ℝ} (h : -1/2 ≤ x ∧ x < 1) : (2 * x + 1) / (1 - x) ≥ 0 :=
sorry

end inequality_condition_l145_145650


namespace find_x3_y3_l145_145018

theorem find_x3_y3 (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 + y^2 = 18) : x^3 + y^3 = 54 := 
by 
  sorry

end find_x3_y3_l145_145018


namespace park_cycling_time_l145_145368

def length_breadth_ratio (L B : ℕ) : Prop := L / B = 1 / 3
def area_of_park (L B : ℕ) : Prop := L * B = 120000
def speed_of_cyclist : ℕ := 200 -- meters per minute
def perimeter (L B : ℕ) : ℕ := 2 * L + 2 * B
def time_to_complete_round (P v : ℕ) : ℕ := P / v

theorem park_cycling_time
  (L B : ℕ)
  (h_ratio : length_breadth_ratio L B)
  (h_area : area_of_park L B)
  : time_to_complete_round (perimeter L B) speed_of_cyclist = 8 :=
by
  sorry

end park_cycling_time_l145_145368


namespace initial_money_l145_145388

/-- Given the following conditions:
  (1) June buys 4 maths books at $20 each.
  (2) June buys 6 more science books than maths books at $10 each.
  (3) June buys twice as many art books as maths books at $20 each.
  (4) June spends $160 on music books.
  Prove that June had initially $500 for buying school supplies. -/
theorem initial_money (maths_books : ℕ) (science_books : ℕ) (art_books : ℕ) (music_books_cost : ℕ)
  (h_math_books : maths_books = 4) (price_per_math_book : ℕ) (price_per_science_book : ℕ) 
  (price_per_art_book : ℕ) (price_per_music_books_cost : ℕ) (h_maths_price : price_per_math_book = 20)
  (h_science_books : science_books = maths_books + 6) (h_science_price : price_per_science_book = 10)
  (h_art_books : art_books = 2 * maths_books) (h_art_price : price_per_art_book = 20)
  (h_music_books_cost : music_books_cost = 160) :
  4 * 20 + (4 + 6) * 10 + (2 * 4) * 20 + 160 = 500 :=
by sorry

end initial_money_l145_145388


namespace ball_hits_ground_at_time_l145_145003

theorem ball_hits_ground_at_time :
  ∀ (t : ℝ), (-18 * t^2 + 30 * t + 60 = 0) ↔ (t = (5 + Real.sqrt 145) / 6) :=
sorry

end ball_hits_ground_at_time_l145_145003


namespace determine_a_for_nonnegative_function_l145_145357

def function_positive_on_interval (a : ℝ) : Prop :=
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → a * x^3 - 3 * x + 1 ≥ 0

theorem determine_a_for_nonnegative_function :
  ∀ (a : ℝ), function_positive_on_interval a ↔ a = 4 :=
by
  sorry

end determine_a_for_nonnegative_function_l145_145357


namespace c_completes_in_three_days_l145_145754

variables (r_A r_B r_C : ℝ)
variables (h1 : r_A + r_B = 1/3)
variables (h2 : r_B + r_C = 1/3)
variables (h3 : r_A + r_C = 2/3)

theorem c_completes_in_three_days : 1 / r_C = 3 :=
by sorry

end c_completes_in_three_days_l145_145754


namespace chocolate_cost_is_3_l145_145489

-- Definitions based on the conditions
def dan_has_5_dollars : Prop := true
def cost_candy_bar : ℕ := 2
def cost_chocolate : ℕ := cost_candy_bar + 1

-- Theorem to prove
theorem chocolate_cost_is_3 : cost_chocolate = 3 :=
by {
  -- This is where the proof steps would go
  sorry
}

end chocolate_cost_is_3_l145_145489


namespace rationalize_denominator_correct_l145_145435

noncomputable def rationalize_denominator : Prop :=
  1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end rationalize_denominator_correct_l145_145435


namespace add_to_fraction_eq_l145_145616

theorem add_to_fraction_eq (n : ℤ) : (3 + n : ℚ) / (5 + n) = 5 / 6 → n = 7 := 
by
  sorry

end add_to_fraction_eq_l145_145616


namespace inequality_of_positive_numbers_l145_145529

theorem inequality_of_positive_numbers (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) : 
  (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) / (a + b + c) ≥ a * b * c :=
sorry

end inequality_of_positive_numbers_l145_145529


namespace intersection_A_B_l145_145014

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := { x | 2 ≤ x ∧ x ≤ 5 }

theorem intersection_A_B : A ∩ B = {3, 5} :=
  sorry

end intersection_A_B_l145_145014


namespace fraction_of_janes_age_is_five_eighths_l145_145323

/-- Jane's current age -/
def jane_current_age : ℕ := 34

/-- Number of years ago Jane stopped babysitting -/
def years_since_stopped_babysitting : ℕ := 10

/-- Current age of the oldest child Jane could have babysat -/
def oldest_child_current_age : ℕ := 25

/-- Calculate Jane's age when she stopped babysitting -/
def jane_age_when_stopped_babysitting : ℕ := jane_current_age - years_since_stopped_babysitting

/-- Calculate the child's age when Jane stopped babysitting -/
def oldest_child_age_when_jane_stopped : ℕ := oldest_child_current_age - years_since_stopped_babysitting 

/-- Calculate the fraction of Jane's age that the child could be at most -/
def babysitting_age_fraction : ℚ := (oldest_child_age_when_jane_stopped : ℚ) / (jane_age_when_stopped_babysitting : ℚ)

theorem fraction_of_janes_age_is_five_eighths :
  babysitting_age_fraction = 5 / 8 :=
by 
  -- Declare the proof steps (this part is the placeholder as proof is not required)
  sorry

end fraction_of_janes_age_is_five_eighths_l145_145323


namespace friendly_triangle_angle_l145_145888

theorem friendly_triangle_angle (α : ℝ) (β : ℝ) (γ : ℝ) (hα12β : α = 2 * β) (h_sum : α + β + γ = 180) :
    (α = 42 ∨ α = 84 ∨ α = 92) ∧ (42 = β ∨ 42 = γ) := 
sorry

end friendly_triangle_angle_l145_145888


namespace number_in_tenth_group_l145_145339

-- Number of students
def students : ℕ := 1000

-- Number of groups
def groups : ℕ := 100

-- Interval between groups
def interval : ℕ := students / groups

-- First number drawn
def first_number : ℕ := 6

-- Number drawn from n-th group given first_number and interval
def number_in_group (n : ℕ) : ℕ := first_number + interval * (n - 1)

-- Statement to prove
theorem number_in_tenth_group :
  number_in_group 10 = 96 :=
by
  sorry

end number_in_tenth_group_l145_145339


namespace least_tablets_l145_145143

theorem least_tablets (num_A num_B : ℕ) (hA : num_A = 10) (hB : num_B = 14) :
  ∃ n, n = 12 ∧
  ∀ extracted_tablets, extracted_tablets > 0 →
    (∃ (a b : ℕ), a + b = extracted_tablets ∧ a ≥ 2 ∧ b ≥ 2) :=
by
  sorry

end least_tablets_l145_145143


namespace original_price_hat_l145_145195

theorem original_price_hat 
  (x : ℝ)
  (discounted_price := x / 5)
  (final_price := discounted_price * 1.2)
  (h : final_price = 8) :
  x = 100 / 3 :=
by
  sorry

end original_price_hat_l145_145195


namespace intersection_of_M_and_N_l145_145660

noncomputable def setM : Set ℝ := { x : ℝ | x^2 - 2 * x - 3 < 0 }
noncomputable def setN : Set ℝ := { x : ℝ | Real.log x / Real.log 2 < 1 }

theorem intersection_of_M_and_N : { x : ℝ | x ∈ setM ∧ x ∈ setN } = { x : ℝ | 0 < x ∧ x < 2 } :=
by
  sorry

end intersection_of_M_and_N_l145_145660


namespace range_of_a_l145_145689

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, a * x ^ 2 + 2 * a * x + 1 ≤ 0) →
  0 ≤ a ∧ a < 1 :=
by
  -- sorry to skip the proof
  sorry

end range_of_a_l145_145689


namespace age_difference_l145_145834

variables (O N A : ℕ)

theorem age_difference (avg_age_stable : 10 * A = 10 * A + 50 - O + N) :
  O - N = 50 :=
by
  -- proof would go here
  sorry

end age_difference_l145_145834


namespace ones_digit_largest_power_of_2_divides_32_factorial_ones_digit_largest_power_of_3_divides_32_factorial_l145_145608

theorem ones_digit_largest_power_of_2_divides_32_factorial : 
  (2^31 % 10) = 8 := 
by
  sorry

theorem ones_digit_largest_power_of_3_divides_32_factorial : 
  (3^14 % 10) = 9 := 
by
  sorry

end ones_digit_largest_power_of_2_divides_32_factorial_ones_digit_largest_power_of_3_divides_32_factorial_l145_145608


namespace find_f2_l145_145023

theorem find_f2 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = x^2) : f 2 = 1 := 
by
  sorry

end find_f2_l145_145023


namespace additional_fertilizer_on_final_day_l145_145177

noncomputable def normal_usage_per_day : ℕ := 2
noncomputable def total_days : ℕ := 9
noncomputable def total_fertilizer_used : ℕ := 22

theorem additional_fertilizer_on_final_day :
  total_fertilizer_used - (normal_usage_per_day * total_days) = 4 := by
  sorry

end additional_fertilizer_on_final_day_l145_145177


namespace sum_solutions_eq_l145_145966

theorem sum_solutions_eq : 
  let a := 12
  let b := -19
  let c := -21
  (4 * x + 3) * (3 * x - 7) = 0 → (b/a) = 19/12 :=
by
  sorry

end sum_solutions_eq_l145_145966


namespace simplify_expression_at_zero_l145_145481

-- Define the expression f(x)
def f (x : ℚ) : ℚ := (2 * x + 4) / (x^2 - 6 * x + 9) / ((2 * x - 1) / (x - 3) - 1)

-- State that for the given value x = 0, the simplified expression equals -2/3
theorem simplify_expression_at_zero :
  f 0 = -2 / 3 :=
by
  sorry

end simplify_expression_at_zero_l145_145481


namespace problem_1_problem_2_problem_3_l145_145360

-- Definition for question 1:
def gcd_21n_4_14n_3 (n : ℕ) : Prop := (Nat.gcd (21 * n + 4) (14 * n + 3)) = 1

-- Definition for question 2:
def gcd_n_factorial_plus_1 (n : ℕ) : Prop := (Nat.gcd (Nat.factorial n + 1) (Nat.factorial (n + 1) + 1)) = 1

-- Definition for question 3:
def fermat_number (k : ℕ) : ℕ := 2^(2^k) + 1
def gcd_fermat_numbers (m n : ℕ) (h : m ≠ n) : Prop := (Nat.gcd (fermat_number m) (fermat_number n)) = 1

-- Theorem statements
theorem problem_1 (n : ℕ) (h_pos : 0 < n) : gcd_21n_4_14n_3 n := sorry

theorem problem_2 (n : ℕ) (h_pos : 0 < n) : gcd_n_factorial_plus_1 n := sorry

theorem problem_3 (m n : ℕ) (h_pos1 : 0 ≠ m) (h_pos2 : 0 ≠ n) (h_neq : m ≠ n) : gcd_fermat_numbers m n h_neq := sorry

end problem_1_problem_2_problem_3_l145_145360


namespace relationship_abc_d_l145_145758

theorem relationship_abc_d : 
  ∀ (a b c d : ℝ), 
  a < b → 
  d < c → 
  (c - a) * (c - b) < 0 → 
  (d - a) * (d - b) > 0 → 
  d < a ∧ a < c ∧ c < b :=
by
  intros a b c d a_lt_b d_lt_c h1 h2
  sorry

end relationship_abc_d_l145_145758


namespace BoatCrafters_l145_145731

/-
  Let J, F, M, A represent the number of boats built in January, February,
  March, and April respectively.

  Conditions:
  1. J = 4
  2. F = J / 2
  3. M = F * 3
  4. A = M * 3

  Goal:
  Prove that J + F + M + A = 30.
-/

def BoatCrafters.total_boats_built : Nat := 4 + (4 / 2) + ((4 / 2) * 3) + (((4 / 2) * 3) * 3)

theorem BoatCrafters.boats_built_by_end_of_April : 
  BoatCrafters.total_boats_built = 30 :=   
by 
  sorry

end BoatCrafters_l145_145731


namespace total_cost_of_ads_l145_145794

-- Define the conditions
def cost_ad1 := 3500
def minutes_ad1 := 2
def cost_ad2 := 4500
def minutes_ad2 := 3
def cost_ad3 := 3000
def minutes_ad3 := 3
def cost_ad4 := 4000
def minutes_ad4 := 2
def cost_ad5 := 5500
def minutes_ad5 := 5

-- Define the function to calculate the total cost
def total_cost :=
  (cost_ad1 * minutes_ad1) +
  (cost_ad2 * minutes_ad2) +
  (cost_ad3 * minutes_ad3) +
  (cost_ad4 * minutes_ad4) +
  (cost_ad5 * minutes_ad5)

-- The statement to prove
theorem total_cost_of_ads : total_cost = 66000 := by
  sorry

end total_cost_of_ads_l145_145794


namespace tan_triple_angle_formula_l145_145503

variable (θ : ℝ)
variable (h : Real.tan θ = 4)

theorem tan_triple_angle_formula : Real.tan (3 * θ) = 52 / 47 :=
by
  sorry  -- Proof is omitted

end tan_triple_angle_formula_l145_145503


namespace liam_savings_per_month_l145_145782

theorem liam_savings_per_month (trip_cost bill_cost left_after_bills : ℕ) 
                               (months_in_two_years : ℕ) (total_savings_per_month : ℕ) :
  trip_cost = 7000 →
  bill_cost = 3500 →
  left_after_bills = 8500 →
  months_in_two_years = 24 →
  total_savings_per_month = 19000 →
  total_savings_per_month / months_in_two_years = 79167 / 100 :=
by
  intros
  sorry

end liam_savings_per_month_l145_145782


namespace midpoint_to_plane_distance_l145_145548

noncomputable def distance_to_plane (A B P: ℝ) (dA dB: ℝ) : ℝ :=
if h : A = B then |dA|
else if h1 : dA + dB = (2 : ℝ) * (dA + dB) / 2 then (dA + dB) / 2
else if h2 : |dB - dA| = (2 : ℝ) * |dB - dA| / 2 then |dB - dA| / 2
else 0

theorem midpoint_to_plane_distance
  (α : Type*)
  (A B P: ℝ)
  {dA dB : ℝ}
  (h_dA : dA = 3)
  (h_dB : dB = 5) :
  distance_to_plane A B P dA dB = 4 ∨ distance_to_plane A B P dA dB = 1 :=
by sorry

end midpoint_to_plane_distance_l145_145548


namespace probability_club_then_queen_l145_145633

theorem probability_club_then_queen : 
  let total_cards := 52
  let total_clubs := 13
  let total_queens := 4
  let queen_of_clubs := 1
  let non_queen_clubs := total_clubs - queen_of_clubs
  
  let prob_queen_of_clubs_then_other_queen := (queen_of_clubs / total_cards) * ((total_queens - 1) / (total_cards - 1))
  let prob_non_queen_clubs_then_queen := (non_queen_clubs / total_cards) * (total_queens / (total_cards - 1))
  let total_probability := prob_queen_of_clubs_then_other_queen + prob_non_queen_clubs_then_queen
  
  total_probability = 1 / 52 := by
  let total_cards := 52
  let total_clubs := 13
  let total_queens := 4
  let queen_of_clubs := 1
  let non_queen_clubs := total_clubs - queen_of_clubs
  
  let prob_queen_of_clubs_then_other_queen := (queen_of_clubs / total_cards) * ((total_queens - 1) / (total_cards - 1))
  let prob_non_queen_clubs_then_queen := (non_queen_clubs / total_cards) * (total_queens / (total_cards - 1))
  let total_probability := prob_queen_of_clubs_then_other_queen + prob_non_queen_clubs_then_queen
  
  sorry

end probability_club_then_queen_l145_145633


namespace edge_c_eq_3_or_5_l145_145394

noncomputable def a := 7
noncomputable def b := 8
noncomputable def A := Real.pi / 3

theorem edge_c_eq_3_or_5 (c : ℝ) (h : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) : c = 3 ∨ c = 5 :=
by
  sorry

end edge_c_eq_3_or_5_l145_145394


namespace algebraic_expression_value_l145_145200

theorem algebraic_expression_value (m : ℝ) (h : m^2 - m - 2 = 0) : 2 * m^2 - 2 * m = 4 := by
  sorry

end algebraic_expression_value_l145_145200


namespace conference_games_l145_145201

theorem conference_games (teams_per_division : ℕ) (divisions : ℕ) 
  (intradivision_games_per_team : ℕ) (interdivision_games_per_team : ℕ) 
  (total_teams : ℕ) (total_games : ℕ) : 
  total_teams = teams_per_division * divisions →
  intradivision_games_per_team = (teams_per_division - 1) * 2 →
  interdivision_games_per_team = teams_per_division →
  total_games = (total_teams * (intradivision_games_per_team + interdivision_games_per_team)) / 2 →
  total_games = 133 :=
by
  intros
  sorry

end conference_games_l145_145201


namespace sodium_bicarbonate_moles_combined_l145_145097

theorem sodium_bicarbonate_moles_combined (HCl NaCl NaHCO3 : ℝ) (reaction : HCl + NaHCO3 = NaCl) 
  (HCl_eq_one : HCl = 1) (NaCl_eq_one : NaCl = 1) : 
  NaHCO3 = 1 := 
by 
  -- Placeholder for the proof
  sorry

end sodium_bicarbonate_moles_combined_l145_145097


namespace complete_the_square_solution_l145_145803

theorem complete_the_square_solution (x : ℝ) :
  (∃ x, x^2 + 2 * x - 1 = 0) → (x + 1)^2 = 2 :=
sorry

end complete_the_square_solution_l145_145803


namespace quadrant_of_half_angle_in_second_quadrant_l145_145342

theorem quadrant_of_half_angle_in_second_quadrant (θ : ℝ) (h : π / 2 < θ ∧ θ < π) :
  (0 < θ / 2 ∧ θ / 2 < π / 2) ∨ (π < θ / 2 ∧ θ / 2 < 3 * π / 2) :=
by
  sorry

end quadrant_of_half_angle_in_second_quadrant_l145_145342


namespace widget_difference_l145_145372

variable (w t : ℕ)

def monday_widgets (w t : ℕ) : ℕ := w * t
def tuesday_widgets (w t : ℕ) : ℕ := (w + 5) * (t - 3)

theorem widget_difference (h : w = 3 * t) :
  monday_widgets w t - tuesday_widgets w t = 4 * t + 15 :=
by
  sorry

end widget_difference_l145_145372


namespace remaining_kids_l145_145002

def initial_kids : Float := 22.0
def kids_who_went_home : Float := 14.0

theorem remaining_kids : initial_kids - kids_who_went_home = 8.0 :=
by 
  sorry

end remaining_kids_l145_145002


namespace solution_set_a_eq_2_solution_set_a_eq_neg_2_solution_set_neg_2_lt_a_lt_2_solution_set_a_lt_neg_2_or_a_gt_2_l145_145508

def solve_inequality (a x : ℝ) : Prop :=
  a^2 * x - 6 < 4 * x + 3 * a

theorem solution_set_a_eq_2 :
  ∀ x : ℝ, solve_inequality 2 x ↔ true :=
sorry

theorem solution_set_a_eq_neg_2 :
  ∀ x : ℝ, ¬ solve_inequality (-2) x :=
sorry

theorem solution_set_neg_2_lt_a_lt_2 (a : ℝ) (h : -2 < a ∧ a < 2) :
  ∀ x : ℝ, solve_inequality a x ↔ x > 3 / (a - 2) :=
sorry

theorem solution_set_a_lt_neg_2_or_a_gt_2 (a : ℝ) (h : a < -2 ∨ a > 2) :
  ∀ x : ℝ, solve_inequality a x ↔ x < 3 / (a - 2) :=
sorry

end solution_set_a_eq_2_solution_set_a_eq_neg_2_solution_set_neg_2_lt_a_lt_2_solution_set_a_lt_neg_2_or_a_gt_2_l145_145508


namespace missed_questions_l145_145952

theorem missed_questions (F M : ℕ) (h1 : M = 5 * F) (h2 : M + F = 216) : M = 180 :=
by
  sorry

end missed_questions_l145_145952


namespace find_P_l145_145636

theorem find_P 
  (digits : Finset ℕ) 
  (h_digits : digits = {1, 2, 3, 4, 5, 6}) 
  (P Q R S T U : ℕ)
  (h_unique : P ∈ digits ∧ Q ∈ digits ∧ R ∈ digits ∧ S ∈ digits ∧ T ∈ digits ∧ U ∈ digits ∧ 
              P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ P ≠ U ∧
              Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ Q ≠ U ∧
              R ≠ S ∧ R ≠ T ∧ R ≠ U ∧ 
              S ≠ T ∧ S ≠ U ∧ 
              T ≠ U) 
  (h_div5 : (100 * P + 10 * Q + R) % 5 = 0)
  (h_div3 : (100 * Q + 10 * R + S) % 3 = 0)
  (h_div2 : (100 * R + 10 * S + T) % 2 = 0) :
  P = 2 :=
sorry

end find_P_l145_145636


namespace regular_polygon_perimeter_l145_145958

def exterior_angle (n : ℕ) := 360 / n

theorem regular_polygon_perimeter
  (side_length : ℕ)
  (exterior_angle_deg : ℕ)
  (polygon_perimeter : ℕ)
  (h1 : side_length = 8)
  (h2 : exterior_angle_deg = 72)
  (h3 : ∃ n : ℕ, exterior_angle n = exterior_angle_deg)
  (h4 : ∀ n : ℕ, exterior_angle n = exterior_angle_deg → polygon_perimeter = n * side_length) :
  polygon_perimeter = 40 :=
sorry

end regular_polygon_perimeter_l145_145958


namespace students_play_at_least_one_sport_l145_145844

def B := 12
def C := 10
def S := 9
def Ba := 6

def B_and_C := 5
def B_and_S := 4
def B_and_Ba := 3
def C_and_S := 2
def C_and_Ba := 3
def S_and_Ba := 2

def B_and_C_and_S_and_Ba := 1

theorem students_play_at_least_one_sport : 
  B + C + S + Ba - B_and_C - B_and_S - B_and_Ba - C_and_S - C_and_Ba - S_and_Ba + B_and_C_and_S_and_Ba = 19 :=
by
  sorry

end students_play_at_least_one_sport_l145_145844


namespace math_problem_l145_145889

theorem math_problem (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, (x - 1) * (deriv f x) ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 :=
sorry

end math_problem_l145_145889


namespace find_b_from_root_and_constant_l145_145559

theorem find_b_from_root_and_constant
  (b k : ℝ)
  (h₁ : k = 44)
  (h₂ : ∃ (x : ℝ), x = 4 ∧ 2*x^2 + b*x - k = 0) :
  b = 3 :=
by
  sorry

end find_b_from_root_and_constant_l145_145559


namespace base_number_pow_k_eq_4_pow_2k_plus_2_eq_64_l145_145105

theorem base_number_pow_k_eq_4_pow_2k_plus_2_eq_64 (x k : ℝ) (h1 : x^k = 4) (h2 : x^(2 * k + 2) = 64) : x = 2 :=
sorry

end base_number_pow_k_eq_4_pow_2k_plus_2_eq_64_l145_145105


namespace sequence_property_l145_145236

theorem sequence_property (a : ℕ+ → ℤ) (h_add : ∀ p q : ℕ+, a (p + q) = a p + a q) (h_a2 : a 2 = -6) :
  a 10 = -30 := 
sorry

end sequence_property_l145_145236


namespace find_number_l145_145653

theorem find_number (x : ℝ) : 60 + (x * 12) / (180 / 3) = 61 ↔ x = 5 := by
  sorry  -- proof can be filled in here when needed

end find_number_l145_145653


namespace general_term_l145_145541

noncomputable def S (n : ℕ) : ℕ := sorry
noncomputable def a (n : ℕ) : ℕ := sorry

axiom S2 : S 2 = 4
axiom a_recurrence : ∀ n : ℕ, n > 0 → a (n + 1) = 2 * S n + 1

theorem general_term (n : ℕ) : a n = 3 ^ (n - 1) :=
by
  sorry

end general_term_l145_145541


namespace shift_down_two_units_l145_145411

theorem shift_down_two_units (x : ℝ) : 
  (y = 2 * x) → (y - 2 = 2 * x - 2) := by
sorry

end shift_down_two_units_l145_145411


namespace minimum_value_l145_145774

theorem minimum_value (a : ℝ) (h₀ : 0 < a) (h₁ : a < 3) :
  ∃ a : ℝ, (0 < a ∧ a < 3) ∧ (1 / a + 4 / (8 - a) = 9 / 8) := by
sorry

end minimum_value_l145_145774


namespace trapezoid_area_l145_145253

theorem trapezoid_area (a b H : ℝ) (h_lat1 : a = 10) (h_lat2 : b = 8) (h_height : H = b) : 
∃ S : ℝ, S = 104 :=
by sorry

end trapezoid_area_l145_145253


namespace max_value_of_f_l145_145578

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem max_value_of_f :
  ∃ x ∈ Set.Icc (0 : ℝ) 4, ∀ y ∈ Set.Icc (0 : ℝ) 4, f y ≤ f x ∧ f x = 1 / Real.exp 1 := 
by
  sorry

end max_value_of_f_l145_145578


namespace max_value_of_expression_l145_145207

theorem max_value_of_expression 
  (a b c : ℝ)
  (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  6 * a + 3 * b + 10 * c ≤ 3.2 :=
sorry

end max_value_of_expression_l145_145207


namespace perimeter_of_square_l145_145326

/-- The perimeter of a square with side length 15 cm is 60 cm -/
theorem perimeter_of_square (side_length : ℝ) (area : ℝ) (h1 : side_length = 15) (h2 : area = 225) :
  (4 * side_length = 60) :=
by
  -- Proof steps would go here (omitted)
  sorry

end perimeter_of_square_l145_145326


namespace sin_double_angle_l145_145597

theorem sin_double_angle (α : ℝ) 
  (h1 : Real.cos (α + Real.pi / 4) = 3 / 5)
  (h2 : Real.pi / 2 ≤ α ∧ α ≤ 3 * Real.pi / 2) : 
  Real.sin (2 * α) = 7 / 25 := 
by sorry

end sin_double_angle_l145_145597


namespace frustum_volume_l145_145444

noncomputable def volume_of_frustum (V₁ V₂ : ℝ) : ℝ :=
  V₁ - V₂

theorem frustum_volume : 
  let base_edge_original := 15
  let height_original := 10
  let base_edge_smaller := 9
  let height_smaller := 6
  let base_area_original := base_edge_original ^ 2
  let base_area_smaller := base_edge_smaller ^ 2
  let V_original := (1 / 3 : ℝ) * base_area_original * height_original
  let V_smaller := (1 / 3 : ℝ) * base_area_smaller * height_smaller
  volume_of_frustum V_original V_smaller = 588 := 
by
  sorry

end frustum_volume_l145_145444


namespace complement_of_M_l145_145404

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x ≥ 1}

theorem complement_of_M :
  (U \ M) = {x | x < 1} :=
by
  sorry

end complement_of_M_l145_145404


namespace complement_A_eq_B_subset_complement_A_l145_145493

-- Definitions of sets A and B
def A : Set ℝ := {x | x^2 + 4 * x > 0 }
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1 }

-- The universal set U is the set of all real numbers
def U : Set ℝ := Set.univ

-- Complement of A in U
def complement_U_A : Set ℝ := {x | -4 ≤ x ∧ x ≤ 0}

-- Proof statement for part (1)
theorem complement_A_eq : complement_U_A = {x | -4 ≤ x ∧ x ≤ 0} :=
  sorry 

-- Proof statement for part (2)
theorem B_subset_complement_A (a : ℝ) : B a ⊆ complement_U_A ↔ -3 ≤ a ∧ a ≤ -1 :=
  sorry 

end complement_A_eq_B_subset_complement_A_l145_145493


namespace shopkeeper_intended_profit_l145_145090

noncomputable def intended_profit_percentage (C L S : ℝ) : ℝ :=
  (L / C) - 1

theorem shopkeeper_intended_profit (C L S : ℝ) (h1 : L = C * (1 + intended_profit_percentage C L S))
  (h2 : S = 0.90 * L) (h3 : S = 1.35 * C) : intended_profit_percentage C L S = 0.5 :=
by
  -- We indicate that the proof is skipped
  sorry

end shopkeeper_intended_profit_l145_145090


namespace value_of_a_l145_145390

theorem value_of_a 
  (a : ℝ) 
  (h : 0.005 * a = 0.85) : 
  a = 170 :=
sorry

end value_of_a_l145_145390
