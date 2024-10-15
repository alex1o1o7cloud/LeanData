import Mathlib

namespace NUMINAMATH_GPT_abs_sum_bound_l1832_183226

theorem abs_sum_bound (k : ℝ) : (∀ x : ℝ, |x + 2| + |x + 1| > k) → k < 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_abs_sum_bound_l1832_183226


namespace NUMINAMATH_GPT_smallest_digits_to_append_l1832_183202

theorem smallest_digits_to_append (n : ℕ) (h : n = 2014) : 
  ∃ k : ℕ, (10^4 * n + k) % 2520 = 0 ∧ ∀ m, (10^m * n + k) % 2520 ≠ 0 → m > 4 := by
sorry

end NUMINAMATH_GPT_smallest_digits_to_append_l1832_183202


namespace NUMINAMATH_GPT_katherine_bottle_caps_l1832_183237

-- Define the initial number of bottle caps Katherine has
def initial_bottle_caps : ℕ := 34

-- Define the number of bottle caps eaten by the hippopotamus
def eaten_bottle_caps : ℕ := 8

-- Define the remaining number of bottle caps Katherine should have
def remaining_bottle_caps : ℕ := initial_bottle_caps - eaten_bottle_caps

-- Theorem stating that Katherine will have 26 bottle caps after the hippopotamus eats 8 of them
theorem katherine_bottle_caps : remaining_bottle_caps = 26 := by
  sorry

end NUMINAMATH_GPT_katherine_bottle_caps_l1832_183237


namespace NUMINAMATH_GPT_compare_2_5_sqrt_6_l1832_183288

theorem compare_2_5_sqrt_6 : 2.5 > Real.sqrt 6 := by
  sorry

end NUMINAMATH_GPT_compare_2_5_sqrt_6_l1832_183288


namespace NUMINAMATH_GPT_find_y_find_x_l1832_183296

-- Define vectors as per the conditions
def a : ℝ × ℝ := (3, -2)
def b (y : ℝ) : ℝ × ℝ := (-1, y)
def c (x : ℝ) : ℝ × ℝ := (x, 5)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the condition for perpendicular vectors
def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

-- Define the condition for parallel vectors
def is_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Question 1 Proof Statement
theorem find_y : ∀ (y : ℝ), is_perpendicular a (b y) → y = 3 / 2 :=
by
  intros y h
  unfold is_perpendicular at h
  unfold dot_product at h
  sorry

-- Question 2 Proof Statement
theorem find_x : ∀ (x : ℝ), is_parallel a (c x) → x = 15 / 2 :=
by
  intros x h
  unfold is_parallel at h
  sorry

end NUMINAMATH_GPT_find_y_find_x_l1832_183296


namespace NUMINAMATH_GPT_company_starts_to_make_profit_in_third_year_first_option_more_cost_effective_l1832_183239

-- Define the conditions about the fishing company's boat purchase and expenses
def initial_purchase_cost : ℕ := 980000
def first_year_expenses : ℕ := 120000
def expense_increment : ℕ := 40000
def annual_income : ℕ := 500000

-- Prove that the company starts to make a profit in the third year
theorem company_starts_to_make_profit_in_third_year : 
  ∃ (year : ℕ), year = 3 ∧ 
  annual_income * year > initial_purchase_cost + first_year_expenses + (expense_increment * (year - 1) * year / 2) :=
sorry

-- Prove that the first option is more cost-effective
theorem first_option_more_cost_effective : 
  (annual_income * 3 - (initial_purchase_cost + first_year_expenses + expense_increment * (3 - 1) * 3 / 2) + 260000) > 
  (annual_income * 5 - (initial_purchase_cost + first_year_expenses + expense_increment * (5 - 1) * 5 / 2) + 80000) :=
sorry

end NUMINAMATH_GPT_company_starts_to_make_profit_in_third_year_first_option_more_cost_effective_l1832_183239


namespace NUMINAMATH_GPT_hibiscus_flower_ratio_l1832_183207

theorem hibiscus_flower_ratio (x : ℕ) 
  (h1 : 2 + x + 4 * x = 22) : x / 2 = 2 := 
sorry

end NUMINAMATH_GPT_hibiscus_flower_ratio_l1832_183207


namespace NUMINAMATH_GPT_integer_roots_l1832_183222

-- Define the polynomial
def poly (x : ℤ) : ℤ := x^3 - 4 * x^2 - 11 * x + 24

-- State the theorem
theorem integer_roots : {x : ℤ | poly x = 0} = {-1, 2, 3} := 
  sorry

end NUMINAMATH_GPT_integer_roots_l1832_183222


namespace NUMINAMATH_GPT_sum_three_smallest_m_l1832_183209

theorem sum_three_smallest_m :
  (∃ a m, 
    (a - 2 + a + a + 2) / 3 = 7 
    ∧ m % 4 = 3 
    ∧ m ≠ 5 ∧ m ≠ 7 ∧ m ≠ 9 
    ∧ (5 + 7 + 9 + m) % 4 = 0 
    ∧ m > 0) 
  → 3 + 11 + 15 = 29 :=
sorry

end NUMINAMATH_GPT_sum_three_smallest_m_l1832_183209


namespace NUMINAMATH_GPT_correct_statement_d_l1832_183290

theorem correct_statement_d : 
  (∃ x : ℝ, 2^x < x^2) ↔ ¬(∀ x : ℝ, 2^x ≥ x^2) :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_d_l1832_183290


namespace NUMINAMATH_GPT_cost_reduction_l1832_183275

variable (a : ℝ) -- original cost
variable (p : ℝ) -- percentage reduction (in decimal form)
variable (m : ℕ) -- number of years

def cost_after_years (a p : ℝ) (m : ℕ) : ℝ :=
  a * (1 - p) ^ m

theorem cost_reduction (a p : ℝ) (m : ℕ) :
  m > 0 → cost_after_years a p m = a * (1 - p) ^ m :=
sorry

end NUMINAMATH_GPT_cost_reduction_l1832_183275


namespace NUMINAMATH_GPT_binomial_sum_to_220_l1832_183270

open Nat

def binomial_coeff (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binomial_sum_to_220 :
  binomial_coeff 2 2 + binomial_coeff 3 2 + binomial_coeff 4 2 + binomial_coeff 5 2 +
  binomial_coeff 6 2 + binomial_coeff 7 2 + binomial_coeff 8 2 + binomial_coeff 9 2 +
  binomial_coeff 10 2 + binomial_coeff 11 2 = 220 :=
by
  /- Proof goes here, use the computed value of combinations -/
  sorry

end NUMINAMATH_GPT_binomial_sum_to_220_l1832_183270


namespace NUMINAMATH_GPT_richmond_tigers_tickets_l1832_183267

theorem richmond_tigers_tickets (total_tickets first_half_tickets : ℕ) 
  (h1 : total_tickets = 9570)
  (h2 : first_half_tickets = 3867) : 
  total_tickets - first_half_tickets = 5703 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_richmond_tigers_tickets_l1832_183267


namespace NUMINAMATH_GPT_james_vegetable_consumption_l1832_183214

theorem james_vegetable_consumption : 
  (1/4 + 1/4) * 2 * 7 + 3 = 10 := 
by
  sorry

end NUMINAMATH_GPT_james_vegetable_consumption_l1832_183214


namespace NUMINAMATH_GPT_avg_of_multiples_of_4_is_even_l1832_183200

theorem avg_of_multiples_of_4_is_even (m n : ℤ) (hm : m % 4 = 0) (hn : n % 4 = 0) :
  (m + n) / 2 % 2 = 0 := sorry

end NUMINAMATH_GPT_avg_of_multiples_of_4_is_even_l1832_183200


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1832_183246

-- Define the conditions of the hyperbola and points
variables {a b c m d : ℝ} (ha : a > 0) (hb : b > 0) 
noncomputable def F1 : ℝ := sorry -- Placeholder for focus F1
noncomputable def F2 : ℝ := sorry -- Placeholder for focus F2
noncomputable def P : ℝ := sorry  -- Placeholder for point P

-- Define the sides of the triangle in terms of an arithmetic progression
def PF2 (m d : ℝ) : ℝ := m - d
def PF1 (m : ℝ) : ℝ := m
def F1F2 (m d : ℝ) : ℝ := m + d

-- Prove that the eccentricity is 5 given the conditions
theorem hyperbola_eccentricity 
  (m d : ℝ) (hc : c = (5 / 2) * d )  
  (h1 : PF1 m = 2 * a)
  (h2 : F1F2 m d = 2 * c)
  (h3 : (PF2 m d)^2 + (PF1 m)^2 = (F1F2 m d)^2 ) :
  (c / a) = 5 := 
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1832_183246


namespace NUMINAMATH_GPT_comparison_of_logs_l1832_183238

noncomputable def a : ℝ := Real.logb 4 6
noncomputable def b : ℝ := Real.logb 4 0.2
noncomputable def c : ℝ := Real.logb 2 3

theorem comparison_of_logs : c > a ∧ a > b := by
  sorry

end NUMINAMATH_GPT_comparison_of_logs_l1832_183238


namespace NUMINAMATH_GPT_abs_x_minus_1_lt_2_is_necessary_but_not_sufficient_l1832_183291

theorem abs_x_minus_1_lt_2_is_necessary_but_not_sufficient (x : ℝ) :
  (-1 < x ∧ x < 3) ↔ (0 < x ∧ x < 3) :=
sorry

end NUMINAMATH_GPT_abs_x_minus_1_lt_2_is_necessary_but_not_sufficient_l1832_183291


namespace NUMINAMATH_GPT_intersection_is_correct_l1832_183248

def M : Set ℤ := {x | x^2 + 3 * x + 2 > 0}
def N : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_is_correct : M ∩ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_GPT_intersection_is_correct_l1832_183248


namespace NUMINAMATH_GPT_carlos_paid_l1832_183299

theorem carlos_paid (a b c : ℝ) 
  (h1 : a = (1 / 3) * (b + c))
  (h2 : b = (1 / 4) * (a + c))
  (h3 : a + b + c = 120) :
  c = 72 :=
by
-- Proof omitted
sorry

end NUMINAMATH_GPT_carlos_paid_l1832_183299


namespace NUMINAMATH_GPT_jane_total_investment_in_stocks_l1832_183255

-- Definitions
def total_investment := 220000
def bonds_investment := 13750
def stocks_investment := 5 * bonds_investment
def mutual_funds_investment := 2 * stocks_investment

-- Condition: The total amount invested
def total_investment_condition : Prop := 
  bonds_investment + stocks_investment + mutual_funds_investment = total_investment

-- Theorem: Jane's total investment in stocks
theorem jane_total_investment_in_stocks :
  total_investment_condition →
  stocks_investment = 68750 :=
by sorry

end NUMINAMATH_GPT_jane_total_investment_in_stocks_l1832_183255


namespace NUMINAMATH_GPT_barbara_initial_candies_l1832_183210

noncomputable def initialCandies (used left: ℝ) := used + left

theorem barbara_initial_candies (used left: ℝ) (h_used: used = 9.0) (h_left: left = 9) : initialCandies used left = 18 := 
by
  rw [h_used, h_left]
  norm_num
  sorry

end NUMINAMATH_GPT_barbara_initial_candies_l1832_183210


namespace NUMINAMATH_GPT_incorrect_conclusions_l1832_183229

theorem incorrect_conclusions :
  let p := (∀ x y : ℝ, x * y ≠ 6 → x ≠ 2 ∨ y ≠ 3)
  let q := (2, 1) ∈ { p : ℝ × ℝ | p.2 = 2 * p.1 - 3 }
  (p ∨ ¬q) = false ∧ (¬p ∨ q) = false ∧ (p ∧ ¬q) = false :=
by
  sorry

end NUMINAMATH_GPT_incorrect_conclusions_l1832_183229


namespace NUMINAMATH_GPT_inequality_proof_l1832_183278

theorem inequality_proof
  (a b c d : ℝ)
  (a_nonneg : 0 ≤ a)
  (b_nonneg : 0 ≤ b)
  (c_nonneg : 0 ≤ c)
  (d_nonneg : 0 ≤ d)
  (sum_eq_one : a + b + c + d = 1) :
  abc + bcd + cda + dab ≤ (1 / 27) + (176 * abcd / 27) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1832_183278


namespace NUMINAMATH_GPT_calculate_expression_l1832_183245

theorem calculate_expression :
  (π - 1)^0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1832_183245


namespace NUMINAMATH_GPT_initial_action_figures_l1832_183218

theorem initial_action_figures (x : ℕ) (h : x + 4 - 1 = 6) : x = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_action_figures_l1832_183218


namespace NUMINAMATH_GPT_age_of_teacher_l1832_183242

/-- Given that the average age of 23 students is 22 years, and the average age increases
by 1 year when the teacher's age is included, prove that the teacher's age is 46 years. -/
theorem age_of_teacher (n : ℕ) (s_avg : ℕ) (new_avg : ℕ) (teacher_age : ℕ) :
  n = 23 →
  s_avg = 22 →
  new_avg = s_avg + 1 →
  teacher_age = new_avg * (n + 1) - s_avg * n →
  teacher_age = 46 :=
by
  intros h_n h_s_avg h_new_avg h_teacher_age
  sorry

end NUMINAMATH_GPT_age_of_teacher_l1832_183242


namespace NUMINAMATH_GPT_stock_price_at_end_of_second_year_l1832_183258

def stock_price_first_year (initial_price : ℝ) : ℝ :=
  initial_price * 2

def stock_price_second_year (price_after_first_year : ℝ) : ℝ :=
  price_after_first_year * 0.75

theorem stock_price_at_end_of_second_year : 
  (stock_price_second_year (stock_price_first_year 100) = 150) :=
by
  sorry

end NUMINAMATH_GPT_stock_price_at_end_of_second_year_l1832_183258


namespace NUMINAMATH_GPT_express_y_in_terms_of_x_l1832_183215

theorem express_y_in_terms_of_x (x y : ℝ) (h : 5 * x + y = 1) : y = 1 - 5 * x :=
by
  sorry

end NUMINAMATH_GPT_express_y_in_terms_of_x_l1832_183215


namespace NUMINAMATH_GPT_car_speed_l1832_183279

variable (D : ℝ) (V : ℝ)

theorem car_speed
  (h1 : 1 / ((D / 3) / 80) + (D / 3) / 15 + (D / 3) / V = D / 30) :
  V = 35.625 :=
by 
  sorry

end NUMINAMATH_GPT_car_speed_l1832_183279


namespace NUMINAMATH_GPT_number_of_odd_blue_faces_cubes_l1832_183281

/-
A wooden block is 5 inches long, 5 inches wide, and 1 inch high.
The block is painted blue on all six sides and then cut into twenty-five 1 inch cubes.
Prove that the number of cubes each have a total number of blue faces that is an odd number is 9.
-/

def cubes_with_odd_blue_faces : ℕ :=
  let corner_cubes := 4
  let edge_cubes_not_corners := 16
  let center_cubes := 5
  corner_cubes + center_cubes

theorem number_of_odd_blue_faces_cubes : cubes_with_odd_blue_faces = 9 := by
  have h1 : cubes_with_odd_blue_faces = 4 + 5 := sorry
  have h2 : 4 + 5 = 9 := by norm_num
  exact Eq.trans h1 h2

end NUMINAMATH_GPT_number_of_odd_blue_faces_cubes_l1832_183281


namespace NUMINAMATH_GPT_min_C_over_D_l1832_183240

theorem min_C_over_D (x C D : ℝ) (h1 : x^2 + 1/x^2 = C) (h2 : x + 1/x = D) (hC_pos : 0 < C) (hD_pos : 0 < D) : 
  (∃ m : ℝ, m = 2 * Real.sqrt 2 ∧ ∀ y : ℝ, y = C / D → y ≥ m) :=
  sorry

end NUMINAMATH_GPT_min_C_over_D_l1832_183240


namespace NUMINAMATH_GPT_root_implies_m_values_l1832_183213

theorem root_implies_m_values (m : ℝ) :
  (∃ x : ℝ, x = 1 ∧ (m + 2) * x^2 - 2 * x + m^2 - 2 * m - 6 = 0) →
  (m = 3 ∨ m = -2) :=
by
  sorry

end NUMINAMATH_GPT_root_implies_m_values_l1832_183213


namespace NUMINAMATH_GPT_jessa_gave_3_bills_l1832_183263

variable (J G K : ℕ)
variable (billsGiven : ℕ)

/-- Initial conditions and question for the problem -/
def initial_conditions :=
  G = 16 ∧
  K = J - 2 ∧
  G = 2 * K ∧
  (J - billsGiven = 7)

/-- The theorem to prove: Jessa gave 3 bills to Geric -/
theorem jessa_gave_3_bills (h : initial_conditions J G K billsGiven) : billsGiven = 3 := 
sorry

end NUMINAMATH_GPT_jessa_gave_3_bills_l1832_183263


namespace NUMINAMATH_GPT_sums_equal_l1832_183230

theorem sums_equal (A B C : Type) (a b c : ℕ) :
  (a + b + c) = (a + (b + c)) ∧
  (a + b + c) = (b + (c + a)) ∧
  (a + b + c) = (c + (a + b)) :=
by 
  sorry

end NUMINAMATH_GPT_sums_equal_l1832_183230


namespace NUMINAMATH_GPT_symmetric_point_proof_l1832_183221

def Point3D := (ℝ × ℝ × ℝ)

def symmetric_point_yOz (p : Point3D) : Point3D :=
  let (x, y, z) := p
  (-x, y, z)

theorem symmetric_point_proof :
  symmetric_point_yOz (1, -2, 3) = (-1, -2, 3) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_proof_l1832_183221


namespace NUMINAMATH_GPT_a2022_value_l1832_183254

theorem a2022_value 
  (a : Fin 2022 → ℤ)
  (h : ∀ n k : Fin 2022, a n - a k ≥ n.1^3 - k.1^3)
  (a1011 : a 1010 = 0) :
  a 2021 = 2022^3 - 1011^3 :=
by
  sorry

end NUMINAMATH_GPT_a2022_value_l1832_183254


namespace NUMINAMATH_GPT_problem_statement_l1832_183212

variable (a b c p q r α β γ : ℝ)

-- Given conditions
def plane_condition : Prop := (a / α) + (b / β) + (c / γ) = 1
def sphere_conditions : Prop := p^3 = α ∧ q^3 = β ∧ r^3 = γ

-- The statement to prove
theorem problem_statement (h_plane : plane_condition a b c α β γ) (h_sphere : sphere_conditions p q r α β γ) :
  (a / p^3) + (b / q^3) + (c / r^3) = 1 := sorry

end NUMINAMATH_GPT_problem_statement_l1832_183212


namespace NUMINAMATH_GPT_students_neither_l1832_183276

def total_students : ℕ := 150
def students_math : ℕ := 85
def students_physics : ℕ := 63
def students_chemistry : ℕ := 40
def students_math_physics : ℕ := 20
def students_physics_chemistry : ℕ := 15
def students_math_chemistry : ℕ := 10
def students_all_three : ℕ := 5

theorem students_neither:
  total_students - 
  (students_math + students_physics + students_chemistry 
  - students_math_physics - students_physics_chemistry 
  - students_math_chemistry + students_all_three) = 2 := 
by sorry

end NUMINAMATH_GPT_students_neither_l1832_183276


namespace NUMINAMATH_GPT_distance_between_intersections_is_sqrt3_l1832_183235

noncomputable def intersection_distance : ℝ :=
  let C1_polar := (θ : ℝ) → θ = (2 * Real.pi / 3)
  let C2_standard := (x y : ℝ) → (x + Real.sqrt 3)^2 + (y + 2)^2 = 1
  let C3 := (θ : ℝ) → θ = (Real.pi / 3) 
  let C3_cartesian := (x y : ℝ) → y = Real.sqrt 3 * x
  let center := (-Real.sqrt 3, -2)
  let dist_to_C3 := abs (-3 + 2) / 2
  2 * Real.sqrt (1 - (dist_to_C3)^2)

theorem distance_between_intersections_is_sqrt3:
  intersection_distance = Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_distance_between_intersections_is_sqrt3_l1832_183235


namespace NUMINAMATH_GPT_paul_sold_11_books_l1832_183262

variable (initial_books : ℕ) (books_given : ℕ) (books_left : ℕ) (books_sold : ℕ)

def number_of_books_sold (initial_books books_given books_left books_sold : ℕ) : Prop :=
  initial_books - books_given - books_left = books_sold

theorem paul_sold_11_books : number_of_books_sold 108 35 62 11 :=
by
  sorry

end NUMINAMATH_GPT_paul_sold_11_books_l1832_183262


namespace NUMINAMATH_GPT_cds_probability_l1832_183250

def probability (total favorable : ℕ) : ℚ := favorable / total

theorem cds_probability :
  probability 120 24 = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cds_probability_l1832_183250


namespace NUMINAMATH_GPT_minimum_sum_of_nine_consecutive_integers_l1832_183211

-- We will define the consecutive sequence and the conditions as described.
structure ConsecutiveIntegers (a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℕ) : Prop :=
(seq : a1 + 1 = a2 ∧ a2 + 1 = a3 ∧ a3 + 1 = a4 ∧ a4 + 1 = a5 ∧ a5 + 1 = a6 ∧ a6 + 1 = a7 ∧ a7 + 1 = a8 ∧ a8 + 1 = a9)
(sq_cond : ∃ k : ℕ, (a1 + a3 + a5 + a7 + a9) = k * k)
(cube_cond : ∃ l : ℕ, (a2 + a4 + a6 + a8) = l * l * l)

theorem minimum_sum_of_nine_consecutive_integers :
  ∃ a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℕ,
  ConsecutiveIntegers a1 a2 a3 a4 a5 a6 a7 a8 a9 ∧ (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 = 18000) :=
  sorry

end NUMINAMATH_GPT_minimum_sum_of_nine_consecutive_integers_l1832_183211


namespace NUMINAMATH_GPT_max_tickets_l1832_183251

-- Define the conditions
def ticket_cost (n : ℕ) : ℝ :=
  if n ≤ 6 then 15 * n
  else 13.5 * n

-- Define the main theorem
theorem max_tickets (budget : ℝ) : (∀ n : ℕ, ticket_cost n ≤ budget) → budget = 120 → n ≤ 8 :=
  by
  sorry

end NUMINAMATH_GPT_max_tickets_l1832_183251


namespace NUMINAMATH_GPT_moles_of_silver_nitrate_needed_l1832_183203

structure Reaction :=
  (reagent1 : String)
  (reagent2 : String)
  (product1 : String)
  (product2 : String)
  (ratio_reagent1_to_product2 : ℕ) -- Moles of reagent1 to product2 in the balanced reaction

def silver_nitrate_hydrochloric_acid_reaction : Reaction :=
  { reagent1 := "AgNO3",
    reagent2 := "HCl",
    product1 := "AgCl",
    product2 := "HNO3",
    ratio_reagent1_to_product2 := 1 }

theorem moles_of_silver_nitrate_needed
  (reaction : Reaction)
  (hCl_initial_moles : ℕ)
  (hno3_target_moles : ℕ) :
  hno3_target_moles = 2 →
  (reaction.ratio_reagent1_to_product2 = 1 ∧ hCl_initial_moles = 2) →
  (hno3_target_moles = reaction.ratio_reagent1_to_product2 * 2 ∧ hno3_target_moles = 2) :=
by
  sorry

end NUMINAMATH_GPT_moles_of_silver_nitrate_needed_l1832_183203


namespace NUMINAMATH_GPT_sandbox_perimeter_l1832_183294

def sandbox_width : ℝ := 5
def sandbox_length := 2 * sandbox_width
def perimeter (length width : ℝ) := 2 * (length + width)

theorem sandbox_perimeter : perimeter sandbox_length sandbox_width = 30 := 
by
  sorry

end NUMINAMATH_GPT_sandbox_perimeter_l1832_183294


namespace NUMINAMATH_GPT_solution_l1832_183277

noncomputable def problem_statement : ℝ :=
  let a := 6
  let b := 5
  let x := 10 * a + b
  let y := 10 * b + a
  let m := 16.5
  x + y + m

theorem solution : problem_statement = 137.5 :=
by
  sorry

end NUMINAMATH_GPT_solution_l1832_183277


namespace NUMINAMATH_GPT_neg_existential_proposition_l1832_183283

open Nat

theorem neg_existential_proposition :
  (¬ (∃ n : ℕ, n + 10 / n < 4)) ↔ (∀ n : ℕ, n + 10 / n ≥ 4) :=
by
  sorry

end NUMINAMATH_GPT_neg_existential_proposition_l1832_183283


namespace NUMINAMATH_GPT_minimum_shoeing_time_l1832_183292

theorem minimum_shoeing_time 
  (blacksmiths : ℕ) (horses : ℕ) (hooves_per_horse : ℕ) (time_per_hoof : ℕ) 
  (total_hooves : ℕ := horses * hooves_per_horse) 
  (time_for_one_blacksmith : ℕ := total_hooves * time_per_hoof) 
  (total_parallel_time : ℕ := time_for_one_blacksmith / blacksmiths)
  (h : blacksmiths = 48)
  (h' : horses = 60)
  (h'' : hooves_per_horse = 4)
  (h''' : time_per_hoof = 5) : 
  total_parallel_time = 25 :=
by
  sorry

end NUMINAMATH_GPT_minimum_shoeing_time_l1832_183292


namespace NUMINAMATH_GPT_sum_of_roots_unique_solution_l1832_183298

open Real

def operation (a b : ℝ) : ℝ := a^2 + 2 * a * b - b^2

def f (x : ℝ) : ℝ := operation x 2

theorem sum_of_roots_unique_solution
  (x1 x2 x3 x4 : ℝ)
  (h1 : ∀ x, f x = log (abs (x + 2)) → x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4)
  (h2 : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) :
  x1 + x2 + x3 + x4 = -8 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_unique_solution_l1832_183298


namespace NUMINAMATH_GPT_parabola_equation_l1832_183227

theorem parabola_equation (M : ℝ × ℝ) (hM : M = (5, 3))
    (h_dist : ∀ a : ℝ, |5 + 1/(4*a)| = 6) :
    (y = (1/12)*x^2) ∨ (y = -(1/36)*x^2) :=
sorry

end NUMINAMATH_GPT_parabola_equation_l1832_183227


namespace NUMINAMATH_GPT_probability_points_one_unit_apart_l1832_183252

theorem probability_points_one_unit_apart :
  let points := 10
  let rect_length := 3
  let rect_width := 2
  let total_pairs := (points * (points - 1)) / 2
  let favorable_pairs := 10  -- derived from solution steps
  (favorable_pairs / total_pairs : ℚ) = (2 / 9 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_probability_points_one_unit_apart_l1832_183252


namespace NUMINAMATH_GPT_least_number_to_add_l1832_183286

theorem least_number_to_add (n : ℕ) (h : (1052 + n) % 37 = 0) : n = 19 := by
  sorry

end NUMINAMATH_GPT_least_number_to_add_l1832_183286


namespace NUMINAMATH_GPT_largest_integer_satisfying_sin_cos_condition_proof_l1832_183269

noncomputable def largest_integer_satisfying_sin_cos_condition :=
  ∀ (x : ℝ) (n : ℕ), (∀ (n' : ℕ), (∀ x : ℝ, (Real.sin x ^ n' + Real.cos x ^ n' ≥ 2 / n') → n ≤ n')) → n = 4

theorem largest_integer_satisfying_sin_cos_condition_proof :
  largest_integer_satisfying_sin_cos_condition :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_satisfying_sin_cos_condition_proof_l1832_183269


namespace NUMINAMATH_GPT_production_days_l1832_183206

noncomputable def daily_production (n : ℕ) : Prop :=
50 * n + 90 = 58 * (n + 1)

theorem production_days (n : ℕ) (h : daily_production n) : n = 4 :=
by sorry

end NUMINAMATH_GPT_production_days_l1832_183206


namespace NUMINAMATH_GPT_solve_for_x_l1832_183253

theorem solve_for_x (x : ℝ) (h : 3 * x - 5 * x + 6 * x = 150) : x = 37.5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1832_183253


namespace NUMINAMATH_GPT_randy_initial_blocks_l1832_183204

theorem randy_initial_blocks (x : ℕ) (used_blocks : ℕ) (left_blocks : ℕ) 
  (h1 : used_blocks = 36) (h2 : left_blocks = 23) (h3 : x = used_blocks + left_blocks) :
  x = 59 := by 
  sorry

end NUMINAMATH_GPT_randy_initial_blocks_l1832_183204


namespace NUMINAMATH_GPT_proposition_truth_count_l1832_183273

namespace Geometry

def is_obtuse_angle (A : Type) : Prop := sorry
def is_obtuse_triangle (ABC : Type) : Prop := sorry

def original_proposition (A : Type) (ABC : Type) : Prop :=
is_obtuse_angle A → is_obtuse_triangle ABC

def contrapositive_proposition (A : Type) (ABC : Type) : Prop :=
¬ (is_obtuse_triangle ABC) → ¬ (is_obtuse_angle A)

def converse_proposition (ABC : Type) (A : Type) : Prop :=
is_obtuse_triangle ABC → is_obtuse_angle A

def inverse_proposition (A : Type) (ABC : Type) : Prop :=
¬ (is_obtuse_angle A) → ¬ (is_obtuse_triangle ABC)

theorem proposition_truth_count (A : Type) (ABC : Type) :
  (original_proposition A ABC ∧ contrapositive_proposition A ABC ∧
  ¬ (converse_proposition ABC A) ∧ ¬ (inverse_proposition A ABC)) →
  ∃ n : ℕ, n = 2 :=
sorry

end Geometry

end NUMINAMATH_GPT_proposition_truth_count_l1832_183273


namespace NUMINAMATH_GPT_probability_not_all_dice_show_different_l1832_183201

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  (total_outcomes - same_number_outcomes) / total_outcomes

theorem probability_not_all_dice_show_different : 
  probability_not_all_same = 4095 / 4096 := by
  sorry

end NUMINAMATH_GPT_probability_not_all_dice_show_different_l1832_183201


namespace NUMINAMATH_GPT_incorrect_statement_d_l1832_183261

-- Definitions from the problem:
variables (x y : ℝ)
variables (b a : ℝ)
variables (x_bar y_bar : ℝ)

-- Linear regression equation:
def linear_regression (x y : ℝ) (b a : ℝ) : Prop :=
  y = b * x + a

-- Properties given in the problem:
axiom pass_through_point : ∀ (x_bar y_bar : ℝ), ∃ b a, y_bar = b * x_bar + a
axiom avg_increase : ∀ (b a : ℝ), y = b * (x + 1) + a → y = b * x + a + b
axiom possible_at_origin : ∀ (b a : ℝ), ∃ y, y = a

-- The statement D which is incorrect:
theorem incorrect_statement_d : ¬ (∀ (b a : ℝ), ∀ y, x = 0 → y = a) :=
sorry

end NUMINAMATH_GPT_incorrect_statement_d_l1832_183261


namespace NUMINAMATH_GPT_SufficientCondition_l1832_183220

theorem SufficientCondition :
  ∀ x y z : ℤ, x = z ∧ y = x - 1 → x * (x - y) + y * (y - z) + z * (z - x) = 2 :=
by
  intros x y z h
  cases h with
  | intro h1 h2 =>
  sorry

end NUMINAMATH_GPT_SufficientCondition_l1832_183220


namespace NUMINAMATH_GPT_cooking_time_at_least_l1832_183297

-- Definitions based on conditions
def total_potatoes : ℕ := 35
def cooked_potatoes : ℕ := 11
def time_per_potato : ℕ := 7 -- in minutes
def salad_time : ℕ := 15 -- in minutes

-- The statement to prove
theorem cooking_time_at_least (oven_capacity : ℕ) :
  ∃ t : ℕ, t ≥ salad_time :=
by
  sorry

end NUMINAMATH_GPT_cooking_time_at_least_l1832_183297


namespace NUMINAMATH_GPT_discount_price_equation_correct_l1832_183234

def original_price := 200
def final_price := 148
variable (a : ℝ) -- assuming a is a real number representing the percentage discount

theorem discount_price_equation_correct :
  original_price * (1 - a / 100) ^ 2 = final_price :=
sorry

end NUMINAMATH_GPT_discount_price_equation_correct_l1832_183234


namespace NUMINAMATH_GPT_find_range_of_a_l1832_183266

theorem find_range_of_a (a : ℝ) (x : ℝ) (y : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 2 ≤ y ∧ y ≤ 3) 
    (hineq : x * y ≤ a * x^2 + 2 * y^2) : 
    -1 ≤ a := sorry

end NUMINAMATH_GPT_find_range_of_a_l1832_183266


namespace NUMINAMATH_GPT_no_integer_solutions_for_system_l1832_183241

theorem no_integer_solutions_for_system :
  ∀ (y z : ℤ),
    (2 * y^2 - 2 * y * z - z^2 = 15) ∧ 
    (6 * y * z + 2 * z^2 = 60) ∧ 
    (y^2 + 8 * z^2 = 90) 
    → False :=
by 
  intro y z
  simp
  sorry

end NUMINAMATH_GPT_no_integer_solutions_for_system_l1832_183241


namespace NUMINAMATH_GPT_time_to_print_800_flyers_l1832_183244

theorem time_to_print_800_flyers (x : ℝ) (h1 : 0 < x) :
  (1 / 6) + (1 / x) = 1 / 1.5 ↔ ∀ y : ℝ, 800 / 6 + 800 / x = 800 / 1.5 :=
by sorry

end NUMINAMATH_GPT_time_to_print_800_flyers_l1832_183244


namespace NUMINAMATH_GPT_train_crossing_time_l1832_183256

-- Definitions of the given problem conditions
def train_length : ℕ := 120  -- in meters.
def speed_kmph : ℕ := 144   -- in km/h.

-- Conversion factor
def km_per_hr_to_m_per_s (speed : ℕ) : ℚ :=
  speed * (1000 / 3600 : ℚ)

-- Speed in m/s
def train_speed : ℚ := km_per_hr_to_m_per_s speed_kmph

-- Time calculation
def time_to_cross_pole (length : ℕ) (speed : ℚ) : ℚ :=
  length / speed

-- The theorem we want to prove.
theorem train_crossing_time :
  time_to_cross_pole train_length train_speed = 3 := by 
  sorry

end NUMINAMATH_GPT_train_crossing_time_l1832_183256


namespace NUMINAMATH_GPT_sandy_correct_sums_l1832_183293

-- Definitions based on the conditions
variables (c i : ℕ)

-- Conditions as Lean statements
axiom h1 : 3 * c - 2 * i = 65
axiom h2 : c + i = 30

-- Proof goal
theorem sandy_correct_sums : c = 25 := 
by
  sorry

end NUMINAMATH_GPT_sandy_correct_sums_l1832_183293


namespace NUMINAMATH_GPT_second_train_cross_time_l1832_183289

noncomputable def time_to_cross_second_train : ℝ :=
  let length := 120
  let t1 := 10
  let t_cross := 13.333333333333334
  let v1 := length / t1
  let v_combined := 240 / t_cross
  let v2 := v_combined - v1
  length / v2

theorem second_train_cross_time :
  let t2 := time_to_cross_second_train
  t2 = 20 :=
by
  sorry

end NUMINAMATH_GPT_second_train_cross_time_l1832_183289


namespace NUMINAMATH_GPT_max_value_neg_a_inv_l1832_183231

theorem max_value_neg_a_inv (a : ℝ) (h : a < 0) : a + (1 / a) ≤ -2 := 
by
  sorry

end NUMINAMATH_GPT_max_value_neg_a_inv_l1832_183231


namespace NUMINAMATH_GPT_amanda_earnings_l1832_183295

def hourly_rate : ℝ := 20.00

def hours_monday : ℝ := 5 * 1.5

def hours_tuesday : ℝ := 3

def hours_thursday : ℝ := 2 * 2

def hours_saturday : ℝ := 6

def total_hours : ℝ := hours_monday + hours_tuesday + hours_thursday + hours_saturday

def total_earnings : ℝ := hourly_rate * total_hours

theorem amanda_earnings : total_earnings = 410.00 :=
by
  -- Proof steps can be filled here
  sorry

end NUMINAMATH_GPT_amanda_earnings_l1832_183295


namespace NUMINAMATH_GPT_range_of_a_l1832_183219

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x - a) * (x + 1 - a) >= 0 → x ≠ 1) ↔ (1 < a ∧ a < 2) := 
sorry

end NUMINAMATH_GPT_range_of_a_l1832_183219


namespace NUMINAMATH_GPT_borrowed_years_l1832_183243

noncomputable def principal : ℝ := 5396.103896103896
noncomputable def interest_rate : ℝ := 0.06
noncomputable def total_returned : ℝ := 8310

theorem borrowed_years :
  ∃ t : ℝ, (total_returned - principal) = principal * interest_rate * t ∧ t = 9 :=
by
  sorry

end NUMINAMATH_GPT_borrowed_years_l1832_183243


namespace NUMINAMATH_GPT_price_reduction_required_l1832_183274

variable (x : ℝ)
variable (profit_per_piece : ℝ := 40)
variable (initial_sales : ℝ := 20)
variable (additional_sales_per_unit_reduction : ℝ := 2)
variable (desired_profit : ℝ := 1200)

theorem price_reduction_required :
  (profit_per_piece - x) * (initial_sales + additional_sales_per_unit_reduction * x) = desired_profit → x = 20 :=
sorry

end NUMINAMATH_GPT_price_reduction_required_l1832_183274


namespace NUMINAMATH_GPT_roots_squared_sum_l1832_183236

theorem roots_squared_sum :
  (∀ x, x^2 + 2 * x - 8 = 0 → (x = x1 ∨ x = x2)) →
  x1 + x2 = -2 ∧ x1 * x2 = -8 →
  x1^2 + x2^2 = 20 :=
by
  intros roots_eq_sum_prod_eq
  sorry

end NUMINAMATH_GPT_roots_squared_sum_l1832_183236


namespace NUMINAMATH_GPT_investment_Y_l1832_183224

theorem investment_Y
  (X_investment : ℝ)
  (Y_investment : ℝ)
  (Z_investment : ℝ)
  (X_months : ℝ)
  (Y_months : ℝ)
  (Z_months : ℝ)
  (total_profit : ℝ)
  (Z_profit_share : ℝ)
  (h1 : X_investment = 36000)
  (h2 : Z_investment = 48000)
  (h3 : X_months = 12)
  (h4 : Y_months = 12)
  (h5 : Z_months = 8)
  (h6 : total_profit = 13970)
  (h7 : Z_profit_share = 4064) :
  Y_investment = 75000 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_investment_Y_l1832_183224


namespace NUMINAMATH_GPT_rational_square_of_one_minus_product_l1832_183287

theorem rational_square_of_one_minus_product (x y : ℚ) (h : x^5 + y^5 = 2 * x^2 * y^2) : 
  ∃ (q : ℚ), 1 - x * y = q^2 := 
by 
  sorry

end NUMINAMATH_GPT_rational_square_of_one_minus_product_l1832_183287


namespace NUMINAMATH_GPT_james_pays_37_50_l1832_183233

/-- 
James gets 20 singing lessons.
First lesson is free.
After the first 10 paid lessons, he only needs to pay for every other lesson.
Each lesson costs $5.
His uncle pays for half.
Prove that James pays $37.50.
--/

theorem james_pays_37_50 :
  let first_lessons := 1
  let total_lessons := 20
  let paid_lessons := 10
  let remaining_lessons := total_lessons - first_lessons - paid_lessons
  let paid_remaining_lessons := (remaining_lessons + 1) / 2
  let total_paid_lessons := paid_lessons + paid_remaining_lessons
  let cost_per_lesson := 5
  let total_payment := total_paid_lessons * cost_per_lesson
  let payment_by_james := total_payment / 2
  payment_by_james = 37.5 := 
by
  sorry

end NUMINAMATH_GPT_james_pays_37_50_l1832_183233


namespace NUMINAMATH_GPT_sum_of_interior_diagonals_l1832_183232

theorem sum_of_interior_diagonals (a b c : ℝ)
  (h₁ : 2 * (a * b + b * c + c * a) = 166)
  (h₂ : a + b + c = 16) :
  4 * Real.sqrt (a ^ 2 + b ^ 2 + c ^ 2) = 12 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_diagonals_l1832_183232


namespace NUMINAMATH_GPT_find_b_value_l1832_183271

/-- Given a line segment from point (0, b) to (8, 0) with a slope of -3/2, 
    prove that the value of b is 12. -/
theorem find_b_value (b : ℝ) : (8 - 0) ≠ 0 ∧ ((0 - b) / (8 - 0) = -3/2) → b = 12 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_find_b_value_l1832_183271


namespace NUMINAMATH_GPT_log_product_identity_l1832_183259

noncomputable def log {a b : ℝ} (ha : 1 < a) (hb : 0 < b) : ℝ := Real.log b / Real.log a

theorem log_product_identity : 
  log (by norm_num : (1 : ℝ) < 2) (by norm_num : (0 : ℝ) < 9) * 
  log (by norm_num : (1 : ℝ) < 3) (by norm_num : (0 : ℝ) < 8) = 6 :=
sorry

end NUMINAMATH_GPT_log_product_identity_l1832_183259


namespace NUMINAMATH_GPT_sum_fractions_bounds_l1832_183205

theorem sum_fractions_bounds {a b c : ℝ} (h : a * b * c = 1) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 < (a / (a + 1)) + (b / (b + 1)) + (c / (c + 1)) ∧ 
  (a / (a + 1)) + (b / (b + 1)) + (c / (c + 1)) < 2 :=
  sorry

end NUMINAMATH_GPT_sum_fractions_bounds_l1832_183205


namespace NUMINAMATH_GPT_div_by_5_implication_l1832_183264

theorem div_by_5_implication (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
    (h3 : ∃ k : ℕ, ab = 5 * k) : (∃ k : ℕ, a = 5 * k) ∨ (∃ k : ℕ, b = 5 * k) := 
by
  sorry

end NUMINAMATH_GPT_div_by_5_implication_l1832_183264


namespace NUMINAMATH_GPT_min_a_for_50_pow_2023_div_17_l1832_183285

theorem min_a_for_50_pow_2023_div_17 (a : ℕ) (h : 17 ∣ (50 ^ 2023 + a)) : a = 18 :=
sorry

end NUMINAMATH_GPT_min_a_for_50_pow_2023_div_17_l1832_183285


namespace NUMINAMATH_GPT_stock_price_percentage_increase_l1832_183268

theorem stock_price_percentage_increase :
  ∀ (total higher lower : ℕ), 
    total = 1980 →
    higher = 1080 →
    higher > lower →
    lower = total - higher →
  ((higher - lower) / lower : ℚ) * 100 = 20 :=
by
  intros total higher lower total_eq higher_eq higher_gt lower_eq
  sorry

end NUMINAMATH_GPT_stock_price_percentage_increase_l1832_183268


namespace NUMINAMATH_GPT_prime_dates_in_2008_l1832_183265

noncomputable def num_prime_dates_2008 : Nat := 52

theorem prime_dates_in_2008 : 
  let prime_days := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let prime_months_days := [(2, 29), (3, 31), (5, 31), (7, 31), (11, 30)]
  -- Count the prime days for each month considering the list
  let prime_day_count (days : Nat) := (prime_days.filter (λ d => d <= days)).length
  -- Sum the counts for each prime month
  (prime_months_days.map (λ (m, days) => prime_day_count days)).sum = num_prime_dates_2008 :=
by
  sorry

end NUMINAMATH_GPT_prime_dates_in_2008_l1832_183265


namespace NUMINAMATH_GPT_diagonals_in_15_sided_polygon_l1832_183284

def numberOfDiagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem diagonals_in_15_sided_polygon : numberOfDiagonals 15 = 90 := by
  sorry

end NUMINAMATH_GPT_diagonals_in_15_sided_polygon_l1832_183284


namespace NUMINAMATH_GPT_height_of_table_l1832_183216

/-- 
Given:
1. Combined initial measurement (l + h - w + t) = 40
2. Combined changed measurement (w + h - l + t) = 34
3. Width of each wood block (w) = 6 inches
4. Visible edge-on thickness of the table (t) = 4 inches
Prove:
The height of the table (h) is 33 inches.
-/
theorem height_of_table (l h t w : ℕ) (h_combined_initial : l + h - w + t = 40)
    (h_combined_changed : w + h - l + t = 34) (h_width : w = 6) (h_thickness : t = 4) : 
    h = 33 :=
by
  sorry

end NUMINAMATH_GPT_height_of_table_l1832_183216


namespace NUMINAMATH_GPT_number_of_persons_in_room_l1832_183228

theorem number_of_persons_in_room (n : ℕ) (h : n * (n - 1) / 2 = 78) : n = 13 :=
by
  /- We have:
     n * (n - 1) / 2 = 78,
     We need to prove n = 13 -/
  sorry

end NUMINAMATH_GPT_number_of_persons_in_room_l1832_183228


namespace NUMINAMATH_GPT_distinct_license_plates_count_l1832_183208

def num_digit_choices : Nat := 10
def num_letter_choices : Nat := 26
def num_digits : Nat := 5
def num_letters : Nat := 3

theorem distinct_license_plates_count :
  (num_digit_choices ^ num_digits) * (num_letter_choices ^ num_letters) = 1757600000 := 
sorry

end NUMINAMATH_GPT_distinct_license_plates_count_l1832_183208


namespace NUMINAMATH_GPT_hyperbola_symmetric_slopes_l1832_183280

/-- 
Let \(M(x_0, y_0)\) and \(N(-x_0, -y_0)\) be points symmetric about the origin on the hyperbola 
\(\frac{x^2}{16} - \frac{y^2}{4} = 1\). Let \(P(x, y)\) be any point on the hyperbola. 
When the slopes \(k_{PM}\) and \(k_{PN}\) both exist, then \(k_{PM} \cdot k_{PN} = \frac{1}{4}\),
independent of the position of \(P\).
-/
theorem hyperbola_symmetric_slopes (x x0 y y0: ℝ) 
  (hP: x^2 / 16 - y^2 / 4 = 1)
  (hM: x0^2 / 16 - y0^2 / 4 = 1)
  (h_slop_M : x ≠ x0)
  (h_slop_N : x ≠ x0):
  ((y - y0) / (x - x0)) * ((y + y0) / (x + x0)) = 1 / 4 := 
sorry

end NUMINAMATH_GPT_hyperbola_symmetric_slopes_l1832_183280


namespace NUMINAMATH_GPT_remainder_11_pow_1000_mod_500_l1832_183247

theorem remainder_11_pow_1000_mod_500 : (11 ^ 1000) % 500 = 1 :=
by
  have h1 : 11 % 5 = 1 := by norm_num
  have h2 : (11 ^ 10) % 100 = 1 := by
    -- Some steps omitted to satisfy conditions; normally would be generalized
    sorry
  have h3 : 500 = 5 * 100 := by norm_num
  -- Further omitted steps aligning with the Chinese Remainder Theorem application.
  sorry

end NUMINAMATH_GPT_remainder_11_pow_1000_mod_500_l1832_183247


namespace NUMINAMATH_GPT_solve_fractional_eq_l1832_183225

noncomputable def fractional_eq (x : ℝ) : Prop := 
  (3 / (x^2 - 3 * x) + (x - 1) / (x - 3) = 1)

noncomputable def not_zero_denom (x : ℝ) : Prop := 
  (x^2 - 3 * x ≠ 0) ∧ (x - 3 ≠ 0)

theorem solve_fractional_eq : fractional_eq (-3/2) ∧ not_zero_denom (-3/2) :=
by
  sorry

end NUMINAMATH_GPT_solve_fractional_eq_l1832_183225


namespace NUMINAMATH_GPT_minimum_area_integer_triangle_l1832_183217

theorem minimum_area_integer_triangle :
  ∃ (p q : ℤ), p ≠ 0 ∧ q ≠ 0 ∧ (∃ (p q : ℤ), 2 ∣ (16 * p - 30 * q)) 
  → (∃ (area : ℝ), area = (1/2 : ℝ) * |16 * p - 30 * q| ∧ area = 1) :=
by
  sorry

end NUMINAMATH_GPT_minimum_area_integer_triangle_l1832_183217


namespace NUMINAMATH_GPT_thousands_digit_is_0_or_5_l1832_183249

theorem thousands_digit_is_0_or_5 (n t : ℕ) (h₁ : n > 1000000) (h₂ : n % 40 = t) (h₃ : n % 625 = t) : 
  ((n / 1000) % 10 = 0) ∨ ((n / 1000) % 10 = 5) :=
sorry

end NUMINAMATH_GPT_thousands_digit_is_0_or_5_l1832_183249


namespace NUMINAMATH_GPT_x_minus_y_eq_14_l1832_183260

theorem x_minus_y_eq_14 (x y : ℝ) (h : x^2 + y^2 = 16 * x - 12 * y + 100) : x - y = 14 :=
sorry

end NUMINAMATH_GPT_x_minus_y_eq_14_l1832_183260


namespace NUMINAMATH_GPT_second_investment_rate_l1832_183282

theorem second_investment_rate (P : ℝ) (r₁ t : ℝ) (I_diff : ℝ) (P900 : P = 900) (r1_4_percent : r₁ = 0.04) (t7 : t = 7) (I_years : I_diff = 31.50) :
∃ r₂ : ℝ, 900 * (r₂ / 100) * 7 - 900 * 0.04 * 7 = 31.50 → r₂ = 4.5 := 
by
  sorry

end NUMINAMATH_GPT_second_investment_rate_l1832_183282


namespace NUMINAMATH_GPT_rational_smaller_than_neg_half_l1832_183257

theorem rational_smaller_than_neg_half : ∃ q : ℚ, q < -1/2 := by
  use (-1 : ℚ)
  sorry

end NUMINAMATH_GPT_rational_smaller_than_neg_half_l1832_183257


namespace NUMINAMATH_GPT_value_of_a_plus_c_l1832_183272

theorem value_of_a_plus_c (a b c r : ℝ)
  (h1 : a + b + c = 114)
  (h2 : a * b * c = 46656)
  (h3 : b = a * r)
  (h4 : c = a * r^2) :
  a + c = 78 :=
sorry

end NUMINAMATH_GPT_value_of_a_plus_c_l1832_183272


namespace NUMINAMATH_GPT_value_at_minus_two_l1832_183223

def f (x : ℝ) (a b c : ℝ) := a * x^5 + b * x^3 + c * x + 1

theorem value_at_minus_two (a b c : ℝ) (h : f 2 a b c = -1) : f (-2) a b c = 3 := by
  sorry

end NUMINAMATH_GPT_value_at_minus_two_l1832_183223
