import Mathlib

namespace NUMINAMATH_GPT_find_percentage_l172_17255

variable (P x : ℝ)

theorem find_percentage (h1 : x = 10)
    (h2 : (P / 100) * x = 0.05 * 500 - 20) : P = 50 := by
  sorry

end NUMINAMATH_GPT_find_percentage_l172_17255


namespace NUMINAMATH_GPT_new_mean_rent_is_880_l172_17268

theorem new_mean_rent_is_880
  (num_friends : ℕ)
  (initial_average_rent : ℝ)
  (increase_percentage : ℝ)
  (original_rent_increased : ℝ)
  (new_mean_rent : ℝ) :
  num_friends = 4 →
  initial_average_rent = 800 →
  increase_percentage = 20 →
  original_rent_increased = 1600 →
  new_mean_rent = 880 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_new_mean_rent_is_880_l172_17268


namespace NUMINAMATH_GPT_range_of_a_l172_17240

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2 + (a - 1) * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x, 1 < x ∧ x < 4 → deriv (f a) x < 0) ∧
  (∀ x, 6 < x → deriv (f a) x > 0) →
  5 ≤ a ∧ a ≤ 7 :=
sorry

end NUMINAMATH_GPT_range_of_a_l172_17240


namespace NUMINAMATH_GPT_probability_both_A_and_B_selected_l172_17290

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end NUMINAMATH_GPT_probability_both_A_and_B_selected_l172_17290


namespace NUMINAMATH_GPT_floor_expression_bounds_l172_17237

theorem floor_expression_bounds (x : ℝ) (h : ⌊x * ⌊x / 2⌋⌋ = 12) : 
  4.9 ≤ x ∧ x < 5.1 :=
sorry

end NUMINAMATH_GPT_floor_expression_bounds_l172_17237


namespace NUMINAMATH_GPT_maximize_profit_l172_17262

-- Conditions
def price_bound (p : ℝ) := p ≤ 22
def books_sold (p : ℝ) := 110 - 4 * p
def profit (p : ℝ) := (p - 2) * books_sold p

-- The main theorem statement
theorem maximize_profit : ∃ p : ℝ, price_bound p ∧ profit p = profit 15 :=
sorry

end NUMINAMATH_GPT_maximize_profit_l172_17262


namespace NUMINAMATH_GPT_exponentiation_problem_l172_17235

theorem exponentiation_problem : (8^8 / 8^5) * 2^10 * 2^3 = 2^22 := by
  sorry

end NUMINAMATH_GPT_exponentiation_problem_l172_17235


namespace NUMINAMATH_GPT_greatest_possible_sum_of_digits_l172_17261

theorem greatest_possible_sum_of_digits 
  (n : ℕ) (a b d : ℕ) 
  (h_a : a ≠ 0) (h_b : b ≠ 0) (h_d : d ≠ 0)
  (h1 : ∃ n1 n2 : ℕ, n1 ≠ n2 ∧ (d * ((10 ^ (3 * n1) - 1) / 9) - b * ((10 ^ n1 - 1) / 9) = a^3 * ((10^n1 - 1) / 9)^3) 
                      ∧ (d * ((10 ^ (3 * n2) - 1) / 9) - b * ((10 ^ n2 - 1) / 9) = a^3 * ((10^n2 - 1) / 9)^3)) : 
  a + b + d = 12 := 
sorry

end NUMINAMATH_GPT_greatest_possible_sum_of_digits_l172_17261


namespace NUMINAMATH_GPT_quotient_of_numbers_l172_17275

noncomputable def larger_number : ℕ := 22
noncomputable def smaller_number : ℕ := 8

theorem quotient_of_numbers : (larger_number.toFloat / smaller_number.toFloat) = 2.75 := by
  sorry

end NUMINAMATH_GPT_quotient_of_numbers_l172_17275


namespace NUMINAMATH_GPT_shipping_cost_per_unit_l172_17298

-- Define the conditions
def cost_per_component : ℝ := 80
def fixed_monthly_cost : ℝ := 16500
def num_components : ℝ := 150
def lowest_selling_price : ℝ := 196.67

-- Define the revenue and total cost
def total_cost (S : ℝ) : ℝ := (cost_per_component * num_components) + fixed_monthly_cost + (num_components * S)
def total_revenue : ℝ := lowest_selling_price * num_components

-- Define the proposition to be proved
theorem shipping_cost_per_unit (S : ℝ) :
  total_cost S ≤ total_revenue → S ≤ 6.67 :=
by sorry

end NUMINAMATH_GPT_shipping_cost_per_unit_l172_17298


namespace NUMINAMATH_GPT_tom_age_ratio_l172_17221

-- Definitions of given conditions
variables (T N : ℕ) -- Tom's age (T) and number of years ago (N)

-- Tom's age is T years
-- The sum of the ages of Tom's three children is also T
-- N years ago, Tom's age was twice the sum of his children's ages then

theorem tom_age_ratio (h1 : T - N = 2 * (T - 3 * N)) : T / N = 5 :=
sorry

end NUMINAMATH_GPT_tom_age_ratio_l172_17221


namespace NUMINAMATH_GPT_fifteen_percent_of_x_equals_sixty_l172_17266

theorem fifteen_percent_of_x_equals_sixty (x : ℝ) (h : 0.15 * x = 60) : x = 400 :=
by
  sorry

end NUMINAMATH_GPT_fifteen_percent_of_x_equals_sixty_l172_17266


namespace NUMINAMATH_GPT_tan_nine_pi_over_three_l172_17242

theorem tan_nine_pi_over_three : Real.tan (9 * Real.pi / 3) = 0 := by
  sorry

end NUMINAMATH_GPT_tan_nine_pi_over_three_l172_17242


namespace NUMINAMATH_GPT_adoption_days_l172_17260

def initial_puppies : ℕ := 15
def additional_puppies : ℕ := 62
def adoption_rate : ℕ := 7

def total_puppies : ℕ := initial_puppies + additional_puppies

theorem adoption_days :
  total_puppies / adoption_rate = 11 :=
by
  sorry

end NUMINAMATH_GPT_adoption_days_l172_17260


namespace NUMINAMATH_GPT_ellipse_equation_l172_17271

-- Definitions based on the problem conditions
def hyperbola_foci (x y : ℝ) : Prop := 2 * x^2 - 2 * y^2 = 1
def passes_through_point (p : ℝ × ℝ) (x y : ℝ) : Prop := p = (1, -3 / 2)

-- The statement to be proved
theorem ellipse_equation (c : ℝ) (a b : ℝ) :
    hyperbola_foci (-1) 0 ∧ hyperbola_foci 1 0 ∧
    passes_through_point (1, -3 / 2) 1 (-3 / 2) ∧
    (a = 2) ∧ (b = Real.sqrt 3) ∧ (c = 1)
    → ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_equation_l172_17271


namespace NUMINAMATH_GPT_circumcenter_distance_two_l172_17231

noncomputable def distance_between_circumcenter (A B C M : ℝ × ℝ)
  (hAB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 25)
  (hBC : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 17)
  (hAC : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 16)
  (hM_on_AC : M.1 = C.1 - 1 ∧ M.2 = C.2)
  (hCM : (M.1 - C.1)^2 + (M.2 - C.2)^2 = 1)
  : ℝ :=
dist ( ( (A.1 + B.1) / 2, (A.2 + B.2) / 2 ) ) ( ( (B.1 + C.1) / 2, (B.2 + C.2) / 2 )) 

theorem circumcenter_distance_two (A B C M : ℝ × ℝ)
  (hAB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 25)
  (hBC : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 17)
  (hAC : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 16)
  (hM_on_AC : M.1 = C.1 - 1 ∧ M.2 = C.2)
  (hCM : (M.1 - C.1)^2 + (M.2 - C.2)^2 = 1) 
  : distance_between_circumcenter A B C M hAB hBC hAC hM_on_AC hCM = 2 :=
sorry

end NUMINAMATH_GPT_circumcenter_distance_two_l172_17231


namespace NUMINAMATH_GPT_geometric_sequence_third_fourth_terms_l172_17291

theorem geometric_sequence_third_fourth_terms
  (a : ℕ → ℝ)
  (r : ℝ)
  (ha : ∀ n, a (n + 1) = r * a n)
  (hS2 : a 0 + a 1 = 3 * a 1) :
  (a 2 + a 3) / (a 0 + a 1) = 1 / 4 :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_geometric_sequence_third_fourth_terms_l172_17291


namespace NUMINAMATH_GPT_area_shaded_region_is_correct_l172_17278

noncomputable def radius_of_larger_circle : ℝ := 8
noncomputable def radius_of_smaller_circle := radius_of_larger_circle / 2

-- Define areas
noncomputable def area_of_larger_circle := Real.pi * radius_of_larger_circle ^ 2
noncomputable def area_of_smaller_circle := Real.pi * radius_of_smaller_circle ^ 2
noncomputable def total_area_of_smaller_circles := 2 * area_of_smaller_circle
noncomputable def area_of_shaded_region := area_of_larger_circle - total_area_of_smaller_circles

-- Prove that the area of the shaded region is 32π
theorem area_shaded_region_is_correct : area_of_shaded_region = 32 * Real.pi := by
  sorry

end NUMINAMATH_GPT_area_shaded_region_is_correct_l172_17278


namespace NUMINAMATH_GPT_dynaco_shares_l172_17213

theorem dynaco_shares (M D : ℕ) 
  (h1 : M + D = 300)
  (h2 : 36 * M + 44 * D = 12000) : 
  D = 150 :=
sorry

end NUMINAMATH_GPT_dynaco_shares_l172_17213


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l172_17225

theorem boat_speed_in_still_water (B S : ℝ) (h1 : B + S = 6) (h2 : B - S = 4) : B = 5 := by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l172_17225


namespace NUMINAMATH_GPT_sequence_inequality_l172_17289

theorem sequence_inequality (a : ℕ → ℝ) (h1 : ∀ n m : ℕ, a (n + m) ≤ a n + a m)
  (h2 : ∀ n : ℕ, 0 ≤ a n) (n m : ℕ) (hnm : n ≥ m) : 
  a n ≤ m * a 1 + (n / m - 1) * a m :=
sorry

end NUMINAMATH_GPT_sequence_inequality_l172_17289


namespace NUMINAMATH_GPT_hou_yi_score_l172_17254

theorem hou_yi_score (a b c : ℕ) (h1 : 2 * b + c = 29) (h2 : 2 * a + c = 43) : a + b + c = 36 := 
by 
  sorry

end NUMINAMATH_GPT_hou_yi_score_l172_17254


namespace NUMINAMATH_GPT_train_length_l172_17259

theorem train_length (L : ℝ) :
  (20 * (L + 160) = 15 * (L + 250)) -> L = 110 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_train_length_l172_17259


namespace NUMINAMATH_GPT_B_can_finish_alone_in_27_5_days_l172_17281

-- Definitions of work rates
variable (B A C : Type)

-- Conditions translations
def efficiency_of_A (x : ℝ) : Prop := ∀ (work_rate_A work_rate_B : ℝ), work_rate_A = 1 / (2 * x) ∧ work_rate_B = 1 / x
def efficiency_of_C (x : ℝ) : Prop := ∀ (work_rate_C work_rate_B : ℝ), work_rate_C = 1 / (3 * x) ∧ work_rate_B = 1 / x
def combined_work_rate (x : ℝ) : Prop := (1 / (2 * x) + 1 / x + 1 / (3 * x)) = 1 / 15

-- Proof statement
theorem B_can_finish_alone_in_27_5_days :
  ∃ (x : ℝ), efficiency_of_A x ∧ efficiency_of_C x ∧ combined_work_rate x ∧ x = 27.5 :=
sorry

end NUMINAMATH_GPT_B_can_finish_alone_in_27_5_days_l172_17281


namespace NUMINAMATH_GPT_rectangle_shorter_side_l172_17248

theorem rectangle_shorter_side
  (x : ℝ)
  (a b d : ℝ)
  (h₁ : a = 3 * x)
  (h₂ : b = 4 * x)
  (h₃ : d = 9) :
  a = 5.4 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_shorter_side_l172_17248


namespace NUMINAMATH_GPT_pentomino_symmetry_count_l172_17215

def is_pentomino (shape : Type) : Prop :=
  -- Define the property of being a pentomino as composed of five squares edge to edge
  sorry

def has_reflectional_symmetry (shape : Type) : Prop :=
  -- Define the property of having at least one line of reflectional symmetry
  sorry

def has_rotational_symmetry_of_order_2 (shape : Type) : Prop :=
  -- Define the property of having rotational symmetry of order 2 (180 degrees rotation results in the same shape)
  sorry

noncomputable def count_valid_pentominoes : Nat :=
  -- Assume that we have a list of 18 pentominoes
  -- Count the number of pentominoes that meet both criteria
  sorry

theorem pentomino_symmetry_count :
  count_valid_pentominoes = 4 :=
sorry

end NUMINAMATH_GPT_pentomino_symmetry_count_l172_17215


namespace NUMINAMATH_GPT_find_y_l172_17287

-- Let s be the result of tripling both the base and exponent of c^d
-- Given the condition s = c^d * y^d, we need to prove y = 27c^2

variable (c d y : ℝ)
variable (h_d : d > 0)
variable (h : (3 * c)^(3 * d) = c^d * y^d)

theorem find_y (h_d : d > 0) (h : (3 * c)^(3 * d) = c^d * y^d) : y = 27 * c ^ 2 :=
by sorry

end NUMINAMATH_GPT_find_y_l172_17287


namespace NUMINAMATH_GPT_parallel_lines_necessary_not_sufficient_l172_17207

variables {a1 b1 a2 b2 c1 c2 : ℝ}

def determinant (a1 b1 a2 b2 : ℝ) : ℝ := a1 * b2 - a2 * b1

theorem parallel_lines_necessary_not_sufficient
  (h1 : a1^2 + b1^2 ≠ 0)
  (h2 : a2^2 + b2^2 ≠ 0)
  : (determinant a1 b1 a2 b2 = 0) → 
    (a1 * x + b1 * y + c1 = 0 ∧ a2 * x + b2 * y + c2 =0 → exists k : ℝ, (a1 = k ∧ b1 = k)) ∧ 
    (determinant a1 b1 a2 b2 = 0 → (a2 * x + b2 * y + c2 = a1 * x + b1 * y + c1 → false)) :=
sorry

end NUMINAMATH_GPT_parallel_lines_necessary_not_sufficient_l172_17207


namespace NUMINAMATH_GPT_remainder_when_divided_by_100_l172_17200

theorem remainder_when_divided_by_100 (n : ℤ) (h : ∃ a : ℤ, n = 100 * a - 1) : 
  (n^3 + n^2 + 2 * n + 3) % 100 = 1 :=
by 
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_100_l172_17200


namespace NUMINAMATH_GPT_train_speed_l172_17232

theorem train_speed (length time : ℝ) (h_length : length = 120) (h_time : time = 11.999040076793857) :
  (length / time) * 3.6 = 36.003 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l172_17232


namespace NUMINAMATH_GPT_tim_initial_balls_correct_l172_17267

-- Defining the initial number of balls Robert had
def robert_initial_balls : ℕ := 25

-- Defining the final number of balls Robert had
def robert_final_balls : ℕ := 45

-- Defining the number of balls Tim had initially
def tim_initial_balls := 40

-- Now, we state the proof problem:
theorem tim_initial_balls_correct :
  robert_initial_balls + (tim_initial_balls / 2) = robert_final_balls :=
by
  -- This is the part where you typically write the proof.
  -- However, we put sorry here because the task does not require the proof itself.
  sorry

end NUMINAMATH_GPT_tim_initial_balls_correct_l172_17267


namespace NUMINAMATH_GPT_diminish_value_l172_17244

theorem diminish_value (a b : ℕ) (h1 : a = 1015) (h2 : b = 12) (h3 : b = 16) (h4 : b = 18) (h5 : b = 21) (h6 : b = 28) :
  ∃ k, a - k = lcm (lcm (lcm b b) (lcm b b)) (lcm b b) ∧ k = 7 :=
sorry

end NUMINAMATH_GPT_diminish_value_l172_17244


namespace NUMINAMATH_GPT_integer_remainder_18_l172_17218

theorem integer_remainder_18 (n : ℤ) (h : n ∈ ({14, 15, 16, 17, 18} : Set ℤ)) : n % 7 = 4 :=
by
  sorry

end NUMINAMATH_GPT_integer_remainder_18_l172_17218


namespace NUMINAMATH_GPT_find_b_l172_17252

-- Define the number 1234567 in base 36
def numBase36 : ℤ := 1 * 36^6 + 2 * 36^5 + 3 * 36^4 + 4 * 36^3 + 5 * 36^2 + 6 * 36^1 + 7 * 36^0

-- Prove that for b being an integer such that 0 ≤ b ≤ 10,
-- and given (numBase36 - b) is a multiple of 17, b must be 0
theorem find_b (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 10) (h3 : (numBase36 - b) % 17 = 0) : b = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l172_17252


namespace NUMINAMATH_GPT_perpendicular_k_value_exists_l172_17274

open Real EuclideanSpace

def vector_a : ℝ × ℝ := (-2, 1)
def vector_b : ℝ × ℝ := (3, 2)

theorem perpendicular_k_value_exists : ∃ k : ℝ, (vector_a.1 * (vector_a.1 + k * vector_b.1) + vector_a.2 * (vector_a.2 + k * vector_b.2) = 0) ∧ k = 5 / 4 := by
  sorry

end NUMINAMATH_GPT_perpendicular_k_value_exists_l172_17274


namespace NUMINAMATH_GPT_intersection_eq_union_eq_complement_union_eq_intersection_complements_eq_l172_17272

-- Definitions for U, A, B
def U := { x : ℤ | 0 < x ∧ x <= 10 }
def A : Set ℤ := { 1, 2, 4, 5, 9 }
def B : Set ℤ := { 4, 6, 7, 8, 10 }

-- 1. Prove A ∩ B = {4}
theorem intersection_eq : A ∩ B = {4} := by
  sorry

-- 2. Prove A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}
theorem union_eq : A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10} := by
  sorry

-- 3. Prove complement_U (A ∪ B) = {3}
def complement_U (s : Set ℤ) : Set ℤ := { x ∈ U | ¬ (x ∈ s) }
theorem complement_union_eq : complement_U (A ∪ B) = {3} := by
  sorry

-- 4. Prove (complement_U A) ∩ (complement_U B) = {3}
theorem intersection_complements_eq : (complement_U A) ∩ (complement_U B) = {3} := by
  sorry

end NUMINAMATH_GPT_intersection_eq_union_eq_complement_union_eq_intersection_complements_eq_l172_17272


namespace NUMINAMATH_GPT_hcf_of_210_and_671_l172_17288

theorem hcf_of_210_and_671 :
  let lcm := 2310
  let a := 210
  let b := 671
  gcd a b = 61 :=
by
  let lcm := 2310
  let a := 210
  let b := 671
  let hcf := gcd a b
  have rel : lcm * hcf = a * b := by sorry
  have hcf_eq : hcf = 61 := by sorry
  exact hcf_eq

end NUMINAMATH_GPT_hcf_of_210_and_671_l172_17288


namespace NUMINAMATH_GPT_diamonds_in_F10_l172_17299

def diamonds_in_figure (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 1 + 3 * (Nat.add (Nat.mul (n - 1) n) 0) / 2

theorem diamonds_in_F10 : diamonds_in_figure 10 = 136 :=
by
  sorry

end NUMINAMATH_GPT_diamonds_in_F10_l172_17299


namespace NUMINAMATH_GPT_center_of_square_l172_17284

theorem center_of_square (O : ℝ × ℝ) (A B C D : ℝ × ℝ) 
  (hAB : dist A B = 1) 
  (hA : A = (0, 0)) 
  (hB : B = (1, 0)) 
  (hC : C = (1, 1)) 
  (hD : D = (0, 1)) 
  (h_sum_squares : (dist O A)^2 + (dist O B)^2 + (dist O C)^2 + (dist O D)^2 = 2): 
  O = (1/2, 1/2) :=
by sorry

end NUMINAMATH_GPT_center_of_square_l172_17284


namespace NUMINAMATH_GPT_percentage_mr_william_land_l172_17202

theorem percentage_mr_william_land 
  (T W : ℝ) -- Total taxable land of the village and the total land of Mr. William
  (tax_collected_village : ℝ) -- Total tax collected from the village
  (tax_paid_william : ℝ) -- Tax paid by Mr. William
  (h1 : tax_collected_village = 3840) 
  (h2 : tax_paid_william = 480) 
  (h3 : (480 / 3840) = (25 / 100) * (W / T)) 
: (W / T) * 100 = 12.5 :=
by sorry

end NUMINAMATH_GPT_percentage_mr_william_land_l172_17202


namespace NUMINAMATH_GPT_find_m_perpendicular_l172_17241

-- Define the two vectors
def a (m : ℝ) : ℝ × ℝ := (m, -1)
def b : ℝ × ℝ := (1, 2)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Theorem stating the mathematically equivalent proof problem
theorem find_m_perpendicular (m : ℝ) (h : dot_product (a m) b = 0) : m = 2 :=
by sorry

end NUMINAMATH_GPT_find_m_perpendicular_l172_17241


namespace NUMINAMATH_GPT_rate_of_markup_l172_17227

theorem rate_of_markup (S : ℝ) (hS : S = 8)
  (profit_percent : ℝ) (h_profit_percent : profit_percent = 0.20)
  (expense_percent : ℝ) (h_expense_percent : expense_percent = 0.10) :
  (S - (S * (1 - profit_percent - expense_percent))) / (S * (1 - profit_percent - expense_percent)) * 100 = 42.857 :=
by
  sorry

end NUMINAMATH_GPT_rate_of_markup_l172_17227


namespace NUMINAMATH_GPT_lily_profit_is_correct_l172_17250

-- Define the conditions
def first_ticket_price : ℕ := 1
def price_increment : ℕ := 1
def number_of_tickets : ℕ := 5
def prize_amount : ℕ := 11

-- Define the sum of arithmetic series formula
def total_amount_collected (n : ℕ) (a : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Calculate the total amount collected
def total : ℕ := total_amount_collected number_of_tickets first_ticket_price price_increment

-- Define the profit calculation
def profit : ℕ := total - prize_amount

-- The statement we need to prove
theorem lily_profit_is_correct : profit = 4 := by
  sorry

end NUMINAMATH_GPT_lily_profit_is_correct_l172_17250


namespace NUMINAMATH_GPT_rectangle_area_l172_17229

theorem rectangle_area (length : ℝ) (width : ℝ) (area : ℝ) 
  (h1 : length = 24) 
  (h2 : width = 0.875 * length) 
  (h3 : area = length * width) : 
  area = 504 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l172_17229


namespace NUMINAMATH_GPT_find_m_value_l172_17285

theorem find_m_value : 
  ∃ (m : ℝ), 
  (∃ (x y : ℝ), (x - 2)^2 + (y + 1)^2 = 1 ∧ (x - y + m = 0)) → m = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l172_17285


namespace NUMINAMATH_GPT_original_weight_of_beef_l172_17277

theorem original_weight_of_beef (w_after : ℝ) (loss_percentage : ℝ) (w_before : ℝ) : 
  (w_after = 550) → (loss_percentage = 0.35) → (w_after = 550) → (w_before = 846.15) :=
by
  intros
  sorry

end NUMINAMATH_GPT_original_weight_of_beef_l172_17277


namespace NUMINAMATH_GPT_partners_in_firm_l172_17228

theorem partners_in_firm (P A : ℕ) (h1 : P * 63 = 2 * A) (h2 : P * 34 = 1 * (A + 45)) : P = 18 :=
by
  sorry

end NUMINAMATH_GPT_partners_in_firm_l172_17228


namespace NUMINAMATH_GPT_z_is_real_iff_z_is_complex_iff_z_is_pure_imaginary_iff_l172_17276

def is_real (z : ℂ) := z.im = 0
def is_complex (z : ℂ) := z.im ≠ 0
def is_pure_imaginary (z : ℂ) := z.re = 0 ∧ z.im ≠ 0

def z (m : ℝ) : ℂ := ⟨m - 3, m^2 - 2 * m - 15⟩

theorem z_is_real_iff (m : ℝ) : is_real (z m) ↔ m = -3 ∨ m = 5 :=
by sorry

theorem z_is_complex_iff (m : ℝ) : is_complex (z m) ↔ m ≠ -3 ∧ m ≠ 5 :=
by sorry

theorem z_is_pure_imaginary_iff (m : ℝ) : is_pure_imaginary (z m) ↔ m = 3 :=
by sorry

end NUMINAMATH_GPT_z_is_real_iff_z_is_complex_iff_z_is_pure_imaginary_iff_l172_17276


namespace NUMINAMATH_GPT_perfect_cubes_not_divisible_by_10_l172_17279

-- Definitions based on conditions
def is_divisible_by_10 (n : ℕ) : Prop := 10 ∣ n
def is_perfect_cube (n : ℕ) : Prop := ∃ (k : ℕ), n = k ^ 3
def erase_last_three_digits (n : ℕ) : ℕ := n / 1000

-- Main statement
theorem perfect_cubes_not_divisible_by_10 (x : ℕ) :
  is_perfect_cube x ∧ ¬ is_divisible_by_10 x ∧ is_perfect_cube (erase_last_three_digits x) →
  x = 1331 ∨ x = 1728 :=
by
  sorry

end NUMINAMATH_GPT_perfect_cubes_not_divisible_by_10_l172_17279


namespace NUMINAMATH_GPT_bottles_needed_exceed_initial_l172_17219

-- Define the initial conditions and their relationships
def initial_bottles : ℕ := 4 * 12 -- four dozen bottles

def bottles_first_break (players : ℕ) (bottles_per_player : ℕ) : ℕ :=
  players * bottles_per_player

def bottles_second_break (total_players : ℕ) (bottles_per_player : ℕ) (exhausted_players : ℕ) (extra_bottles : ℕ) : ℕ :=
  total_players * bottles_per_player + exhausted_players * extra_bottles

def bottles_third_break (remaining_players : ℕ) (bottles_per_player : ℕ) : ℕ :=
  remaining_players * bottles_per_player

-- Prove that the bottles needed exceed the initial amount by 4
theorem bottles_needed_exceed_initial : 
  bottles_first_break 11 2 + bottles_second_break 14 1 4 1 + bottles_third_break 12 1 = initial_bottles + 4 :=
by
  -- Proof will be completed here
  sorry

end NUMINAMATH_GPT_bottles_needed_exceed_initial_l172_17219


namespace NUMINAMATH_GPT_janet_speed_l172_17214

def janet_sister_speed : ℝ := 12
def lake_width : ℝ := 60
def wait_time : ℝ := 3

theorem janet_speed :
  (lake_width / (lake_width / janet_sister_speed - wait_time)) = 30 := 
sorry

end NUMINAMATH_GPT_janet_speed_l172_17214


namespace NUMINAMATH_GPT_value_of_x_that_makes_sqrt_undefined_l172_17265

theorem value_of_x_that_makes_sqrt_undefined (x : ℕ) (hpos : 0 < x) : (x = 1) ∨ (x = 2) ↔ (x - 3 < 0) := by
  sorry

end NUMINAMATH_GPT_value_of_x_that_makes_sqrt_undefined_l172_17265


namespace NUMINAMATH_GPT_arithmetic_sequence_term_13_l172_17223

variable {a : ℕ → ℝ}
variable {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_term_13 (h_arith : arithmetic_sequence a d)
  (h_a5 : a 5 = 3)
  (h_a9 : a 9 = 6) :
  a 13 = 9 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_13_l172_17223


namespace NUMINAMATH_GPT_x_is_sufficient_but_not_necessary_for_x_squared_eq_one_l172_17217

theorem x_is_sufficient_but_not_necessary_for_x_squared_eq_one : 
  (∀ x : ℝ, x = 1 → x^2 = 1) ∧ (∃ x : ℝ, x^2 = 1 ∧ x ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_x_is_sufficient_but_not_necessary_for_x_squared_eq_one_l172_17217


namespace NUMINAMATH_GPT_relationship_of_y1_y2_l172_17280

theorem relationship_of_y1_y2 (y1 y2 : ℝ) : 
  (∃ y1 y2, (y1 = 2 / -2) ∧ (y2 = 2 / -1)) → (y1 > y2) :=
by
  sorry

end NUMINAMATH_GPT_relationship_of_y1_y2_l172_17280


namespace NUMINAMATH_GPT_van_helsing_removed_percentage_l172_17243

theorem van_helsing_removed_percentage :
  ∀ (V W : ℕ), 
  (5 * V / 2 + 10 * 8 = 105) →
  (W = 4 * V) →
  8 / W * 100 = 20 := 
by
  sorry

end NUMINAMATH_GPT_van_helsing_removed_percentage_l172_17243


namespace NUMINAMATH_GPT_cost_price_of_product_is_100_l172_17216

theorem cost_price_of_product_is_100 
  (x : ℝ) 
  (h : x * 1.2 * 0.9 - x = 8) : 
  x = 100 := 
sorry

end NUMINAMATH_GPT_cost_price_of_product_is_100_l172_17216


namespace NUMINAMATH_GPT_number_of_ways_to_take_one_ball_from_pockets_number_of_ways_to_take_one_ball_each_from_pockets_l172_17208

-- Let's define the conditions.
def balls_in_first_pocket : Nat := 2
def balls_in_second_pocket : Nat := 4
def balls_in_third_pocket : Nat := 5

-- Proof for the first question
theorem number_of_ways_to_take_one_ball_from_pockets : 
  balls_in_first_pocket + balls_in_second_pocket + balls_in_third_pocket = 11 := 
by
  sorry

-- Proof for the second question
theorem number_of_ways_to_take_one_ball_each_from_pockets : 
  balls_in_first_pocket * balls_in_second_pocket * balls_in_third_pocket = 40 := 
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_take_one_ball_from_pockets_number_of_ways_to_take_one_ball_each_from_pockets_l172_17208


namespace NUMINAMATH_GPT_equilateral_triangles_count_in_grid_of_side_4_l172_17204

-- Define a function to calculate the number of equilateral triangles in a triangular grid of side length n
def countEquilateralTriangles (n : ℕ) : ℕ :=
  (n * (n + 1) * (n + 2) * (n + 3)) / 24

-- Define the problem statement for n = 4
theorem equilateral_triangles_count_in_grid_of_side_4 :
  countEquilateralTriangles 4 = 35 := by
  sorry

end NUMINAMATH_GPT_equilateral_triangles_count_in_grid_of_side_4_l172_17204


namespace NUMINAMATH_GPT_problem1_problem2_l172_17211

open Set

variable {U : Set ℝ} (A B : Set ℝ)

def UA : U = univ := by sorry
def A_def : A = { x : ℝ | 0 < x ∧ x ≤ 2 } := by sorry
def B_def : B = { x : ℝ | x < -3 ∨ x > 1 } := by sorry

theorem problem1 : A ∩ B = { x : ℝ | 1 < x ∧ x ≤ 2 } := 
by sorry

theorem problem2 : (U \ A) ∩ (U \ B) = { x : ℝ | -3 ≤ x ∧ x ≤ 0 } := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_l172_17211


namespace NUMINAMATH_GPT_circular_board_area_l172_17226

theorem circular_board_area (C : ℝ) (R T : ℝ) (h1 : R = 62.8) (h2 : T = 10) (h3 : C = R / T) (h4 : C = 2 * Real.pi) : 
  ∀ r A : ℝ, (r = C / (2 * Real.pi)) → (A = Real.pi * r^2)  → A = Real.pi :=
by
  intro r A
  intro hr hA
  sorry

end NUMINAMATH_GPT_circular_board_area_l172_17226


namespace NUMINAMATH_GPT_seating_arrangement_l172_17239

theorem seating_arrangement : 
  ∃ x y z : ℕ, 
  7 * x + 8 * y + 9 * z = 65 ∧ z = 1 ∧ x + y + z = r :=
sorry

end NUMINAMATH_GPT_seating_arrangement_l172_17239


namespace NUMINAMATH_GPT_first_discount_percentage_l172_17230

theorem first_discount_percentage (x : ℕ) :
  let original_price := 175
  let discounted_price := original_price * (100 - x) / 100
  let final_price := discounted_price * 95 / 100
  final_price = 133 → x = 20 :=
by
  sorry

end NUMINAMATH_GPT_first_discount_percentage_l172_17230


namespace NUMINAMATH_GPT_smallest_y_value_l172_17234

theorem smallest_y_value :
  ∃ y : ℝ, (3 * y ^ 2 + 33 * y - 90 = y * (y + 16)) ∧ ∀ z : ℝ, (3 * z ^ 2 + 33 * z - 90 = z * (z + 16)) → y ≤ z :=
sorry

end NUMINAMATH_GPT_smallest_y_value_l172_17234


namespace NUMINAMATH_GPT_sin_of_5pi_over_6_l172_17209

theorem sin_of_5pi_over_6 : Real.sin (5 * Real.pi / 6) = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_sin_of_5pi_over_6_l172_17209


namespace NUMINAMATH_GPT_range_of_k_l172_17251

theorem range_of_k (k : ℝ) :
  (∀ (x1 : ℝ), x1 ∈ Set.Icc (-1 : ℝ) 3 →
    ∃ (x0 : ℝ), x0 ∈ Set.Icc (-1 : ℝ) 3 ∧ (2 * x1^2 + x1 - k) ≤ (x0^3 - 3 * x0)) →
  k ≥ 3 :=
by
  -- This is the place for the proof. 'sorry' is used to indicate that the proof is omitted.
  sorry

end NUMINAMATH_GPT_range_of_k_l172_17251


namespace NUMINAMATH_GPT_eq_op_l172_17247

-- Define the operation ⊕
def op (x y : ℝ) : ℝ := x^3 + 2 * x - y

-- State the theorem to be proven
theorem eq_op (k : ℝ) : op k (op k k) = k := sorry

end NUMINAMATH_GPT_eq_op_l172_17247


namespace NUMINAMATH_GPT_distance_from_top_correct_total_distance_covered_correct_fifth_climb_success_l172_17273

-- We will assume the depth of the well as a constant
def well_depth : ℝ := 4.0

-- Climb and slide distances as per each climb
def first_climb : ℝ := 1.2
def first_slide : ℝ := 0.4
def second_climb : ℝ := 1.4
def second_slide : ℝ := 0.5
def third_climb : ℝ := 1.1
def third_slide : ℝ := 0.3
def fourth_climb : ℝ := 1.2
def fourth_slide : ℝ := 0.2

noncomputable def net_gain_four_climbs : ℝ :=
  (first_climb - first_slide) + (second_climb - second_slide) +
  (third_climb - third_slide) + (fourth_climb - fourth_slide)

noncomputable def distance_from_top_after_four : ℝ := 
  well_depth - net_gain_four_climbs

noncomputable def total_distance_covered_four_climbs : ℝ :=
  first_climb + first_slide + second_climb + second_slide +
  third_climb + third_slide + fourth_climb + fourth_slide

noncomputable def can_climb_out_fifth_climb : Bool :=
  well_depth < (net_gain_four_climbs + first_climb)

-- Now we state the theorems we need to prove

theorem distance_from_top_correct :
  distance_from_top_after_four = 0.5 := by
  sorry

theorem total_distance_covered_correct :
  total_distance_covered_four_climbs = 6.3 := by
  sorry

theorem fifth_climb_success :
  can_climb_out_fifth_climb = true := by
  sorry

end NUMINAMATH_GPT_distance_from_top_correct_total_distance_covered_correct_fifth_climb_success_l172_17273


namespace NUMINAMATH_GPT_fish_size_difference_l172_17263

variables {S J W : ℝ}

theorem fish_size_difference (h1 : S = J + 21.52) (h2 : J = W - 12.64) : S - W = 8.88 :=
sorry

end NUMINAMATH_GPT_fish_size_difference_l172_17263


namespace NUMINAMATH_GPT_total_kids_got_in_equals_148_l172_17293

def total_kids : ℕ := 120 + 90 + 50

def denied_riverside : ℕ := (20 * 120) / 100
def denied_west_side : ℕ := (70 * 90) / 100
def denied_mountaintop : ℕ := 50 / 2

def got_in_riverside : ℕ := 120 - denied_riverside
def got_in_west_side : ℕ := 90 - denied_west_side
def got_in_mountaintop : ℕ := 50 - denied_mountaintop

def total_got_in : ℕ := got_in_riverside + got_in_west_side + got_in_mountaintop

theorem total_kids_got_in_equals_148 :
  total_got_in = 148 := 
by
  unfold total_got_in
  unfold got_in_riverside got_in_west_side got_in_mountaintop
  unfold denied_riverside denied_west_side denied_mountaintop
  sorry

end NUMINAMATH_GPT_total_kids_got_in_equals_148_l172_17293


namespace NUMINAMATH_GPT_no_constant_term_in_expansion_l172_17258

theorem no_constant_term_in_expansion : 
  ∀ (x : ℂ), ¬ ∃ (k : ℕ), ∃ (c : ℂ), c * x ^ (k / 3 - 2 * (12 - k)) = 0 :=
by sorry

end NUMINAMATH_GPT_no_constant_term_in_expansion_l172_17258


namespace NUMINAMATH_GPT_triangle_find_C_angle_triangle_find_perimeter_l172_17246

variable (A B C a b c : ℝ)

theorem triangle_find_C_angle
  (h1 : 2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c) :
  C = π / 3 :=
sorry

theorem triangle_find_perimeter
  (h1 : 2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c)
  (h2 : c = Real.sqrt 7)
  (h3 : a * b = 6) :
  a + b + c = 5 + Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_triangle_find_C_angle_triangle_find_perimeter_l172_17246


namespace NUMINAMATH_GPT_john_pays_in_30_day_month_l172_17295

-- The cost of one pill
def cost_per_pill : ℝ := 1.5

-- The number of pills John takes per day
def pills_per_day : ℕ := 2

-- The number of days in a month
def days_in_month : ℕ := 30

-- The insurance coverage percentage
def insurance_coverage : ℝ := 0.40

-- Calculate the total cost John has to pay after insurance coverage in a 30-day month
theorem john_pays_in_30_day_month : (2 * 30) * 1.5 * 0.60 = 54 :=
by
  sorry

end NUMINAMATH_GPT_john_pays_in_30_day_month_l172_17295


namespace NUMINAMATH_GPT_bottles_per_person_l172_17286

theorem bottles_per_person
  (boxes : ℕ)
  (bottles_per_box : ℕ)
  (bottles_eaten : ℕ)
  (people : ℕ)
  (total_bottles : ℕ := boxes * bottles_per_box)
  (remaining_bottles : ℕ := total_bottles - bottles_eaten)
  (bottles_per_person : ℕ := remaining_bottles / people) :
  boxes = 7 → bottles_per_box = 9 → bottles_eaten = 7 → people = 8 → bottles_per_person = 7 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_bottles_per_person_l172_17286


namespace NUMINAMATH_GPT_peggy_buys_three_folders_l172_17264

theorem peggy_buys_three_folders 
  (red_sheets : ℕ) (green_sheets : ℕ) (blue_sheets : ℕ)
  (red_stickers_per_sheet : ℕ) (green_stickers_per_sheet : ℕ) (blue_stickers_per_sheet : ℕ)
  (total_stickers : ℕ) :
  red_sheets = 10 →
  green_sheets = 10 →
  blue_sheets = 10 →
  red_stickers_per_sheet = 3 →
  green_stickers_per_sheet = 2 →
  blue_stickers_per_sheet = 1 →
  total_stickers = 60 →
  1 + 1 + 1 = 3 :=
by 
  intros _ _ _ _ _ _ _
  sorry

end NUMINAMATH_GPT_peggy_buys_three_folders_l172_17264


namespace NUMINAMATH_GPT_scientific_notation_43300000_l172_17283

theorem scientific_notation_43300000 : 43300000 = 4.33 * 10^7 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_43300000_l172_17283


namespace NUMINAMATH_GPT_total_yield_l172_17203

noncomputable def johnson_hectare_yield_2months : ℕ := 80
noncomputable def neighbor_hectare_yield_multiplier : ℕ := 2
noncomputable def neighbor_hectares : ℕ := 2
noncomputable def months : ℕ := 6

theorem total_yield (jh2 : ℕ := johnson_hectare_yield_2months) 
                    (nhm : ℕ := neighbor_hectare_yield_multiplier) 
                    (nh : ℕ := neighbor_hectares) 
                    (m : ℕ := months): 
                    3 * jh2 + 3 * nh * jh2 * nhm = 1200 :=
by
  sorry

end NUMINAMATH_GPT_total_yield_l172_17203


namespace NUMINAMATH_GPT_sum_first_10_terms_l172_17205

-- Define the general arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a (n + 1) = a n + d

-- Define the conditions of the problem
def given_conditions (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 1 = 2 ∧ (a 2) ^ 2 = 2 * a 4 ∧ arithmetic_seq a d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n * (a 1 + a n)) / 2

-- Statement of the problem
theorem sum_first_10_terms (a : ℕ → ℤ) (d : ℤ) (S₁₀ : ℤ) :
  given_conditions a d →
  (S₁₀ = 20 ∨ S₁₀ = 110) :=
sorry

end NUMINAMATH_GPT_sum_first_10_terms_l172_17205


namespace NUMINAMATH_GPT_score_seventy_five_can_be_achieved_three_ways_l172_17210

-- Defining the problem constraints and goal
def quiz_problem (c u i : ℕ) (S : ℝ) : Prop :=
  c + u + i = 20 ∧ S = 5 * (c : ℝ) + 1.5 * (u : ℝ)

theorem score_seventy_five_can_be_achieved_three_ways :
  ∃ (c1 u1 c2 u2 c3 u3 : ℕ), 0 ≤ (5 * (c1 : ℝ) + 1.5 * (u1 : ℝ)) ∧ (5 * (c1 : ℝ) + 1.5 * (u1 : ℝ)) ≤ 100 ∧
  (5 * (c2 : ℝ) + 1.5 * (u2 : ℝ)) = 75 ∧ (5 * (c3 : ℝ) + 1.5 * (u3 : ℝ)) = 75 ∧
  (c1 ≠ c2 ∧ u1 ≠ u2) ∧ (c2 ≠ c3 ∧ u2 ≠ u3) ∧ (c3 ≠ c1 ∧ u3 ≠ u1) ∧ 
  quiz_problem c1 u1 (20 - c1 - u1) 75 ∧
  quiz_problem c2 u2 (20 - c2 - u2) 75 ∧
  quiz_problem c3 u3 (20 - c3 - u3) 75 :=
sorry

end NUMINAMATH_GPT_score_seventy_five_can_be_achieved_three_ways_l172_17210


namespace NUMINAMATH_GPT_solve_trig_eq_l172_17220

-- Define the equation
def equation (x : ℝ) : Prop := 3 * Real.sin x = 1 + Real.cos (2 * x)

-- Define the solution set
def solution_set (x : ℝ) : Prop := ∃ k : ℤ, x = k * Real.pi + (-1)^k * (Real.pi / 6)

-- The proof problem statement
theorem solve_trig_eq {x : ℝ} : equation x ↔ solution_set x := sorry

end NUMINAMATH_GPT_solve_trig_eq_l172_17220


namespace NUMINAMATH_GPT_sum_of_decimals_is_fraction_l172_17282

theorem sum_of_decimals_is_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 1466 / 6250 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_decimals_is_fraction_l172_17282


namespace NUMINAMATH_GPT_system_of_equations_l172_17292

theorem system_of_equations (x y : ℝ) 
  (h1 : 2019 * x + 2020 * y = 2018) 
  (h2 : 2020 * x + 2019 * y = 2021) :
  x + y = 1 ∧ x - y = 3 :=
by sorry

end NUMINAMATH_GPT_system_of_equations_l172_17292


namespace NUMINAMATH_GPT_width_of_wall_l172_17269

def volume_of_brick (length width height : ℝ) : ℝ :=
  length * width * height

def volume_of_wall (length width height : ℝ) : ℝ :=
  length * width * height

theorem width_of_wall
  (l_b w_b h_b : ℝ) (n : ℝ) (L H : ℝ)
  (volume_brick := volume_of_brick l_b w_b h_b)
  (total_volume_bricks := n * volume_brick) :
  volume_of_wall L (total_volume_bricks / (L * H)) H = total_volume_bricks :=
by
  sorry

end NUMINAMATH_GPT_width_of_wall_l172_17269


namespace NUMINAMATH_GPT_boat_cannot_complete_round_trip_l172_17212

theorem boat_cannot_complete_round_trip
  (speed_still_water : ℝ)
  (speed_current : ℝ)
  (distance : ℝ)
  (total_time : ℝ)
  (speed_still_water_pos : speed_still_water > 0)
  (speed_current_nonneg : speed_current ≥ 0)
  (distance_pos : distance > 0)
  (total_time_pos : total_time > 0) :
  let speed_downstream := speed_still_water + speed_current
  let speed_upstream := speed_still_water - speed_current
  let time_downstream := distance / speed_downstream
  let time_upstream := distance / speed_upstream
  let total_trip_time := time_downstream + time_upstream
  total_trip_time > total_time :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_boat_cannot_complete_round_trip_l172_17212


namespace NUMINAMATH_GPT_evaluate_expression_l172_17201

theorem evaluate_expression : 
  abs (abs (-abs (3 - 5) + 2) - 4) = 4 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l172_17201


namespace NUMINAMATH_GPT_walk_usual_time_l172_17236

theorem walk_usual_time (T : ℝ) (S : ℝ) (h1 : (5 / 4 : ℝ) = (T + 10) / T) : T = 40 :=
sorry

end NUMINAMATH_GPT_walk_usual_time_l172_17236


namespace NUMINAMATH_GPT_julias_preferred_number_l172_17245

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n % 100) / 10) + (n % 10)

theorem julias_preferred_number : ∃ n : ℕ, n > 100 ∧ n < 200 ∧ n % 13 = 0 ∧ n % 3 ≠ 0 ∧ sum_of_digits n % 5 = 0 ∧ n = 104 :=
by
  sorry

end NUMINAMATH_GPT_julias_preferred_number_l172_17245


namespace NUMINAMATH_GPT_no_such_triples_l172_17270

theorem no_such_triples : ¬ ∃ a b c : ℕ, 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  Prime ((a-2)*(b-2)*(c-2)+12) ∧ 
  ((a-2)*(b-2)*(c-2)+12) ∣ (a^2 + b^2 + c^2 + a*b*c - 2017) := 
by sorry

end NUMINAMATH_GPT_no_such_triples_l172_17270


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l172_17238

theorem quadratic_has_two_distinct_real_roots :
  let a := 1
  let b := 1
  let c := -1
  let discriminant := b^2 - 4 * a * c
  discriminant > 0 :=
by
  let a := 1
  let b := 1
  let c := -1
  let discriminant := b^2 - 4 * a * c
  show discriminant > 0
  sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l172_17238


namespace NUMINAMATH_GPT_factor_x8_minus_81_l172_17256

theorem factor_x8_minus_81 (x : ℝ) : x^8 - 81 = (x^2 - 3) * (x^2 + 3) * (x^4 + 9) := 
by 
  sorry

end NUMINAMATH_GPT_factor_x8_minus_81_l172_17256


namespace NUMINAMATH_GPT_min_value_of_b_plus_3_div_a_l172_17222

theorem min_value_of_b_plus_3_div_a (a : ℝ) (b : ℝ) :
  0 < a →
  (∀ x, 0 < x → (a * x - 2) * (-x^2 - b * x + 4) ≤ 0) →
  b + 3 / a = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_b_plus_3_div_a_l172_17222


namespace NUMINAMATH_GPT_first_positive_term_is_7_l172_17297

-- Define the conditions and the sequence
def a1 : ℚ := -1
def d : ℚ := 1 / 5

-- Define the general term of the sequence
def a_n (n : ℕ) : ℚ := a1 + (n - 1) * d

-- Define the proposition that the 7th term is the first positive term
theorem first_positive_term_is_7 :
  ∀ n : ℕ, (0 < a_n n) → (7 <= n) :=
by
  intro n h
  sorry

end NUMINAMATH_GPT_first_positive_term_is_7_l172_17297


namespace NUMINAMATH_GPT_circle_in_fourth_quadrant_l172_17233

theorem circle_in_fourth_quadrant (a : ℝ) :
  (∃ (x y: ℝ), x^2 + y^2 - 2 * a * x + 4 * a * y + 6 * a^2 - a = 0 ∧ (a > 0) ∧ (-2 * y < 0)) → (0 < a ∧ a < 1) :=
by
  sorry

end NUMINAMATH_GPT_circle_in_fourth_quadrant_l172_17233


namespace NUMINAMATH_GPT_base_conversion_positive_b_l172_17296

theorem base_conversion_positive_b :
  (∃ (b : ℝ), 3 * 5^1 + 2 * 5^0 = 17 ∧ 1 * b^2 + 2 * b^1 + 0 * b^0 = 17 ∧ b = -1 + 3 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_base_conversion_positive_b_l172_17296


namespace NUMINAMATH_GPT_find_chocolate_boxes_l172_17206

section
variable (x : Nat)
variable (candy_per_box : Nat := 8)
variable (caramel_boxes : Nat := 3)
variable (total_candy : Nat := 80)

theorem find_chocolate_boxes :
  8 * x + candy_per_box * caramel_boxes = total_candy -> x = 7 :=
by
  sorry
end

end NUMINAMATH_GPT_find_chocolate_boxes_l172_17206


namespace NUMINAMATH_GPT_intersection_A_B_l172_17253

def A : Set ℝ := {y | ∃ x : ℝ, y = Real.cos x}
def B : Set ℝ := {x | x * (x + 1) ≥ 0}

theorem intersection_A_B :
  (A ∩ B) = {x | (0 ≤ x ∧ x ≤ 1) ∨ x = -1} :=
  sorry

end NUMINAMATH_GPT_intersection_A_B_l172_17253


namespace NUMINAMATH_GPT_cubic_inequality_l172_17249

theorem cubic_inequality 
  (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) :
  2 * (x^3 + y^3 + z^3) - (x^2 * y + y^2 * z + z^2 * x) ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_cubic_inequality_l172_17249


namespace NUMINAMATH_GPT_largest_multiple_of_9_less_than_100_l172_17294

theorem largest_multiple_of_9_less_than_100 : ∃ k : ℕ, 9 * k < 100 ∧ (∀ m : ℕ, 9 * m < 100 → 9 * m ≤ 9 * k) ∧ 9 * k = 99 :=
by sorry

end NUMINAMATH_GPT_largest_multiple_of_9_less_than_100_l172_17294


namespace NUMINAMATH_GPT_trail_length_l172_17224

variables (a b c d e : ℕ)

theorem trail_length : 
  a + b + c = 45 ∧
  b + d = 36 ∧
  c + d + e = 60 ∧
  a + d = 32 → 
  a + b + c + d + e = 69 :=
by
  intro h
  obtain ⟨h1, h2, h3, h4⟩ := h
  sorry

end NUMINAMATH_GPT_trail_length_l172_17224


namespace NUMINAMATH_GPT_flowchart_basic_elements_includes_loop_l172_17257

theorem flowchart_basic_elements_includes_loop 
  (sequence_structure : Prop)
  (condition_structure : Prop)
  (loop_structure : Prop)
  : ∃ element : ℕ, element = 2 := 
by
  -- Assume 0 is A: Judgment
  -- Assume 1 is B: Directed line
  -- Assume 2 is C: Loop
  -- Assume 3 is D: Start
  sorry

end NUMINAMATH_GPT_flowchart_basic_elements_includes_loop_l172_17257
