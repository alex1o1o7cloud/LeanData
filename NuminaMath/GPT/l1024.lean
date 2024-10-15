import Mathlib

namespace NUMINAMATH_GPT_age_difference_is_51_l1024_102482

def Milena_age : ℕ := 7
def Grandmother_age : ℕ := 9 * Milena_age
def Grandfather_age : ℕ := Grandmother_age + 2
def Cousin_age : ℕ := 2 * Milena_age
def Age_difference : ℕ := Grandfather_age - Cousin_age

theorem age_difference_is_51 : Age_difference = 51 := by
  sorry

end NUMINAMATH_GPT_age_difference_is_51_l1024_102482


namespace NUMINAMATH_GPT_journey_length_l1024_102487

/-- Define the speed in the urban area as 55 km/h. -/
def urban_speed : ℕ := 55

/-- Define the speed on the highway as 85 km/h. -/
def highway_speed : ℕ := 85

/-- Define the time spent in each area as 3 hours. -/
def travel_time : ℕ := 3

/-- Define the distance traveled in the urban area as the product of the speed and time. -/
def urban_distance : ℕ := urban_speed * travel_time

/-- Define the distance traveled on the highway as the product of the speed and time. -/
def highway_distance : ℕ := highway_speed * travel_time

/-- Define the total distance of the journey. -/
def total_distance : ℕ := urban_distance + highway_distance

/-- The theorem that the total distance is 420 km. -/
theorem journey_length : total_distance = 420 := by
  -- Prove the equality by calculating the distances and summing them up
  sorry

end NUMINAMATH_GPT_journey_length_l1024_102487


namespace NUMINAMATH_GPT_bc_approx_A_l1024_102489

theorem bc_approx_A (A B C D E : ℝ) 
    (hA : 0 < A ∧ A < 1) (hB : 0 < B ∧ B < 1) (hC : 0 < C ∧ C < 1)
    (hD : 0 < D ∧ D < 1) (hE : 1 < E ∧ E < 2)
    (hA_val : A = 0.2) (hB_val : B = 0.4) (hC_val : C = 0.6) (hD_val : D = 0.8) :
    abs (B * C - A) < abs (B * C - B) ∧ abs (B * C - A) < abs (B * C - C) ∧ abs (B * C - A) < abs (B * C - D) := 
by 
  sorry

end NUMINAMATH_GPT_bc_approx_A_l1024_102489


namespace NUMINAMATH_GPT_partition_sum_le_152_l1024_102436

theorem partition_sum_le_152 {S : ℕ} (l : List ℕ) 
  (h1 : ∀ n ∈ l, 1 ≤ n ∧ n ≤ 10) 
  (h2 : l.sum = S) : 
  (∃ l1 l2 : List ℕ, l1.sum ≤ 80 ∧ l2.sum ≤ 80 ∧ l1 ++ l2 = l) ↔ S ≤ 152 := 
by
  sorry

end NUMINAMATH_GPT_partition_sum_le_152_l1024_102436


namespace NUMINAMATH_GPT_intersection_A_B_union_B_complement_A_l1024_102434

open Set

variable (U A B : Set ℝ)

noncomputable def U_def : Set ℝ := {x | -5 ≤ x ∧ x ≤ 5}
noncomputable def A_def : Set ℝ := {x | 0 < x ∧ x ≤ 3}
noncomputable def B_def : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

theorem intersection_A_B : (A ∩ B) = {x | 0 < x ∧ x ≤ 1} :=
sorry

theorem union_B_complement_A : B ∪ (U \ A) = {x | (-5 ≤ x ∧ x ≤ 1) ∨ (3 < x ∧ x ≤ 5)} :=
sorry

attribute [irreducible] U_def A_def B_def

end NUMINAMATH_GPT_intersection_A_B_union_B_complement_A_l1024_102434


namespace NUMINAMATH_GPT_greatest_monthly_drop_in_March_l1024_102441

noncomputable def jan_price_change : ℝ := -3.00
noncomputable def feb_price_change : ℝ := 1.50
noncomputable def mar_price_change : ℝ := -4.50
noncomputable def apr_price_change : ℝ := 2.00
noncomputable def may_price_change : ℝ := -1.00
noncomputable def jun_price_change : ℝ := 0.50

theorem greatest_monthly_drop_in_March :
  mar_price_change < jan_price_change ∧
  mar_price_change < feb_price_change ∧
  mar_price_change < apr_price_change ∧
  mar_price_change < may_price_change ∧
  mar_price_change < jun_price_change :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_monthly_drop_in_March_l1024_102441


namespace NUMINAMATH_GPT_number_of_whole_numbers_without_1_or_2_l1024_102446

/-- There are 439 whole numbers between 1 and 500 that do not contain the digit 1 or 2. -/
theorem number_of_whole_numbers_without_1_or_2 : 
  ∃ n : ℕ, n = 439 ∧ ∀ m, 1 ≤ m ∧ m ≤ 500 → ∀ d ∈ (m.digits 10), d ≠ 1 ∧ d ≠ 2 :=
sorry

end NUMINAMATH_GPT_number_of_whole_numbers_without_1_or_2_l1024_102446


namespace NUMINAMATH_GPT_simplify_expression_l1024_102476

theorem simplify_expression (x : ℝ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1024_102476


namespace NUMINAMATH_GPT_combined_age_71_in_6_years_l1024_102488

-- Given conditions
variable (combinedAgeIn15Years : ℕ) (h_condition : combinedAgeIn15Years = 107)

-- Define the question
def combinedAgeIn6Years : ℕ := combinedAgeIn15Years - 4 * (15 - 6)

-- State the theorem to prove the question == answer given conditions
theorem combined_age_71_in_6_years (h_condition : combinedAgeIn15Years = 107) : combinedAgeIn6Years combinedAgeIn15Years = 71 := 
by 
  sorry

end NUMINAMATH_GPT_combined_age_71_in_6_years_l1024_102488


namespace NUMINAMATH_GPT_sin_identity_l1024_102430

theorem sin_identity {α : ℝ} (h : Real.sin (π / 3 - α) = 1 / 4) : 
  Real.sin (π / 6 - 2 * α) = -7 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_sin_identity_l1024_102430


namespace NUMINAMATH_GPT_find_b_coefficients_l1024_102458

theorem find_b_coefficients (x : ℝ) (b₁ b₂ b₃ b₄ : ℝ) :
  x^4 = (x + 1)^4 + b₁ * (x + 1)^3 + b₂ * (x + 1)^2 + b₃ * (x + 1) + b₄ →
  b₁ = -4 ∧ b₂ = 6 ∧ b₃ = -4 ∧ b₄ = 1 := by
  sorry

end NUMINAMATH_GPT_find_b_coefficients_l1024_102458


namespace NUMINAMATH_GPT_value_of_expression_l1024_102412

theorem value_of_expression (x : ℕ) (h : x = 8) : 
  (x^3 + 3 * (x^2) * 2 + 3 * x * (2^2) + 2^3 = 1000) := by
{
  sorry
}

end NUMINAMATH_GPT_value_of_expression_l1024_102412


namespace NUMINAMATH_GPT_find_minimum_value_max_value_when_g_half_l1024_102426

noncomputable def f (a x : ℝ) : ℝ := 1 - 2 * a - 2 * a * (Real.cos x) - 2 * (Real.sin x) ^ 2

noncomputable def g (a : ℝ) : ℝ :=
  if a < -2 then 1
  else if a <= 2 then -a^2 / 2 - 2 * a - 1
  else 1 - 4 * a

theorem find_minimum_value (a : ℝ) :
  ∃ g_val, g_val = g a :=
  sorry

theorem max_value_when_g_half : 
  g (-1) = 1 / 2 →
  ∃ max_val, max_val = (max (f (-1) π) (f (-1) 0)) :=
  sorry

end NUMINAMATH_GPT_find_minimum_value_max_value_when_g_half_l1024_102426


namespace NUMINAMATH_GPT_bangles_per_box_l1024_102433

-- Define the total number of pairs of bangles
def totalPairs : Nat := 240

-- Define the number of boxes
def numberOfBoxes : Nat := 20

-- Define the proof that each box can hold 24 bangles
theorem bangles_per_box : (totalPairs * 2) / numberOfBoxes = 24 :=
by
  -- Here we're required to do the proof but we'll use 'sorry' to skip it
  sorry

end NUMINAMATH_GPT_bangles_per_box_l1024_102433


namespace NUMINAMATH_GPT_problem_l1024_102466

open Set

theorem problem (a b : ℝ) :
  (A = {x | x^2 - 2*x - 3 > 0}) →
  (B = {x | x^2 + a*x + b ≤ 0}) →
  (A ∪ B = univ) →
  (A ∩ B = Ioo 3 4) →
  a + b = -7 :=
by
  intros hA hB hUnion hIntersection
  sorry

end NUMINAMATH_GPT_problem_l1024_102466


namespace NUMINAMATH_GPT_complex_exponentiation_problem_l1024_102440

theorem complex_exponentiation_problem (z : ℂ) (h : z^2 + z + 1 = 0) : z^2010 + z^2009 + 1 = 0 :=
sorry

end NUMINAMATH_GPT_complex_exponentiation_problem_l1024_102440


namespace NUMINAMATH_GPT_a4_eq_2_or_neg2_l1024_102493

variable (a : ℕ → ℝ)
variable (r : ℝ)

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- Given conditions
axiom h1 : is_geometric_sequence a r
axiom h2 : a 2 * a 6 = 4

-- Theorem to prove
theorem a4_eq_2_or_neg2 : a 4 = 2 ∨ a 4 = -2 :=
sorry

end NUMINAMATH_GPT_a4_eq_2_or_neg2_l1024_102493


namespace NUMINAMATH_GPT_pete_flag_total_circles_squares_l1024_102451

def US_flag_stars : ℕ := 50
def US_flag_stripes : ℕ := 13

def circles (stars : ℕ) : ℕ := (stars / 2) - 3
def squares (stripes : ℕ) : ℕ := (2 * stripes) + 6

theorem pete_flag_total_circles_squares : 
  circles US_flag_stars + squares US_flag_stripes = 54 := 
by
  unfold circles squares US_flag_stars US_flag_stripes
  sorry

end NUMINAMATH_GPT_pete_flag_total_circles_squares_l1024_102451


namespace NUMINAMATH_GPT_aarons_brothers_number_l1024_102449

-- We are defining the conditions as functions

def number_of_aarons_sisters := 4
def bennetts_brothers := 6
def bennetts_cousins := 3
def twice_aarons_brothers_minus_two (Ba : ℕ) := 2 * Ba - 2
def bennetts_cousins_one_more_than_aarons_sisters (As : ℕ) := As + 1

-- We need to prove that Aaron's number of brothers Ba is 4 under these conditions

theorem aarons_brothers_number : ∃ (Ba : ℕ), 
  bennetts_brothers = twice_aarons_brothers_minus_two Ba ∧ 
  bennetts_cousins = bennetts_cousins_one_more_than_aarons_sisters number_of_aarons_sisters ∧ 
  Ba = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_aarons_brothers_number_l1024_102449


namespace NUMINAMATH_GPT_lines_slope_angle_l1024_102402

theorem lines_slope_angle (m n : ℝ) (θ₁ θ₂ : ℝ)
  (h1 : L1 = fun x => m * x)
  (h2 : L2 = fun x => n * x)
  (h3 : θ₁ = 3 * θ₂)
  (h4 : m = 3 * n)
  (h5 : θ₂ ≠ 0) :
  m * n = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_lines_slope_angle_l1024_102402


namespace NUMINAMATH_GPT_inequality_proof_l1024_102448

theorem inequality_proof
  (a b c A α : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < α)
  (h_sum : a + b + c = A)
  (h_A : A ≤ 1) :
  (1 / a - a) ^ α + (1 / b - b) ^ α + (1 / c - c) ^ α ≥ 3 * (3 / A - A / 3) ^ α :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1024_102448


namespace NUMINAMATH_GPT_smallest_number_l1024_102407

theorem smallest_number (a b c d : ℝ) (h1 : a = -5) (h2 : b = 0) (h3 : c = 1/2) (h4 : d = Real.sqrt 2) : a ≤ b ∧ a ≤ c ∧ a ≤ d :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_l1024_102407


namespace NUMINAMATH_GPT_avg_height_of_remaining_students_l1024_102429

-- Define the given conditions
def avg_height_11_members : ℝ := 145.7
def number_of_members : ℝ := 11
def height_of_two_students : ℝ := 142.1

-- Define what we need to prove
theorem avg_height_of_remaining_students :
  (avg_height_11_members * number_of_members - 2 * height_of_two_students) / (number_of_members - 2) = 146.5 :=
by
  sorry

end NUMINAMATH_GPT_avg_height_of_remaining_students_l1024_102429


namespace NUMINAMATH_GPT_arithmetic_mean_is_one_l1024_102435

theorem arithmetic_mean_is_one (x a : ℝ) (hx : x ≠ 0) : 
  (1 / 2) * ((x + a) / x + (x - a) / x) = 1 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_is_one_l1024_102435


namespace NUMINAMATH_GPT_negation_of_p_l1024_102467

-- Define the original proposition p
def p : Prop := ∀ x : ℝ, 2 * x^2 - 1 > 0

-- State the theorem that the negation of p is equivalent to the given existential statement
theorem negation_of_p :
  ¬p ↔ ∃ x : ℝ, 2 * x^2 - 1 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l1024_102467


namespace NUMINAMATH_GPT_parts_sampling_l1024_102496

theorem parts_sampling (first_grade second_grade third_grade : ℕ)
                       (total_sample drawn_third : ℕ)
                       (h_first_grade : first_grade = 24)
                       (h_second_grade : second_grade = 36)
                       (h_total_sample : total_sample = 20)
                       (h_drawn_third : drawn_third = 10)
                       (h_non_third : third_grade = 60 - (24 + 36))
                       (h_total : 2 * (24 + 36) = 120)
                       (h_proportion : 2 * third_grade = 2 * (24 + 36)) :
    (third_grade = 60 ∧ (second_grade * (total_sample - drawn_third) / (24 + 36) = 6)) := by
    simp [h_first_grade, h_second_grade, h_total_sample, h_drawn_third] at *
    sorry

end NUMINAMATH_GPT_parts_sampling_l1024_102496


namespace NUMINAMATH_GPT_find_a_l1024_102468

theorem find_a 
  (x y a m n : ℝ)
  (h1 : x - 5 / 2 * y + 1 = 0) 
  (h2 : x = m + a) 
  (h3 : y = n + 1)  -- since k = 1, so we replace k with 1
  (h4 : m + a = m + 1 / 2) : 
  a = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_l1024_102468


namespace NUMINAMATH_GPT_product_of_possible_values_of_b_l1024_102475

theorem product_of_possible_values_of_b :
  let y₁ := -1
  let y₂ := 4
  let x₁ := 1
  let side_length := y₂ - y₁ -- Since this is 5 units
  let b₁ := x₁ - side_length -- This should be -4
  let b₂ := x₁ + side_length -- This should be 6
  let product := b₁ * b₂ -- So, (-4) * 6
  product = -24 :=
by
  sorry

end NUMINAMATH_GPT_product_of_possible_values_of_b_l1024_102475


namespace NUMINAMATH_GPT_possible_ages_that_sum_to_a_perfect_square_l1024_102459

def two_digit_number (a b : ℕ) := 10 * a + b
def reversed_number (a b : ℕ) := 10 * b + a

def sum_of_number_and_its_reversed (a b : ℕ) : ℕ := 
  two_digit_number a b + reversed_number a b

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem possible_ages_that_sum_to_a_perfect_square :
  ∃ (s : Finset ℕ), s.card = 6 ∧ 
  ∀ x ∈ s, ∃ a b : ℕ, a + b = 11 ∧ s = {two_digit_number a b} ∧ is_perfect_square (sum_of_number_and_its_reversed a b) :=
  sorry

end NUMINAMATH_GPT_possible_ages_that_sum_to_a_perfect_square_l1024_102459


namespace NUMINAMATH_GPT_remainder_div_l1024_102464

theorem remainder_div (P Q R D Q' R' : ℕ) (h₁ : P = Q * D + R) (h₂ : Q = (D - 1) * Q' + R') (h₃ : D > 1) :
  P % (D * (D - 1)) = D * R' + R := by sorry

end NUMINAMATH_GPT_remainder_div_l1024_102464


namespace NUMINAMATH_GPT_scale_drawing_l1024_102495

theorem scale_drawing (length_cm : ℝ) (representation : ℝ) : length_cm * representation = 3750 :=
by
  let length_cm := 7.5
  let representation := 500
  sorry

end NUMINAMATH_GPT_scale_drawing_l1024_102495


namespace NUMINAMATH_GPT_max_reflections_l1024_102401

theorem max_reflections (n : ℕ) (angle_CDA : ℝ) (h_angle : angle_CDA = 12) : n ≤ 7 ↔ 12 * n ≤ 90 := by
    sorry

end NUMINAMATH_GPT_max_reflections_l1024_102401


namespace NUMINAMATH_GPT_find_k_and_shifted_function_l1024_102490

noncomputable def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x + 1

theorem find_k_and_shifted_function (k : ℝ) (h : k ≠ 0) (h1 : linear_function k 1 = 3) :
  k = 2 ∧ linear_function 2 x + 2 = 2 * x + 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_and_shifted_function_l1024_102490


namespace NUMINAMATH_GPT_study_tour_part1_l1024_102453

theorem study_tour_part1 (x y : ℕ) 
  (h1 : 45 * y + 15 = x) 
  (h2 : 60 * (y - 3) = x) : 
  x = 600 ∧ y = 13 :=
by sorry

end NUMINAMATH_GPT_study_tour_part1_l1024_102453


namespace NUMINAMATH_GPT_range_of_a_l1024_102422

theorem range_of_a 
  (a : ℝ) (h : ∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) 
  : -2 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1024_102422


namespace NUMINAMATH_GPT_find_y_l1024_102417

variables (x y : ℝ)

theorem find_y (h1 : x = 103) (h2 : x^3 * y - 4 * x^2 * y + 4 * x * y = 515400) : y = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_y_l1024_102417


namespace NUMINAMATH_GPT_smallest_positive_period_of_f_max_min_values_of_f_l1024_102492

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1 / 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos (2 * x))
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

--(I) Prove the smallest positive period of f(x) is π.
theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + π) = f x :=
sorry

--(II) Prove the maximum and minimum values of f(x) on [0, π / 2] are 1 and -1/2 respectively.
theorem max_min_values_of_f : ∃ max min : ℝ, (max = 1) ∧ (min = -1 / 2) ∧
  (∀ x ∈ Set.Icc 0 (π / 2), f x ≤ max ∧ f x ≥ min) :=
sorry

end NUMINAMATH_GPT_smallest_positive_period_of_f_max_min_values_of_f_l1024_102492


namespace NUMINAMATH_GPT_javier_first_throw_l1024_102494

theorem javier_first_throw 
  (second third first : ℝ)
  (h1 : first = 2 * second)
  (h2 : first = (1 / 2) * third)
  (h3 : first + second + third = 1050) :
  first = 300 := 
by sorry

end NUMINAMATH_GPT_javier_first_throw_l1024_102494


namespace NUMINAMATH_GPT_tim_initial_soda_l1024_102428

-- Define the problem
def initial_cans (x : ℕ) : Prop :=
  let after_jeff_takes := x - 6
  let after_buying_more := after_jeff_takes + after_jeff_takes / 2
  after_buying_more = 24

-- Theorem stating the problem in Lean 4
theorem tim_initial_soda (x : ℕ) (h: initial_cans x) : x = 22 :=
by
  sorry

end NUMINAMATH_GPT_tim_initial_soda_l1024_102428


namespace NUMINAMATH_GPT_find_y_coordinate_of_P_l1024_102480

-- Define the conditions as Lean definitions
def distance_x_axis_to_P (P : ℝ × ℝ) :=
  abs P.2

def distance_y_axis_to_P (P : ℝ × ℝ) :=
  abs P.1

-- Lean statement of the problem
theorem find_y_coordinate_of_P (P : ℝ × ℝ)
  (h1 : distance_x_axis_to_P P = (1/2) * distance_y_axis_to_P P)
  (h2 : distance_y_axis_to_P P = 10) :
  P.2 = 5 ∨ P.2 = -5 :=
sorry

end NUMINAMATH_GPT_find_y_coordinate_of_P_l1024_102480


namespace NUMINAMATH_GPT_sin_value_l1024_102437

theorem sin_value (α : ℝ) (h : Real.cos (π / 6 - α) = (Real.sqrt 3) / 3) :
    Real.sin (5 * π / 6 - 2 * α) = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sin_value_l1024_102437


namespace NUMINAMATH_GPT_max_M_is_2_l1024_102474

theorem max_M_is_2 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hdisc : b^2 - 4 * a * c ≥ 0) :
    max (min (b + c / a) (min (c + a / b) (a + b / c))) = 2 := by
    sorry

end NUMINAMATH_GPT_max_M_is_2_l1024_102474


namespace NUMINAMATH_GPT_problem_solution_l1024_102450

theorem problem_solution
  (n m k l : ℕ)
  (h1 : n ≠ 1)
  (h2 : 0 < n)
  (h3 : 0 < m)
  (h4 : 0 < k)
  (h5 : 0 < l)
  (h6 : n^k + m * n^l + 1 ∣ n^(k + l) - 1) :
  (m = 1 ∧ l = 2 * k) ∨ (l ∣ k ∧ m = (n^(k - l) - 1) / (n^l - 1)) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1024_102450


namespace NUMINAMATH_GPT_percentage_of_second_solution_is_16point67_l1024_102479

open Real

def percentage_second_solution (x : ℝ) : Prop :=
  let v₁ := 50
  let c₁ := 0.10
  let c₂ := x / 100
  let v₂ := 200 - v₁
  let c_final := 0.15
  let v_final := 200
  (c₁ * v₁) + (c₂ * v₂) = (c_final * v_final)

theorem percentage_of_second_solution_is_16point67 :
  ∃ x, percentage_second_solution x ∧ x = (50/3) :=
sorry

end NUMINAMATH_GPT_percentage_of_second_solution_is_16point67_l1024_102479


namespace NUMINAMATH_GPT_find_shirts_yesterday_l1024_102497

def shirts_per_minute : ℕ := 8
def total_minutes : ℕ := 2
def shirts_today : ℕ := 3

def total_shirts : ℕ := shirts_per_minute * total_minutes
def shirts_yesterday : ℕ := total_shirts - shirts_today

theorem find_shirts_yesterday : shirts_yesterday = 13 := by
  sorry

end NUMINAMATH_GPT_find_shirts_yesterday_l1024_102497


namespace NUMINAMATH_GPT_math_proof_l1024_102419

variable {a b c A B C : ℝ}
variable {S : ℝ}

noncomputable def problem_statement (h1 : b + c = 2 * a * Real.cos B)
    (h2 : S = a^2 / 4) : Prop :=
    (∃ A B : ℝ, (A = 2 * B) ∧ (A = 90)) 

theorem math_proof (h1 : b + c = 2 * a * Real.cos B)
    (h2 : S = a^2 / 4) :
    problem_statement h1 h2 :=
    sorry

end NUMINAMATH_GPT_math_proof_l1024_102419


namespace NUMINAMATH_GPT_max_triangles_9261_l1024_102461

-- Define the problem formally
noncomputable def max_triangles (points : ℕ) (circ_radius : ℝ) (min_side_length : ℝ) : ℕ :=
  -- Function definition for calculating the maximum number of triangles
  sorry

-- State the conditions and the expected maximum number of triangles
theorem max_triangles_9261 :
  max_triangles 63 10 9 = 9261 :=
sorry

end NUMINAMATH_GPT_max_triangles_9261_l1024_102461


namespace NUMINAMATH_GPT_slope_of_line_l1024_102420

theorem slope_of_line (a : ℝ) (h : a = (Real.tan (Real.pi / 3))) : a = Real.sqrt 3 := by
sorry

end NUMINAMATH_GPT_slope_of_line_l1024_102420


namespace NUMINAMATH_GPT_first_payment_amount_l1024_102424

-- The number of total payments
def total_payments : Nat := 65

-- The number of the first payments
def first_payments : Nat := 20

-- The number of remaining payments
def remaining_payments : Nat := total_payments - first_payments

-- The extra amount added to the remaining payments
def extra_amount : Int := 65

-- The average payment
def average_payment : Int := 455

-- The total amount paid over the year
def total_amount_paid : Int := average_payment * total_payments

-- The variable we want to solve for: amount of each of the first 20 payments
variable (x : Int)

-- The equation for total amount paid
def total_payments_equation : Prop :=
  20 * x + 45 * (x + 65) = 455 * 65

-- The theorem stating the amount of each of the first 20 payments
theorem first_payment_amount : x = 410 :=
  sorry

end NUMINAMATH_GPT_first_payment_amount_l1024_102424


namespace NUMINAMATH_GPT_part_I_solution_part_II_solution_l1024_102445

-- Definition of the function f(x)
def f (x a : ℝ) := |x - a| + |2 * x - 1|

-- Part (I) when a = 1, find the solution set for f(x) ≤ 2
theorem part_I_solution (x : ℝ) : f x 1 ≤ 2 ↔ 0 ≤ x ∧ x ≤ 4 / 3 :=
by sorry

-- Part (II) if the solution set for f(x) ≤ |2x + 1| contains [1/2, 1], find the range of a
theorem part_II_solution (a : ℝ) :
  (∀ x : ℝ, 1 / 2 ≤ x ∧ x ≤ 1 → f x a ≤ |2 * x + 1|) → -1 ≤ a ∧ a ≤ 5 / 2 :=
by sorry

end NUMINAMATH_GPT_part_I_solution_part_II_solution_l1024_102445


namespace NUMINAMATH_GPT_fifteen_percent_of_x_l1024_102421

variables (x : ℝ)

-- Condition: Given x% of 60 is 12
def is_x_percent_of_60 : Prop := (x / 100) * 60 = 12

-- Prove: 15% of x is 3
theorem fifteen_percent_of_x (h : is_x_percent_of_60 x) : (15 / 100) * x = 3 :=
by
  sorry

end NUMINAMATH_GPT_fifteen_percent_of_x_l1024_102421


namespace NUMINAMATH_GPT_cos_double_angle_l1024_102471

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 1/3) : Real.cos (2 * α) = 7/9 :=
by
    sorry

end NUMINAMATH_GPT_cos_double_angle_l1024_102471


namespace NUMINAMATH_GPT_min_value_expression_l1024_102481

theorem min_value_expression (a b t : ℝ) (h : a + b = t) : 
  ∃ c : ℝ, c = ((a^2 + 1)^2 + (b^2 + 1)^2) → c = (t^4 + 8 * t^2 + 16) / 8 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1024_102481


namespace NUMINAMATH_GPT_oranges_after_eating_l1024_102438

def initial_oranges : ℝ := 77.0
def eaten_oranges : ℝ := 2.0
def final_oranges : ℝ := 75.0

theorem oranges_after_eating :
  initial_oranges - eaten_oranges = final_oranges := by
  sorry

end NUMINAMATH_GPT_oranges_after_eating_l1024_102438


namespace NUMINAMATH_GPT_fifteenth_term_l1024_102427

noncomputable def seq : ℕ → ℝ
| 0       => 3
| 1       => 4
| (n + 2) => 12 / seq (n + 1)

theorem fifteenth_term :
  seq 14 = 3 :=
sorry

end NUMINAMATH_GPT_fifteenth_term_l1024_102427


namespace NUMINAMATH_GPT_product_of_reciprocals_plus_one_geq_nine_l1024_102410

theorem product_of_reciprocals_plus_one_geq_nine
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hab : a + b = 1) :
  (1 / a + 1) * (1 / b + 1) ≥ 9 :=
sorry

end NUMINAMATH_GPT_product_of_reciprocals_plus_one_geq_nine_l1024_102410


namespace NUMINAMATH_GPT_tenth_term_l1024_102491

-- Define the conditions
variables {a d : ℤ}

-- The conditions of the problem
axiom third_term_condition : a + 2 * d = 10
axiom sixth_term_condition : a + 5 * d = 16

-- The goal is to prove the tenth term
theorem tenth_term : a + 9 * d = 24 :=
by
  sorry

end NUMINAMATH_GPT_tenth_term_l1024_102491


namespace NUMINAMATH_GPT_num_three_digit_ints_with_odd_factors_l1024_102443

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end NUMINAMATH_GPT_num_three_digit_ints_with_odd_factors_l1024_102443


namespace NUMINAMATH_GPT_x_share_for_each_rupee_w_gets_l1024_102486

theorem x_share_for_each_rupee_w_gets (w_share : ℝ) (y_per_w : ℝ) (total_amount : ℝ) (a : ℝ) :
  w_share = 10 →
  y_per_w = 0.20 →
  total_amount = 15 →
  (w_share + w_share * a + w_share * y_per_w = total_amount) →
  a = 0.30 :=
by
  intros h_w h_y h_total h_eq
  sorry

end NUMINAMATH_GPT_x_share_for_each_rupee_w_gets_l1024_102486


namespace NUMINAMATH_GPT_inversely_proportional_y_value_l1024_102460

theorem inversely_proportional_y_value (x y k : ℝ)
  (h1 : ∀ x y : ℝ, x * y = k)
  (h2 : ∃ y : ℝ, x = 3 * y ∧ x + y = 36 ∧ x * y = k)
  (h3 : x = -9) : y = -27 := 
by
  sorry

end NUMINAMATH_GPT_inversely_proportional_y_value_l1024_102460


namespace NUMINAMATH_GPT_find_angle_4_l1024_102403

/-- Given angle conditions, prove that angle 4 is 22.5 degrees. -/
theorem find_angle_4 (angle : ℕ → ℝ) 
  (h1 : angle 1 + angle 2 = 180) 
  (h2 : angle 3 = angle 4) 
  (h3 : angle 1 = 85) 
  (h4 : angle 5 = 45) 
  (h5 : angle 1 + angle 5 + angle 6 = 180) : 
  angle 4 = 22.5 :=
sorry

end NUMINAMATH_GPT_find_angle_4_l1024_102403


namespace NUMINAMATH_GPT_total_seashells_l1024_102418

-- Define the conditions from part a)
def unbroken_seashells : ℕ := 2
def broken_seashells : ℕ := 4

-- Define the proof problem
theorem total_seashells :
  unbroken_seashells + broken_seashells = 6 :=
by
  sorry

end NUMINAMATH_GPT_total_seashells_l1024_102418


namespace NUMINAMATH_GPT_quadratics_roots_l1024_102456

theorem quadratics_roots (m n : ℝ) (r₁ r₂ : ℝ) 
  (h₁ : r₁^2 - m * r₁ + n = 0) (h₂ : r₂^2 - m * r₂ + n = 0) 
  (p q : ℝ) (h₃ : (r₁^2 - r₂^2)^2 + p * (r₁^2 - r₂^2) + q = 0) :
  p = 0 ∧ q = -m^4 + 4 * m^2 * n := 
sorry

end NUMINAMATH_GPT_quadratics_roots_l1024_102456


namespace NUMINAMATH_GPT_doubled_cost_percent_l1024_102462

-- Definitions
variable (t b : ℝ)
def cost (b : ℝ) : ℝ := t * b^4

-- Theorem statement
theorem doubled_cost_percent :
  cost t (2 * b) = 16 * cost t b :=
by
  -- To be proved
  sorry

end NUMINAMATH_GPT_doubled_cost_percent_l1024_102462


namespace NUMINAMATH_GPT_mul_72516_9999_l1024_102484

theorem mul_72516_9999 : 72516 * 9999 = 724787484 :=
by
  sorry

end NUMINAMATH_GPT_mul_72516_9999_l1024_102484


namespace NUMINAMATH_GPT_regular_tetrahedron_subdivision_l1024_102485

theorem regular_tetrahedron_subdivision :
  ∃ (n : ℕ), n ≤ 7 ∧ (∀ (i : ℕ) (h : i ≥ n), (1 / 2^i) < (1 / 100)) :=
by
  sorry

end NUMINAMATH_GPT_regular_tetrahedron_subdivision_l1024_102485


namespace NUMINAMATH_GPT_webinar_active_minutes_l1024_102406

theorem webinar_active_minutes :
  let hours := 13
  let extra_minutes := 17
  let break_minutes := 22
  (hours * 60 + extra_minutes) - break_minutes = 775 := by
  sorry

end NUMINAMATH_GPT_webinar_active_minutes_l1024_102406


namespace NUMINAMATH_GPT_next_time_10_10_11_15_l1024_102455

noncomputable def next_time_angle_x (current_time : ℕ × ℕ) (x : ℕ) : ℕ × ℕ := sorry

theorem next_time_10_10_11_15 :
  ∀ (x : ℕ), next_time_angle_x (10, 10) 115 = (11, 15) := sorry

end NUMINAMATH_GPT_next_time_10_10_11_15_l1024_102455


namespace NUMINAMATH_GPT_borgnine_tarantulas_needed_l1024_102413

def total_legs_goal : ℕ := 1100
def chimp_legs : ℕ := 12 * 4
def lion_legs : ℕ := 8 * 4
def lizard_legs : ℕ := 5 * 4
def tarantula_legs : ℕ := 8

theorem borgnine_tarantulas_needed : 
  let total_legs_seen := chimp_legs + lion_legs + lizard_legs
  let legs_needed := total_legs_goal - total_legs_seen
  let num_tarantulas := legs_needed / tarantula_legs
  num_tarantulas = 125 := 
by
  sorry

end NUMINAMATH_GPT_borgnine_tarantulas_needed_l1024_102413


namespace NUMINAMATH_GPT_total_number_of_balls_l1024_102425

-- Define the conditions
def balls_per_box : Nat := 3
def number_of_boxes : Nat := 2

-- Define the proposition
theorem total_number_of_balls : (balls_per_box * number_of_boxes) = 6 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_balls_l1024_102425


namespace NUMINAMATH_GPT_sum_of_squares_of_reciprocals_l1024_102478

-- Definitions based on the problem's conditions
variables (a b : ℝ) (hab : a + b = 3 * a * b + 1) (h_an : a ≠ 0) (h_bn : b ≠ 0)

-- Statement of the problem to be proved
theorem sum_of_squares_of_reciprocals :
  (1 / a^2) + (1 / b^2) = (4 * a * b + 10) / (a^2 * b^2) :=
sorry

end NUMINAMATH_GPT_sum_of_squares_of_reciprocals_l1024_102478


namespace NUMINAMATH_GPT_correct_conclusions_l1024_102454

noncomputable def M : Set ℝ := sorry

axiom non_empty : Nonempty M
axiom mem_2 : (2 : ℝ) ∈ M
axiom closed_under_sub : ∀ {x y : ℝ}, x ∈ M → y ∈ M → (x - y) ∈ M
axiom closed_under_div : ∀ {x : ℝ}, x ∈ M → x ≠ 0 → (1 / x) ∈ M

theorem correct_conclusions :
  (0 : ℝ) ∈ M ∧
  (∀ x y : ℝ, x ∈ M → y ∈ M → (x + y) ∈ M) ∧
  (∀ x y : ℝ, x ∈ M → y ∈ M → (x * y) ∈ M) ∧
  ¬ (1 ∉ M) := sorry

end NUMINAMATH_GPT_correct_conclusions_l1024_102454


namespace NUMINAMATH_GPT_evaluate_expression_l1024_102498

theorem evaluate_expression : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 = -1 := 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1024_102498


namespace NUMINAMATH_GPT_scatter_plot_can_be_made_l1024_102452

theorem scatter_plot_can_be_made
    (data : List (ℝ × ℝ)) :
    ∃ (scatter_plot : List (ℝ × ℝ)), scatter_plot = data :=
by
  sorry

end NUMINAMATH_GPT_scatter_plot_can_be_made_l1024_102452


namespace NUMINAMATH_GPT_probability_two_red_crayons_l1024_102447

def num_crayons : ℕ := 6
def num_red : ℕ := 3
def num_blue : ℕ := 2
def num_green : ℕ := 1
def num_choose (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_two_red_crayons :
  let total_pairs := num_choose num_crayons 2
  let red_pairs := num_choose num_red 2
  (red_pairs : ℚ) / (total_pairs : ℚ) = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_red_crayons_l1024_102447


namespace NUMINAMATH_GPT_three_numbers_sum_l1024_102483

theorem three_numbers_sum (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 10)
  (h4 : (a + b + c) / 3 = a + 8) (h5 : (a + b + c) / 3 = c - 20) : 
  a + b + c = 66 :=
sorry

end NUMINAMATH_GPT_three_numbers_sum_l1024_102483


namespace NUMINAMATH_GPT_find_large_monkey_doll_cost_l1024_102405

-- Define the conditions and the target property
def large_monkey_doll_cost (L : ℝ) (condition1 : 300 / (L - 2) = 300 / L + 25)
                           (condition2 : 300 / (L + 1) = 300 / L - 15) : Prop :=
  L = 6

-- The main theorem with the conditions
theorem find_large_monkey_doll_cost (L : ℝ)
  (h1 : 300 / (L - 2) = 300 / L + 25)
  (h2 : 300 / (L + 1) = 300 / L - 15) : large_monkey_doll_cost L h1 h2 :=
  sorry

end NUMINAMATH_GPT_find_large_monkey_doll_cost_l1024_102405


namespace NUMINAMATH_GPT_percentage_neither_language_l1024_102416

def total_diplomats : ℕ := 150
def french_speaking : ℕ := 17
def russian_speaking : ℕ := total_diplomats - 32
def both_languages : ℕ := 10 * total_diplomats / 100

theorem percentage_neither_language :
  let at_least_one_language := french_speaking + russian_speaking - both_languages
  let neither_language := total_diplomats - at_least_one_language
  neither_language * 100 / total_diplomats = 20 :=
by
  let at_least_one_language := french_speaking + russian_speaking - both_languages
  let neither_language := total_diplomats - at_least_one_language
  sorry

end NUMINAMATH_GPT_percentage_neither_language_l1024_102416


namespace NUMINAMATH_GPT_find_common_difference_l1024_102400

-- Definitions of the conditions
def arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ k : ℕ, a_n (k + 1) = a_n k + d

def sum_of_first_n_terms (a_n : ℕ → ℝ) (n : ℕ) (S_n : ℝ) : Prop :=
  S_n = (n : ℝ) / 2 * (a_n 1 + a_n n)

variables {a_1 d : ℝ}
variables (a_n : ℕ → ℝ)
variables (S_3 S_9 : ℝ)

-- Conditions from the problem statement
axiom a2_eq_3 : a_n 2 = 3
axiom S9_eq_6S3 : S_9 = 6 * S_3

-- The proof we need to write
theorem find_common_difference 
  (h1 : arithmetic_sequence a_n d)
  (h2 : sum_of_first_n_terms a_n 3 S_3)
  (h3 : sum_of_first_n_terms a_n 9 S_9) :
  d = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_common_difference_l1024_102400


namespace NUMINAMATH_GPT_ab_squared_ab_cubed_ab_power_n_l1024_102414

-- Definitions of a and b as real numbers, and n as a natural number
variables (a b : ℝ) (n : ℕ)

theorem ab_squared (a b : ℝ) : (a * b) ^ 2 = a ^ 2 * b ^ 2 := by 
  sorry

theorem ab_cubed (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by 
  sorry

theorem ab_power_n (a b : ℝ) (n : ℕ) : (a * b) ^ n = a ^ n * b ^ n := by 
  sorry

end NUMINAMATH_GPT_ab_squared_ab_cubed_ab_power_n_l1024_102414


namespace NUMINAMATH_GPT_find_smallest_x_l1024_102470

-- Definition of the conditions
def cong1 (x : ℤ) : Prop := x % 5 = 4
def cong2 (x : ℤ) : Prop := x % 7 = 6
def cong3 (x : ℤ) : Prop := x % 8 = 7

-- Statement of the problem
theorem find_smallest_x :
  ∃ (x : ℕ), x > 0 ∧ cong1 x ∧ cong2 x ∧ cong3 x ∧ x = 279 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_x_l1024_102470


namespace NUMINAMATH_GPT_round_robin_tournament_points_l1024_102463

theorem round_robin_tournament_points :
  ∀ (teams : Finset ℕ), teams.card = 6 →
  ∀ (matches_played : ℕ), matches_played = 12 →
  ∀ (total_points : ℤ), total_points = 32 →
  ∀ (third_highest_points : ℤ), third_highest_points = 7 →
  ∀ (draws : ℕ), draws = 4 →
  ∃ (fifth_highest_points_min fifth_highest_points_max : ℤ),
    fifth_highest_points_min = 1 ∧
    fifth_highest_points_max = 3 :=
by
  sorry

end NUMINAMATH_GPT_round_robin_tournament_points_l1024_102463


namespace NUMINAMATH_GPT_square_completion_form_l1024_102409

theorem square_completion_form (x k m: ℝ) (h: 16*x^2 - 32*x - 512 = 0):
  (x + k)^2 = m ↔ m = 65 :=
by
  sorry

end NUMINAMATH_GPT_square_completion_form_l1024_102409


namespace NUMINAMATH_GPT_abscissa_of_A_is_3_l1024_102473

-- Definitions of the points A, B, line l and conditions
def in_first_quadrant (A : ℝ × ℝ) := (A.1 > 0) ∧ (A.2 > 0)

def on_line_l (A : ℝ × ℝ) := A.2 = 2 * A.1

def point_B : ℝ × ℝ := (5, 0)

def diameter_circle (A B : ℝ × ℝ) (P : ℝ × ℝ) :=
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

-- Vectors AB and CD
def vector_AB (A B : ℝ × ℝ) := (B.1 - A.1, B.2 - A.2)

def vector_CD (C D : ℝ × ℝ) := (D.1 - C.1, D.2 - C.2)

def dot_product_zero (A B C D : ℝ × ℝ) := (vector_AB A B).1 * (vector_CD C D).1 + (vector_AB A B).2 * (vector_CD C D).2 = 0

-- Statement to prove
theorem abscissa_of_A_is_3 (A : ℝ × ℝ) (D : ℝ × ℝ) (a : ℝ) :
  in_first_quadrant A →
  on_line_l A →
  diameter_circle A point_B D →
  dot_product_zero A point_B (a, a) D →
  A.1 = 3 :=
by
  sorry

end NUMINAMATH_GPT_abscissa_of_A_is_3_l1024_102473


namespace NUMINAMATH_GPT_formation_enthalpy_benzene_l1024_102432

/-- Define the enthalpy changes based on given conditions --/
def ΔH_acetylene : ℝ := 226.7 -- kJ/mol for C₂H₂
def ΔH_benzene_formation : ℝ := 631.1 -- kJ for reactions forming C₆H₆
def ΔH_benzene_phase_change : ℝ := -33.9 -- kJ for phase change of C₆H₆

/-- Define the enthalpy change of formation for benzene --/
def ΔH_formation_benzene : ℝ := 3 * ΔH_acetylene + ΔH_benzene_formation + ΔH_benzene_phase_change

/-- Theorem stating the heat change in the reaction equals the calculated value --/
theorem formation_enthalpy_benzene :
  ΔH_formation_benzene = -82.9 :=
by
  sorry

end NUMINAMATH_GPT_formation_enthalpy_benzene_l1024_102432


namespace NUMINAMATH_GPT_cat_food_per_day_l1024_102477

theorem cat_food_per_day
  (bowl_empty_weight : ℕ)
  (bowl_weight_after_eating : ℕ)
  (food_eaten : ℕ)
  (days_per_fill : ℕ)
  (daily_food : ℕ) :
  (bowl_empty_weight = 420) →
  (bowl_weight_after_eating = 586) →
  (food_eaten = 14) →
  (days_per_fill = 3) →
  (bowl_weight_after_eating - bowl_empty_weight + food_eaten = days_per_fill * daily_food) →
  daily_food = 60 :=
by
  sorry

end NUMINAMATH_GPT_cat_food_per_day_l1024_102477


namespace NUMINAMATH_GPT_minimum_sum_of_original_numbers_l1024_102415

theorem minimum_sum_of_original_numbers 
  (m n : ℕ) 
  (h1 : m < n) 
  (h2 : 23 * m - 20 * n = 460) 
  (h3 : ∀ m n, 23 * m - 20 * n = 460 → m < n):
  m + n = 321 :=
sorry

end NUMINAMATH_GPT_minimum_sum_of_original_numbers_l1024_102415


namespace NUMINAMATH_GPT_part1_part2_l1024_102469

-- Define the triangle with sides a, b, c and the properties given.
variable (a b c : ℝ) (A B C : ℝ)
variable (A_ne_zero : A ≠ 0)
variable (b_cos_C a_cos_A c_cos_B : ℝ)

-- Given conditions
variable (h1 : b_cos_C = b * Real.cos C)
variable (h2 : a_cos_A = a * Real.cos A)
variable (h3 : c_cos_B = c * Real.cos B)
variable (h_seq : b_cos_C + c_cos_B = 2 * a_cos_A)
variable (A_plus_B_plus_C_eq_pi : A + B + C = Real.pi)

-- Part 1
theorem part1 : (A = Real.pi / 3) :=
by sorry

-- Part 2 with additional conditions
variable (h_a : a = 3 * Real.sqrt 2)
variable (h_bc_sum : b + c = 6)

theorem part2 : (|Real.sqrt (b ^ 2 + c ^ 2 - b * c)| = Real.sqrt 30) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1024_102469


namespace NUMINAMATH_GPT_shen_winning_probability_sum_l1024_102442

/-!
# Shen Winning Probability

Prove that the sum of the numerator and the denominator, m + n, 
of the simplified fraction representing Shen's winning probability is 184.
-/

theorem shen_winning_probability_sum :
  let m := 67
  let n := 117
  m + n = 184 :=
by sorry

end NUMINAMATH_GPT_shen_winning_probability_sum_l1024_102442


namespace NUMINAMATH_GPT_area_at_stage_8_l1024_102444

theorem area_at_stage_8 
  (side_length : ℕ)
  (stage : ℕ)
  (num_squares : ℕ)
  (square_area : ℕ) 
  (total_area : ℕ) 
  (h1 : side_length = 4) 
  (h2 : stage = 8) 
  (h3 : num_squares = stage) 
  (h4 : square_area = side_length * side_length) 
  (h5 : total_area = num_squares * square_area) :
  total_area = 128 :=
sorry

end NUMINAMATH_GPT_area_at_stage_8_l1024_102444


namespace NUMINAMATH_GPT_intersection_of_sets_l1024_102411

noncomputable def universal_set (x : ℝ) := true

def set_A (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0

def set_B (x : ℝ) : Prop := ∃ y, y = Real.log (1 - x)

def complement_U_B (x : ℝ) : Prop := ¬ set_B x

theorem intersection_of_sets :
  { x : ℝ | set_A x } ∩ { x | complement_U_B x } = { x : ℝ | 1 ≤ x ∧ x < 3 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1024_102411


namespace NUMINAMATH_GPT_common_ratio_geometric_series_l1024_102431

theorem common_ratio_geometric_series {a r S : ℝ} (h₁ : S = (a / (1 - r))) (h₂ : (ar^4 / (1 - r)) = S / 64) (h₃ : S ≠ 0) : r = 1 / 2 :=
sorry

end NUMINAMATH_GPT_common_ratio_geometric_series_l1024_102431


namespace NUMINAMATH_GPT_multiplication_even_a_b_multiplication_even_a_a_l1024_102423

def a : Int := 4
def b : Int := 3

theorem multiplication_even_a_b : a * b = 12 := by sorry
theorem multiplication_even_a_a : a * a = 16 := by sorry

end NUMINAMATH_GPT_multiplication_even_a_b_multiplication_even_a_a_l1024_102423


namespace NUMINAMATH_GPT_trains_cross_time_l1024_102465

theorem trains_cross_time
  (length : ℝ)
  (time1 time2 : ℝ)
  (h_length : length = 120)
  (h_time1 : time1 = 5)
  (h_time2 : time2 = 15)
  : (2 * length) / ((length / time1) + (length / time2)) = 7.5 := by
  sorry

end NUMINAMATH_GPT_trains_cross_time_l1024_102465


namespace NUMINAMATH_GPT_PropositionA_PropositionB_PropositionC_PropositionD_l1024_102457

-- Proposition A (Incorrect)
theorem PropositionA : ¬(∀ a b c : ℝ, a > b ∧ b > 0 → a * c^2 > b * c^2) :=
sorry

-- Proposition B (Correct)
theorem PropositionB : ∀ a b : ℝ, -2 < a ∧ a < 3 ∧ 1 < b ∧ b < 2 → -4 < a - b ∧ a - b < 2 :=
sorry

-- Proposition C (Correct)
theorem PropositionC : ∀ a b c : ℝ, a > b ∧ b > 0 ∧ c < 0 → c / (a^2) > c / (b^2) :=
sorry

-- Proposition D (Incorrect)
theorem PropositionD : ¬(∀ a b c : ℝ, c > a ∧ a > b → a / (c - a) > b / (c - b)) :=
sorry

end NUMINAMATH_GPT_PropositionA_PropositionB_PropositionC_PropositionD_l1024_102457


namespace NUMINAMATH_GPT_percent_increase_hypotenuse_l1024_102439

theorem percent_increase_hypotenuse :
  let l1 := 3
  let l2 := 1.25 * l1
  let l3 := 1.25 * l2
  let l4 := 1.25 * l3
  let h1 := l1 * Real.sqrt 2
  let h4 := l4 * Real.sqrt 2
  ((h4 - h1) / h1) * 100 = 95.3 :=
by
  sorry

end NUMINAMATH_GPT_percent_increase_hypotenuse_l1024_102439


namespace NUMINAMATH_GPT_product_of_coefficients_is_negative_integer_l1024_102404

theorem product_of_coefficients_is_negative_integer
  (a b c : ℤ)
  (habc_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (discriminant_positive : (b * b - 4 * a * c) > 0)
  (product_cond : a * b * c = (c / a)) :
  ∃ k : ℤ, k < 0 ∧ k = a * b * c :=
by
  sorry

end NUMINAMATH_GPT_product_of_coefficients_is_negative_integer_l1024_102404


namespace NUMINAMATH_GPT_total_money_l1024_102499

theorem total_money (A B C : ℝ) (h1 : A = 1 / 2 * (B + C))
  (h2 : B = 2 / 3 * (A + C)) (h3 : A = 122) :
  A + B + C = 366 := by
  sorry

end NUMINAMATH_GPT_total_money_l1024_102499


namespace NUMINAMATH_GPT_valid_parameterizations_l1024_102408

def point_on_line (x y : ℝ) : Prop := (y = 2 * x - 5)

def direction_vector_valid (vx vy : ℝ) : Prop := (∃ (k : ℝ), vx = k * 1 ∧ vy = k * 2)

def parametric_option_valid (px py vx vy : ℝ) : Prop := 
  point_on_line px py ∧ direction_vector_valid vx vy

theorem valid_parameterizations : 
  (parametric_option_valid 10 15 5 10) ∧ 
  (parametric_option_valid 3 1 0.5 1) ∧ 
  (parametric_option_valid 7 9 2 4) ∧ 
  (parametric_option_valid 0 (-5) 10 20) :=
  by sorry

end NUMINAMATH_GPT_valid_parameterizations_l1024_102408


namespace NUMINAMATH_GPT_total_amount_shared_l1024_102472

theorem total_amount_shared (a b c : ℕ) (h_ratio : a * 5 = b * 3) (h_ben : b = 25) (h_ratio_ben : b * 12 = c * 5) :
  a + b + c = 100 := by
  sorry

end NUMINAMATH_GPT_total_amount_shared_l1024_102472
