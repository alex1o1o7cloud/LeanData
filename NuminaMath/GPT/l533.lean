import Mathlib

namespace NUMINAMATH_GPT_exists_x1_x2_l533_53326

noncomputable def f (a x : ℝ) := a * x + Real.log x

theorem exists_x1_x2 (a : ℝ) (h : a < 0) :
  ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ f a x1 ≥ f a x2 :=
by
  sorry

end NUMINAMATH_GPT_exists_x1_x2_l533_53326


namespace NUMINAMATH_GPT_polynomial_characterization_l533_53352

noncomputable def homogeneous_polynomial (P : ℝ → ℝ → ℝ) (n : ℕ) :=
  ∀ t x y : ℝ, P (t * x) (t * y) = t^n * P x y

def polynomial_condition (P : ℝ → ℝ → ℝ) :=
  ∀ a b c : ℝ, P (a + b) c + P (b + c) a + P (c + a) b = 0

def P_value (P : ℝ → ℝ → ℝ) :=
  P 1 0 = 1

theorem polynomial_characterization (P : ℝ → ℝ → ℝ) (n : ℕ) :
  homogeneous_polynomial P n →
  polynomial_condition P →
  P_value P →
  ∃ A : ℝ → ℝ → ℝ, ∀ x y : ℝ, P x y = (x + y)^(n - 1) * (x - 2 * y) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_characterization_l533_53352


namespace NUMINAMATH_GPT_episodes_per_season_before_loss_l533_53321

-- Define the given conditions
def initial_total_seasons : ℕ := 12 + 14
def episodes_lost_per_season : ℕ := 2
def remaining_episodes : ℕ := 364
def total_episodes_lost : ℕ := 12 * episodes_lost_per_season + 14 * episodes_lost_per_season
def initial_total_episodes : ℕ := remaining_episodes + total_episodes_lost

-- Define the theorem to prove
theorem episodes_per_season_before_loss : initial_total_episodes / initial_total_seasons = 16 :=
by
  sorry

end NUMINAMATH_GPT_episodes_per_season_before_loss_l533_53321


namespace NUMINAMATH_GPT_students_between_min_and_hos_l533_53391

theorem students_between_min_and_hos
  (total_students : ℕ)
  (minyoung_left_position : ℕ)
  (hoseok_right_position : ℕ)
  (total_students_eq : total_students = 13)
  (minyoung_left_position_eq : minyoung_left_position = 8)
  (hoseok_right_position_eq : hoseok_right_position = 9) :
  (minyoung_left_position - (total_students - hoseok_right_position + 1) - 1) = 2 := 
by
  sorry

end NUMINAMATH_GPT_students_between_min_and_hos_l533_53391


namespace NUMINAMATH_GPT_dot_product_example_l533_53347

def vector := ℝ × ℝ

-- Define the dot product function
def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_example : dot_product (-1, 0) (0, 2) = 0 := by
  sorry

end NUMINAMATH_GPT_dot_product_example_l533_53347


namespace NUMINAMATH_GPT_find_intersection_l533_53313

noncomputable def setM : Set ℝ := {x : ℝ | x^2 ≤ 9}
noncomputable def setN : Set ℝ := {x : ℝ | x ≤ 1}
noncomputable def intersection : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 1}

theorem find_intersection (x : ℝ) : (x ∈ setM ∧ x ∈ setN) ↔ (x ∈ intersection) := 
by sorry

end NUMINAMATH_GPT_find_intersection_l533_53313


namespace NUMINAMATH_GPT_outlined_square_digit_l533_53382

theorem outlined_square_digit :
  ∀ (digit : ℕ), (digit ∈ {n | ∃ (m : ℕ), 10 ≤ 3^m ∧ 3^m < 1000 ∧ digit = (3^m / 10) % 10 }) →
  (digit ∈ {n | ∃ (n : ℕ), 10 ≤ 7^n ∧ 7^n < 1000 ∧ digit = (7^n / 10) % 10 }) →
  digit = 4 :=
by sorry

end NUMINAMATH_GPT_outlined_square_digit_l533_53382


namespace NUMINAMATH_GPT_taco_castle_num_dodge_trucks_l533_53331

theorem taco_castle_num_dodge_trucks
  (D F T V H C : ℕ)
  (hV : V = 5)
  (h1 : F = D / 3)
  (h2 : F = 2 * T)
  (h3 : V = T / 2)
  (h4 : H = 3 * F / 4)
  (h5 : C = 2 * H / 3) :
  D = 60 :=
by
  sorry

end NUMINAMATH_GPT_taco_castle_num_dodge_trucks_l533_53331


namespace NUMINAMATH_GPT_sqrt_nested_expression_l533_53334

theorem sqrt_nested_expression : 
  Real.sqrt (32 * Real.sqrt (16 * Real.sqrt (8 * Real.sqrt 4))) = 16 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_nested_expression_l533_53334


namespace NUMINAMATH_GPT_AB_not_together_correct_l533_53376

-- Definitions based on conditions
def total_people : ℕ := 5

-- The result from the complementary counting principle
def total_arrangements : ℕ := 120
def AB_together_arrangements : ℕ := 48

-- The arrangement count of A and B not next to each other
def AB_not_together_arrangements : ℕ := total_arrangements - AB_together_arrangements

theorem AB_not_together_correct : 
  AB_not_together_arrangements = 72 :=
sorry

end NUMINAMATH_GPT_AB_not_together_correct_l533_53376


namespace NUMINAMATH_GPT_gcd_of_7854_and_15246_is_6_six_is_not_prime_l533_53387

theorem gcd_of_7854_and_15246_is_6 : gcd 7854 15246 = 6 := sorry

theorem six_is_not_prime : ¬ Prime 6 := sorry

end NUMINAMATH_GPT_gcd_of_7854_and_15246_is_6_six_is_not_prime_l533_53387


namespace NUMINAMATH_GPT_probability_selecting_girl_l533_53325

def boys : ℕ := 3
def girls : ℕ := 1
def total_candidates : ℕ := boys + girls
def favorable_outcomes : ℕ := girls

theorem probability_selecting_girl : 
  ∃ p : ℚ, p = (favorable_outcomes : ℚ) / (total_candidates : ℚ) ∧ p = 1 / 4 :=
sorry

end NUMINAMATH_GPT_probability_selecting_girl_l533_53325


namespace NUMINAMATH_GPT_rectangular_region_area_l533_53344

-- Definitions based on conditions
variable (w : ℝ) -- length of the shorter sides
variable (l : ℝ) -- length of the longer side
variable (total_fence_length : ℝ) -- total length of the fence

-- Given conditions as hypotheses
theorem rectangular_region_area
  (h1 : l = 2 * w) -- The length of the side opposite the wall is twice the length of each of the other two fenced sides
  (h2 : w + w + l = total_fence_length) -- The total length of the fence is 40 feet
  (h3 : total_fence_length = 40) -- total fence length of 40 feet
: (w * l) = 200 := -- The area of the rectangular region is 200 square feet
sorry

end NUMINAMATH_GPT_rectangular_region_area_l533_53344


namespace NUMINAMATH_GPT_min_value_inequality_l533_53343

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  6 * x / (2 * y + z) + 3 * y / (x + 2 * z) + 9 * z / (x + y) ≥ 83 :=
sorry

end NUMINAMATH_GPT_min_value_inequality_l533_53343


namespace NUMINAMATH_GPT_positive_integers_divisors_of_2_to_the_n_plus_1_l533_53302

theorem positive_integers_divisors_of_2_to_the_n_plus_1:
  ∀ n : ℕ, 0 < n → (n^2 ∣ 2^n + 1) ↔ (n = 1 ∨ n = 3) :=
by
  sorry

end NUMINAMATH_GPT_positive_integers_divisors_of_2_to_the_n_plus_1_l533_53302


namespace NUMINAMATH_GPT_sin_225_eq_neg_sqrt2_over_2_l533_53311

theorem sin_225_eq_neg_sqrt2_over_2 : Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_GPT_sin_225_eq_neg_sqrt2_over_2_l533_53311


namespace NUMINAMATH_GPT_mary_added_peanuts_l533_53398

theorem mary_added_peanuts (initial final added : Nat) 
  (h1 : initial = 4)
  (h2 : final = 16)
  (h3 : final = initial + added) : 
  added = 12 := 
by {
  sorry
}

end NUMINAMATH_GPT_mary_added_peanuts_l533_53398


namespace NUMINAMATH_GPT_rectangle_dimensions_l533_53340

theorem rectangle_dimensions (w l : ℝ) 
  (h1 : l = 3 * w) 
  (h2 : 2 * (l + w) = 2 * l * w) : 
  w = 4 / 3 ∧ l = 4 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_dimensions_l533_53340


namespace NUMINAMATH_GPT_functional_eq_solution_l533_53385

variable (f g : ℝ → ℝ)

theorem functional_eq_solution (h : ∀ x y : ℝ, f (x + y * g x) = g x + x * f y) : f = id := 
sorry

end NUMINAMATH_GPT_functional_eq_solution_l533_53385


namespace NUMINAMATH_GPT_linear_transformation_proof_l533_53307

theorem linear_transformation_proof (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ 1) :
  ∃ (k b : ℝ), k = 4 ∧ b = -1 ∧ (y = k * x + b ∧ -1 ≤ y ∧ y ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_linear_transformation_proof_l533_53307


namespace NUMINAMATH_GPT_value_of_b_l533_53355

theorem value_of_b (a b : ℕ) (h1 : a * b = 2 * (a + b) + 10) (h2 : b - a = 5) : b = 9 := 
by {
  -- Proof is not required, so we use sorry to complete the statement
  sorry
}

end NUMINAMATH_GPT_value_of_b_l533_53355


namespace NUMINAMATH_GPT_james_farmer_walk_distance_l533_53336

theorem james_farmer_walk_distance (d : ℝ) :
  ∃ d : ℝ,
    (∀ w : ℝ, (w = 300 + 50 → d = 20) ∧ 
             (w' = w * 1.30 ∧ w'' = w' * 1.20 → w'' = 546)) :=
by
  sorry

end NUMINAMATH_GPT_james_farmer_walk_distance_l533_53336


namespace NUMINAMATH_GPT_sum_of_all_possible_values_is_correct_l533_53373

noncomputable def M_sum_of_all_possible_values (a b c M : ℝ) : Prop :=
  M = a * b * c ∧ M = 8 * (a + b + c) ∧ c = a + b ∧ b = 2 * a

theorem sum_of_all_possible_values_is_correct :
  ∃ M, (∃ a b c, M_sum_of_all_possible_values a b c M) ∧ M = 96 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_sum_of_all_possible_values_is_correct_l533_53373


namespace NUMINAMATH_GPT_augmented_matrix_solution_l533_53346

theorem augmented_matrix_solution (c1 c2 : ℚ) 
    (h1 : 2 * (3 : ℚ) + 3 * (5 : ℚ) = c1)
    (h2 : (5 : ℚ) = c2) : 
    c1 - c2 = 16 := 
by 
  sorry

end NUMINAMATH_GPT_augmented_matrix_solution_l533_53346


namespace NUMINAMATH_GPT_new_average_l533_53357

open Nat

-- The Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

-- Sum of the first 35 Fibonacci numbers
def sum_fibonacci_first_35 : ℕ :=
  (List.range 35).map fibonacci |>.sum -- or critical to use: List.foldr (λ x acc, fibonacci x + acc) 0 (List.range 35) 

theorem new_average (n : ℕ) (avg : ℕ) (Fib_Sum : ℕ) 
  (h₁ : n = 35) 
  (h₂ : avg = 25) 
  (h₃ : Fib_Sum = sum_fibonacci_first_35) : 
  (25 * Fib_Sum / 35) = avg * (sum_fibonacci_first_35) / n := 
by 
  sorry

end NUMINAMATH_GPT_new_average_l533_53357


namespace NUMINAMATH_GPT_volume_of_rice_pile_l533_53365

theorem volume_of_rice_pile
  (arc_length_bottom : ℝ)
  (height : ℝ)
  (one_fourth_cone : ℝ)
  (approx_pi : ℝ)
  (h_arc : arc_length_bottom = 8)
  (h_height : height = 5)
  (h_one_fourth_cone : one_fourth_cone = 1/4)
  (h_approx_pi : approx_pi = 3) :
  ∃ V : ℝ, V = one_fourth_cone * (1 / 3) * π * (16^2 / π^2) * height :=
by
  sorry

end NUMINAMATH_GPT_volume_of_rice_pile_l533_53365


namespace NUMINAMATH_GPT_min_max_of_f_l533_53367

def f (x : ℝ) : ℝ := -2 * x + 1

-- defining the minimum and maximum values
def min_val : ℝ := -3
def max_val : ℝ := 5

theorem min_max_of_f :
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f x ≥ min_val) ∧ 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f x ≤ max_val) :=
by 
  sorry

end NUMINAMATH_GPT_min_max_of_f_l533_53367


namespace NUMINAMATH_GPT_larger_number_of_product_and_sum_l533_53392

theorem larger_number_of_product_and_sum (x y : ℕ) (h_prod : x * y = 35) (h_sum : x + y = 12) : max x y = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_larger_number_of_product_and_sum_l533_53392


namespace NUMINAMATH_GPT_first_proof_l533_53332

def triangular (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

def covers_all_columns (k : ℕ) : Prop :=
  ∀ c : ℕ, (c < 10) → (∃ m : ℕ, m ≤ k ∧ (triangular m) % 10 = c)

theorem first_proof (k : ℕ) (h : covers_all_columns 28) : 
  triangular k = 435 :=
sorry

end NUMINAMATH_GPT_first_proof_l533_53332


namespace NUMINAMATH_GPT_tan_eq_example_l533_53338

theorem tan_eq_example (x : ℝ) (hx : Real.tan (3 * x) * Real.tan (5 * x) = Real.tan (7 * x) * Real.tan (9 * x)) : x = 30 * Real.pi / 180 :=
  sorry

end NUMINAMATH_GPT_tan_eq_example_l533_53338


namespace NUMINAMATH_GPT_wand_cost_l533_53386

theorem wand_cost (c : ℕ) (h1 : 3 * c = 3 * c) (h2 : 2 * (c + 5) = 130) : c = 60 :=
by
  sorry

end NUMINAMATH_GPT_wand_cost_l533_53386


namespace NUMINAMATH_GPT_relationship_P_Q_l533_53366

variable (a : ℝ)
variable (P : ℝ := Real.sqrt a + Real.sqrt (a + 5))
variable (Q : ℝ := Real.sqrt (a + 2) + Real.sqrt (a + 3))

theorem relationship_P_Q (h : 0 ≤ a) : P < Q :=
by
  sorry

end NUMINAMATH_GPT_relationship_P_Q_l533_53366


namespace NUMINAMATH_GPT_stickers_after_loss_l533_53323

-- Conditions
def stickers_per_page : ℕ := 20
def initial_pages : ℕ := 12
def lost_pages : ℕ := 1

-- Problem statement
theorem stickers_after_loss : (initial_pages - lost_pages) * stickers_per_page = 220 := by
  sorry

end NUMINAMATH_GPT_stickers_after_loss_l533_53323


namespace NUMINAMATH_GPT_workers_together_time_l533_53315

theorem workers_together_time (hA : ℝ) (hB : ℝ) (jobA_time : hA = 10) (jobB_time : hB = 12) : 
  1 / ((1 / hA) + (1 / hB)) = (60 / 11) :=
by
  -- skipping the proof details
  sorry

end NUMINAMATH_GPT_workers_together_time_l533_53315


namespace NUMINAMATH_GPT_inequality_proof_l533_53393

-- Define the context of non-negative real numbers and sum to 1
variable {x y z : ℝ}
variable (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z)
variable (h_sum : x + y + z = 1)

-- State the theorem to be proved
theorem inequality_proof (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : x + y + z = 1) :
    0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
    sorry

end NUMINAMATH_GPT_inequality_proof_l533_53393


namespace NUMINAMATH_GPT_range_of_x_for_expression_meaningful_l533_53303

theorem range_of_x_for_expression_meaningful (x : ℝ) :
  (x - 1 > 0 ∧ x ≠ 1) ↔ x > 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_for_expression_meaningful_l533_53303


namespace NUMINAMATH_GPT_required_line_equation_l533_53358

-- Define the point P
structure Point where
  x : ℝ
  y : ℝ

-- Line structure with general form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- A point P on a line
def on_line (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

-- Perpendicular condition between two lines
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- The known line
def known_line : Line := {a := 1, b := -2, c := 3}

-- The given point
def P : Point := {x := -1, y := 3}

noncomputable def required_line : Line := {a := 2, b := 1, c := -1}

-- The theorem to be proved
theorem required_line_equation (l : Line) (P : Point) :
  (on_line P l) ∧ (perpendicular l known_line) ↔ l = required_line :=
  by
    sorry

end NUMINAMATH_GPT_required_line_equation_l533_53358


namespace NUMINAMATH_GPT_right_triangle_one_leg_div_by_3_l533_53324

theorem right_triangle_one_leg_div_by_3 {a b c : ℕ} (a_pos : 0 < a) (b_pos : 0 < b) 
  (h : a^2 + b^2 = c^2) : 3 ∣ a ∨ 3 ∣ b := 
by 
  sorry

end NUMINAMATH_GPT_right_triangle_one_leg_div_by_3_l533_53324


namespace NUMINAMATH_GPT_value_of_m_l533_53318

theorem value_of_m (m : ℤ) (h1 : abs m = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l533_53318


namespace NUMINAMATH_GPT_lines_divide_circle_into_four_arcs_l533_53308

theorem lines_divide_circle_into_four_arcs (a b : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 1 → y = x + a ∨ y = x + b) →
  a^2 + b^2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_lines_divide_circle_into_four_arcs_l533_53308


namespace NUMINAMATH_GPT_reflection_equation_l533_53350

theorem reflection_equation
  (incident_line : ∀ x y : ℝ, 2 * x - y + 2 = 0)
  (reflection_axis : ∀ x y : ℝ, x + y - 5 = 0) :
  ∃ x y : ℝ, x - 2 * y + 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_reflection_equation_l533_53350


namespace NUMINAMATH_GPT_birds_find_more_than_half_millet_on_sunday_l533_53301

noncomputable def seed_millet_fraction : ℕ → ℚ
| 0 => 2 * 0.2 -- initial amount on Day 1 (Monday)
| (n+1) => 0.7 * seed_millet_fraction n + 0.4

theorem birds_find_more_than_half_millet_on_sunday :
  let dayMillets : ℕ := 7
  let total_seeds : ℚ := 2
  let half_seeds : ℚ := total_seeds / 2
  (seed_millet_fraction dayMillets > half_seeds) := by
    sorry

end NUMINAMATH_GPT_birds_find_more_than_half_millet_on_sunday_l533_53301


namespace NUMINAMATH_GPT_gracie_height_is_56_l533_53378

noncomputable def Gracie_height : Nat := 56

theorem gracie_height_is_56 : Gracie_height = 56 := by
  sorry

end NUMINAMATH_GPT_gracie_height_is_56_l533_53378


namespace NUMINAMATH_GPT_rightmost_three_digits_of_7_pow_2023_l533_53312

theorem rightmost_three_digits_of_7_pow_2023 :
  (7 ^ 2023) % 1000 = 637 :=
sorry

end NUMINAMATH_GPT_rightmost_three_digits_of_7_pow_2023_l533_53312


namespace NUMINAMATH_GPT_minimum_time_for_tomato_egg_soup_l533_53356

noncomputable def cracking_egg_time : ℕ := 1
noncomputable def washing_chopping_tomatoes_time : ℕ := 2
noncomputable def boiling_tomatoes_time : ℕ := 3
noncomputable def adding_eggs_heating_time : ℕ := 1
noncomputable def stirring_egg_time : ℕ := 1

theorem minimum_time_for_tomato_egg_soup :
  washing_chopping_tomatoes_time + boiling_tomatoes_time + adding_eggs_heating_time = 6 :=
by
  -- proof to be filled
  sorry

end NUMINAMATH_GPT_minimum_time_for_tomato_egg_soup_l533_53356


namespace NUMINAMATH_GPT_symmetric_coordinates_l533_53375

structure Point :=
  (x : Int)
  (y : Int)

def symmetric_about_origin (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

theorem symmetric_coordinates (P : Point) (h : P = Point.mk (-1) 2) :
  symmetric_about_origin P = Point.mk 1 (-2) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_coordinates_l533_53375


namespace NUMINAMATH_GPT_point_transformation_l533_53327

theorem point_transformation (a b : ℝ) :
  let P := (a, b)
  let P₁ := (2 * 2 - a, 2 * 3 - b) -- Rotate P 180° counterclockwise around (2, 3)
  let P₂ := (P₁.2, P₁.1)           -- Reflect P₁ about the line y = x
  P₂ = (5, -4) → a - b = 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_point_transformation_l533_53327


namespace NUMINAMATH_GPT_percentage_of_filled_seats_l533_53333

theorem percentage_of_filled_seats (total_seats vacant_seats : ℕ) (h_total : total_seats = 600) (h_vacant : vacant_seats = 240) :
  (total_seats - vacant_seats) * 100 / total_seats = 60 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_filled_seats_l533_53333


namespace NUMINAMATH_GPT_sum_of_x_and_y_greater_equal_twice_alpha_l533_53309

theorem sum_of_x_and_y_greater_equal_twice_alpha (x y α : ℝ) 
  (h : Real.sqrt (1 + x) + Real.sqrt (1 + y) = 2 * Real.sqrt (1 + α)) :
  x + y ≥ 2 * α :=
sorry

end NUMINAMATH_GPT_sum_of_x_and_y_greater_equal_twice_alpha_l533_53309


namespace NUMINAMATH_GPT_min_value_of_expression_l533_53328

theorem min_value_of_expression (n : ℕ) (h_pos : n > 0) : n = 8 → (n / 2 + 32 / n) = 8 :=
by sorry

end NUMINAMATH_GPT_min_value_of_expression_l533_53328


namespace NUMINAMATH_GPT_angle_rotation_l533_53381

theorem angle_rotation (α : ℝ) (β : ℝ) (k : ℤ) :
  (∃ k' : ℤ, α + 30 = 120 + 360 * k') →
  (β = 360 * k + 90) ↔ (∃ k'' : ℤ, β = 360 * k'' + α) :=
by
  sorry

end NUMINAMATH_GPT_angle_rotation_l533_53381


namespace NUMINAMATH_GPT_plane_equation_l533_53306

variable (x y z : ℝ)

def pointA : ℝ × ℝ × ℝ := (3, 0, 0)
def normalVector : ℝ × ℝ × ℝ := (2, -3, 1)

theorem plane_equation : 
  ∃ a b c d, normalVector = (a, b, c) ∧ pointA = (x, y, z) ∧ a * (x - 3) + b * y + c * z = d ∧ d = -6 := 
  sorry

end NUMINAMATH_GPT_plane_equation_l533_53306


namespace NUMINAMATH_GPT_minimum_value_quadratic_function_l533_53396

-- Defining the quadratic function y
def quadratic_function (x : ℝ) : ℝ := 4 * x^2 + 8 * x + 16

-- Statement asserting the minimum value of the quadratic function
theorem minimum_value_quadratic_function : ∃ (y_min : ℝ), (∀ x : ℝ, quadratic_function x ≥ y_min) ∧ y_min = 12 :=
by
  -- Here we would normally insert the proof, but we skip it with sorry
  sorry

end NUMINAMATH_GPT_minimum_value_quadratic_function_l533_53396


namespace NUMINAMATH_GPT_students_taking_neither_l533_53317

def total_students : ℕ := 1200
def music_students : ℕ := 60
def art_students : ℕ := 80
def sports_students : ℕ := 30
def music_and_art_students : ℕ := 25
def music_and_sports_students : ℕ := 15
def art_and_sports_students : ℕ := 20
def all_three_students : ℕ := 10

theorem students_taking_neither :
  total_students - (music_students + art_students + sports_students 
  - music_and_art_students - music_and_sports_students - art_and_sports_students 
  + all_three_students) = 1080 := sorry

end NUMINAMATH_GPT_students_taking_neither_l533_53317


namespace NUMINAMATH_GPT_double_rooms_booked_l533_53304

theorem double_rooms_booked (S D : ℕ) 
  (h1 : S + D = 260) 
  (h2 : 35 * S + 60 * D = 14000) : 
  D = 196 :=
by
  sorry

end NUMINAMATH_GPT_double_rooms_booked_l533_53304


namespace NUMINAMATH_GPT_minimum_value_of_quadratic_function_l533_53339

def quadratic_function (a x : ℝ) : ℝ :=
  4 * x ^ 2 - 4 * a * x + (a ^ 2 - 2 * a + 2)

def min_value_in_interval (f : ℝ → ℝ) (a : ℝ) (interval : Set ℝ) (min_val : ℝ) : Prop :=
  ∀ x ∈ interval, f x ≥ min_val ∧ ∃ y ∈ interval, f y = min_val

theorem minimum_value_of_quadratic_function :
  ∃ a : ℝ, min_value_in_interval (quadratic_function a) a {x | 0 ≤ x ∧ x ≤ 1} 2 ↔ (a = 0 ∨ a = 3 + Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_quadratic_function_l533_53339


namespace NUMINAMATH_GPT_louisa_average_speed_l533_53364

theorem louisa_average_speed :
  ∃ v : ℝ, 
  (100 / v = 175 / v - 3) ∧ 
  v = 25 :=
by
  sorry

end NUMINAMATH_GPT_louisa_average_speed_l533_53364


namespace NUMINAMATH_GPT_find_positive_integers_divisors_l533_53314

theorem find_positive_integers_divisors :
  ∃ n_list : List ℕ, 
    (∀ n ∈ n_list, n > 0 ∧ (n * (n + 1)) / 2 ∣ 10 * n) ∧ n_list.length = 5 :=
sorry

end NUMINAMATH_GPT_find_positive_integers_divisors_l533_53314


namespace NUMINAMATH_GPT_initial_number_divisible_by_15_l533_53335

theorem initial_number_divisible_by_15 (N : ℕ) (h : (N - 7) % 15 = 0) : N = 22 := 
by
  sorry

end NUMINAMATH_GPT_initial_number_divisible_by_15_l533_53335


namespace NUMINAMATH_GPT_solve_system_l533_53390

theorem solve_system (x y : ℝ) (h1 : x + 3 * y = 20) (h2 : x + y = 10) : x = 5 ∧ y = 5 := 
by 
  sorry

end NUMINAMATH_GPT_solve_system_l533_53390


namespace NUMINAMATH_GPT_largest_y_coordinate_l533_53372

theorem largest_y_coordinate (x y : ℝ) (h : (x^2 / 25) + ((y - 3)^2 / 25) = 0) : y = 3 := by
  sorry

end NUMINAMATH_GPT_largest_y_coordinate_l533_53372


namespace NUMINAMATH_GPT_total_profit_correct_l533_53394

-- We define the conditions
variables (a m : ℝ)

-- The item's cost per piece
def cost_per_piece : ℝ := a
-- The markup percentage
def markup_percentage : ℝ := 0.20
-- The discount percentage
def discount_percentage : ℝ := 0.10
-- The number of pieces sold
def pieces_sold : ℝ := m

-- Definitions derived from conditions
def selling_price_markup : ℝ := cost_per_piece a * (1 + markup_percentage)
def selling_price_discount : ℝ := selling_price_markup a * (1 - discount_percentage)
def profit_per_piece : ℝ := selling_price_discount a - cost_per_piece a
def total_profit : ℝ := profit_per_piece a * pieces_sold m

theorem total_profit_correct (a m : ℝ) : total_profit a m = 0.08 * a * m :=
by sorry

end NUMINAMATH_GPT_total_profit_correct_l533_53394


namespace NUMINAMATH_GPT_exists_smallest_n_l533_53319

theorem exists_smallest_n :
  ∃ n : ℕ, (n^2 + 20 * n + 19) % 2019 = 0 ∧ n = 2000 :=
sorry

end NUMINAMATH_GPT_exists_smallest_n_l533_53319


namespace NUMINAMATH_GPT_lawn_length_is_70_l533_53379

-- Definitions for conditions
def width_of_lawn : ℕ := 50
def road_width : ℕ := 10
def cost_of_roads : ℕ := 3600
def cost_per_sqm : ℕ := 3

-- Proof problem
theorem lawn_length_is_70 :
  ∃ L : ℕ, 10 * L + 10 * width_of_lawn = cost_of_roads / cost_per_sqm ∧ L = 70 := by
  sorry

end NUMINAMATH_GPT_lawn_length_is_70_l533_53379


namespace NUMINAMATH_GPT_money_distribution_l533_53362

variable (A B C : ℝ)

theorem money_distribution
  (h₁ : A + B + C = 500)
  (h₂ : A + C = 200)
  (h₃ : C = 60) :
  B + C = 360 :=
by
  sorry

end NUMINAMATH_GPT_money_distribution_l533_53362


namespace NUMINAMATH_GPT_sequence_general_term_and_sum_sum_tn_bound_l533_53329

theorem sequence_general_term_and_sum (c : ℝ) (h₁ : c = 1) 
  (f : ℕ → ℝ) (hf : ∀ x, f x = (1 / 3) ^ x) :
  (∀ n, a_n = -2 / 3 ^ n) ∧ (∀ n, b_n = 2 * n - 1) :=
by {
  sorry
}

theorem sum_tn_bound (h₂ : ∀ n > 0, T_n = (1 / 2) * (1 - 1 / (2 * n + 1))) :
  ∃ n, T_n > 1005 / 2014 ∧ n = 252 :=
by {
  sorry
}

end NUMINAMATH_GPT_sequence_general_term_and_sum_sum_tn_bound_l533_53329


namespace NUMINAMATH_GPT_infant_weight_in_4th_month_l533_53300

-- Given conditions
def a : ℕ := 3000
def x : ℕ := 4
def y : ℕ := a + 700 * x

-- Theorem stating the weight of the infant in the 4th month equals 5800 grams
theorem infant_weight_in_4th_month : y = 5800 := by
  sorry

end NUMINAMATH_GPT_infant_weight_in_4th_month_l533_53300


namespace NUMINAMATH_GPT_stock_initial_value_l533_53399

theorem stock_initial_value (V : ℕ) (h : ∀ n ≤ 99, V + n = 200 - (99 - n)) : V = 101 :=
sorry

end NUMINAMATH_GPT_stock_initial_value_l533_53399


namespace NUMINAMATH_GPT_candy_bar_calories_l533_53368

theorem candy_bar_calories (calories : ℕ) (bars : ℕ) (dozen : ℕ) (total_calories : ℕ) 
  (H1 : total_calories = 2016) (H2 : bars = 42) (H3 : dozen = 12) 
  (H4 : total_calories = bars * calories) : 
  calories / dozen = 4 := 
by 
  sorry

end NUMINAMATH_GPT_candy_bar_calories_l533_53368


namespace NUMINAMATH_GPT_analysis_method_inequality_l533_53397

def analysis_method_seeks (inequality : Prop) : Prop :=
  ∃ (sufficient_condition : Prop), (inequality → sufficient_condition)

theorem analysis_method_inequality (inequality : Prop) :
  (∃ sufficient_condition, (inequality → sufficient_condition)) :=
sorry

end NUMINAMATH_GPT_analysis_method_inequality_l533_53397


namespace NUMINAMATH_GPT_solve_quadratic_l533_53380

theorem solve_quadratic (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : ∀ x : ℝ, x^2 + c*x + d = 0 ↔ x = c ∨ x = d) : (c, d) = (1, -2) :=
sorry

end NUMINAMATH_GPT_solve_quadratic_l533_53380


namespace NUMINAMATH_GPT_total_area_of_frequency_histogram_l533_53345

theorem total_area_of_frequency_histogram (f : ℝ → ℝ) (h_f : ∀ x, 0 ≤ f x ∧ f x ≤ 1) (integral_f_one : ∫ x, f x = 1) :
  ∫ x, f x = 1 := 
sorry

end NUMINAMATH_GPT_total_area_of_frequency_histogram_l533_53345


namespace NUMINAMATH_GPT_total_seeds_in_garden_l533_53383

-- Definitions based on conditions
def large_bed_rows : Nat := 4
def large_bed_seeds_per_row : Nat := 25
def medium_bed_rows : Nat := 3
def medium_bed_seeds_per_row : Nat := 20
def num_large_beds : Nat := 2
def num_medium_beds : Nat := 2

-- Theorem statement to show total seeds
theorem total_seeds_in_garden : 
  num_large_beds * (large_bed_rows * large_bed_seeds_per_row) + 
  num_medium_beds * (medium_bed_rows * medium_bed_seeds_per_row) = 320 := 
by
  sorry

end NUMINAMATH_GPT_total_seeds_in_garden_l533_53383


namespace NUMINAMATH_GPT_red_ball_second_given_red_ball_first_l533_53395

noncomputable def probability_of_red_second_given_first : ℚ :=
  let totalBalls := 6
  let redBallsOnFirst := 4
  let whiteBalls := 2
  let redBallsOnSecond := 3
  let remainingBalls := 5

  let P_A := redBallsOnFirst / totalBalls
  let P_AB := (redBallsOnFirst / totalBalls) * (redBallsOnSecond / remainingBalls)
  P_AB / P_A

theorem red_ball_second_given_red_ball_first :
  probability_of_red_second_given_first = 3 / 5 :=
sorry

end NUMINAMATH_GPT_red_ball_second_given_red_ball_first_l533_53395


namespace NUMINAMATH_GPT_evaluate_expression_l533_53341

theorem evaluate_expression :
  (2 * 10^3)^3 = 8 * 10^9 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l533_53341


namespace NUMINAMATH_GPT_seven_power_expression_l533_53377

theorem seven_power_expression (x y z : ℝ) (h₀ : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) (h₁ : x + y + z = 0) (h₂ : xy + xz + yz ≠ 0) :
  (x^7 + y^7 + z^7) / (xyz * (x^2 + y^2 + z^2)) = 14 :=
by
  sorry

end NUMINAMATH_GPT_seven_power_expression_l533_53377


namespace NUMINAMATH_GPT_probability_not_miss_is_correct_l533_53374

-- Define the probability that Peter will miss his morning train
def p_miss : ℚ := 5 / 12

-- Define the probability that Peter does not miss his morning train
def p_not_miss : ℚ := 1 - p_miss

-- The theorem to prove
theorem probability_not_miss_is_correct : p_not_miss = 7 / 12 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_probability_not_miss_is_correct_l533_53374


namespace NUMINAMATH_GPT_james_total_beverages_l533_53330

-- Define the initial quantities
def initial_sodas := 4 * 10 + 12
def initial_juice_boxes := 3 * 8 + 5
def initial_water_bottles := 2 * 15
def initial_energy_drinks := 7

-- Define the consumption rates
def mon_to_wed_sodas := 3 * 3
def mon_to_wed_juice_boxes := 2 * 3
def mon_to_wed_water_bottles := 1 * 3

def thu_to_sun_sodas := 2 * 4
def thu_to_sun_juice_boxes := 4 * 4
def thu_to_sun_water_bottles := 1 * 4
def thu_to_sun_energy_drinks := 1 * 4

-- Define total beverages consumed
def total_consumed_sodas := mon_to_wed_sodas + thu_to_sun_sodas
def total_consumed_juice_boxes := mon_to_wed_juice_boxes + thu_to_sun_juice_boxes
def total_consumed_water_bottles := mon_to_wed_water_bottles + thu_to_sun_water_bottles
def total_consumed_energy_drinks := thu_to_sun_energy_drinks

-- Define total beverages consumed by the end of the week
def total_beverages_consumed := total_consumed_sodas + total_consumed_juice_boxes + total_consumed_water_bottles + total_consumed_energy_drinks

-- The theorem statement to prove
theorem james_total_beverages : total_beverages_consumed = 50 :=
  by sorry

end NUMINAMATH_GPT_james_total_beverages_l533_53330


namespace NUMINAMATH_GPT_work_completion_alternate_days_l533_53389

theorem work_completion_alternate_days (h₁ : ∀ (work : ℝ), ∃ a_days : ℝ, a_days = 12 → (∀ t : ℕ, t / a_days <= work / 12))
                                      (h₂ : ∀ (work : ℝ), ∃ b_days : ℝ, b_days = 36 → (∀ t : ℕ, t / b_days <= work / 36)) :
  ∃ days : ℝ, days = 18 := by
  sorry

end NUMINAMATH_GPT_work_completion_alternate_days_l533_53389


namespace NUMINAMATH_GPT_combined_salaries_of_B_C_D_E_l533_53384

theorem combined_salaries_of_B_C_D_E
    (A_salary : ℕ)
    (average_salary_all : ℕ)
    (total_individuals : ℕ)
    (combined_salaries_B_C_D_E : ℕ) :
    A_salary = 8000 →
    average_salary_all = 8800 →
    total_individuals = 5 →
    combined_salaries_B_C_D_E = (average_salary_all * total_individuals) - A_salary →
    combined_salaries_B_C_D_E = 36000 :=
by
  sorry

end NUMINAMATH_GPT_combined_salaries_of_B_C_D_E_l533_53384


namespace NUMINAMATH_GPT_margie_change_l533_53361

def cost_of_banana_cents : ℕ := 30
def cost_of_orange_cents : ℕ := 60
def num_bananas : ℕ := 4
def num_oranges : ℕ := 2
def amount_paid_dollars : ℝ := 10.0

noncomputable def cost_of_banana_dollars := (cost_of_banana_cents : ℝ) / 100
noncomputable def cost_of_orange_dollars := (cost_of_orange_cents : ℝ) / 100

noncomputable def total_cost := 
  (num_bananas * cost_of_banana_dollars) + (num_oranges * cost_of_orange_dollars)

noncomputable def change_received := amount_paid_dollars - total_cost

theorem margie_change : change_received = 7.60 := 
by sorry

end NUMINAMATH_GPT_margie_change_l533_53361


namespace NUMINAMATH_GPT_chocolate_bars_l533_53360

theorem chocolate_bars (num_small_boxes : ℕ) (num_bars_per_box : ℕ) (total_bars : ℕ) (h1 : num_small_boxes = 20) (h2 : num_bars_per_box = 32) (h3 : total_bars = num_small_boxes * num_bars_per_box) :
  total_bars = 640 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_bars_l533_53360


namespace NUMINAMATH_GPT_smallest_n_l533_53369

theorem smallest_n (n : ℕ) (h1 : n > 1) (h2 : 2016 ∣ (3 * n^3 + 2013)) : n = 193 := 
sorry

end NUMINAMATH_GPT_smallest_n_l533_53369


namespace NUMINAMATH_GPT_number_of_boxes_l533_53388

theorem number_of_boxes (total_eggs : ℕ) (eggs_per_box : ℕ) (boxes : ℕ) : 
  total_eggs = 21 → eggs_per_box = 7 → boxes = total_eggs / eggs_per_box → boxes = 3 :=
by
  intros h_total_eggs h_eggs_per_box h_boxes
  rw [h_total_eggs, h_eggs_per_box] at h_boxes
  exact h_boxes

end NUMINAMATH_GPT_number_of_boxes_l533_53388


namespace NUMINAMATH_GPT_avg_speed_BC_60_mph_l533_53305

theorem avg_speed_BC_60_mph 
  (d_AB : ℕ) (d_BC : ℕ) (avg_speed_total : ℚ) (time_ratio : ℚ) (t_AB : ℕ) :
  d_AB = 120 ∧ d_BC = 60 ∧ avg_speed_total = 45 ∧ time_ratio = 3 ∧
  t_AB = 3 → (d_BC / (t_AB / time_ratio) = 60) :=
by
  sorry

end NUMINAMATH_GPT_avg_speed_BC_60_mph_l533_53305


namespace NUMINAMATH_GPT_find_n_l533_53316

noncomputable def tangent_line_problem (x0 : ℝ) (n : ℕ) : Prop :=
(x0 ∈ Set.Ioo (Real.sqrt n) (Real.sqrt (n + 1))) ∧
(∃ m : ℝ, 0 < m ∧ m < 1 ∧ (2 * x0 = 1 / m) ∧ (x0^2 = (Real.log m - 1)))

theorem find_n (x0 : ℝ) (n : ℕ) :
  tangent_line_problem x0 n → n = 2 :=
sorry

end NUMINAMATH_GPT_find_n_l533_53316


namespace NUMINAMATH_GPT_total_cost_of_suits_l533_53322

theorem total_cost_of_suits : 
    ∃ o t : ℕ, o = 300 ∧ t = 3 * o + 200 ∧ o + t = 1400 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_suits_l533_53322


namespace NUMINAMATH_GPT_point_P_outside_circle_l533_53320

theorem point_P_outside_circle (a b : ℝ) (h : ∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) :
  a^2 + b^2 > 1 :=
sorry

end NUMINAMATH_GPT_point_P_outside_circle_l533_53320


namespace NUMINAMATH_GPT_prime_factor_of_T_l533_53337

-- Define constants and conditions
def x : ℕ := 2021
def T : ℕ := Nat.sqrt ((x + x) + (x - x) + (x * x) + (x / x))

-- Define what needs to be proved
theorem prime_factor_of_T : ∃ p : ℕ, Nat.Prime p ∧ Nat.factorization T p > 0 ∧ (∀ q : ℕ, Nat.Prime q ∧ Nat.factorization T q > 0 → q ≤ p) :=
sorry

end NUMINAMATH_GPT_prime_factor_of_T_l533_53337


namespace NUMINAMATH_GPT_ratio_a3_a6_l533_53349

variable (a : ℕ → ℝ) (d : ℝ)
-- aₙ is an arithmetic sequence
variable (h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d)
-- d ≠ 0
variable (h_d_nonzero : d ≠ 0)
-- a₃² = a₁a₉
variable (h_condition : (a 2)^2 = (a 0) * (a 8))

theorem ratio_a3_a6 : (a 2) / (a 5) = 1 / 2 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_ratio_a3_a6_l533_53349


namespace NUMINAMATH_GPT_positive_integers_m_divisors_l533_53363

theorem positive_integers_m_divisors :
  ∃ n, n = 3 ∧ ∀ m : ℕ, (0 < m ∧ ∃ k, 2310 = k * (m^2 + 2)) ↔ m = 1 ∨ m = 2 ∨ m = 3 :=
by
  sorry

end NUMINAMATH_GPT_positive_integers_m_divisors_l533_53363


namespace NUMINAMATH_GPT_rectangle_perimeter_l533_53342

theorem rectangle_perimeter {y x : ℝ} (hxy : x < y) : 
  2 * (y - x) + 2 * x = 2 * y :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l533_53342


namespace NUMINAMATH_GPT_votes_difference_l533_53353

theorem votes_difference (T : ℕ) (V_a : ℕ) (V_f : ℕ) 
  (h1 : T = 330) (h2 : V_a = 40 * T / 100) (h3 : V_f = T - V_a) : V_f - V_a = 66 :=
by
  sorry

end NUMINAMATH_GPT_votes_difference_l533_53353


namespace NUMINAMATH_GPT_car_gasoline_tank_capacity_l533_53370

theorem car_gasoline_tank_capacity
    (speed : ℝ)
    (usage_rate : ℝ)
    (travel_time : ℝ)
    (fraction_used : ℝ)
    (tank_capacity : ℝ)
    (gallons_used : ℝ)
    (distance_traveled : ℝ) :
  speed = 50 →
  usage_rate = 1 / 30 →
  travel_time = 5 →
  fraction_used = 0.5555555555555556 →
  distance_traveled = speed * travel_time →
  gallons_used = distance_traveled * usage_rate →
  gallon_used = tank_capacity * fraction_used →
  tank_capacity = 15 :=
by
  intros hs hr ht hf hd hu hf
  sorry

end NUMINAMATH_GPT_car_gasoline_tank_capacity_l533_53370


namespace NUMINAMATH_GPT_imaginaria_city_population_l533_53310

theorem imaginaria_city_population (a b c : ℕ) (h₁ : a^2 + 225 = b^2 + 1) (h₂ : b^2 + 1 + 75 = c^2) : 5 ∣ a^2 :=
by
  sorry

end NUMINAMATH_GPT_imaginaria_city_population_l533_53310


namespace NUMINAMATH_GPT_time_upstream_is_correct_l533_53351

-- Define the conditions
def speed_of_stream : ℝ := 3
def speed_in_still_water : ℝ := 15
def downstream_time : ℝ := 1
def downstream_speed : ℝ := speed_in_still_water + speed_of_stream
def distance_downstream : ℝ := downstream_speed * downstream_time
def upstream_speed : ℝ := speed_in_still_water - speed_of_stream

-- Theorem statement
theorem time_upstream_is_correct :
  (distance_downstream / upstream_speed) = 1.5 := by
  sorry

end NUMINAMATH_GPT_time_upstream_is_correct_l533_53351


namespace NUMINAMATH_GPT_mans_speed_against_current_l533_53359

theorem mans_speed_against_current (V_with_current V_current V_against : ℝ) (h1 : V_with_current = 21) (h2 : V_current = 4.3) : 
  V_against = V_with_current - 2 * V_current := 
sorry

end NUMINAMATH_GPT_mans_speed_against_current_l533_53359


namespace NUMINAMATH_GPT_personal_income_tax_correct_l533_53354

-- Defining the conditions
def monthly_income : ℕ := 30000
def vacation_bonus : ℕ := 20000
def car_sale_income : ℕ := 250000
def land_purchase_cost : ℕ := 300000

def standard_deduction_car_sale : ℕ := 250000
def property_deduction_land_purchase : ℕ := 300000

-- Define total income
def total_income : ℕ := (monthly_income * 12) + vacation_bonus + car_sale_income

-- Define total deductions
def total_deductions : ℕ := standard_deduction_car_sale + property_deduction_land_purchase

-- Define taxable income (total income - total deductions)
def taxable_income : ℕ := total_income - total_deductions

-- Define tax rate
def tax_rate : ℚ := 0.13

-- Define the correct answer for the tax payable
def tax_payable : ℚ := taxable_income * tax_rate

-- Prove the tax payable is 10400 rubles
theorem personal_income_tax_correct : tax_payable = 10400 := by
  sorry

end NUMINAMATH_GPT_personal_income_tax_correct_l533_53354


namespace NUMINAMATH_GPT_distance_to_grocery_store_l533_53348

-- Definitions of given conditions
def miles_to_mall := 6
def miles_to_pet_store := 5
def miles_back_home := 9
def miles_per_gallon := 15
def cost_per_gallon := 3.5
def total_cost := 7

-- The Lean statement to prove the distance driven to the grocery store.
theorem distance_to_grocery_store (miles_to_mall miles_to_pet_store miles_back_home miles_per_gallon cost_per_gallon total_cost : ℝ) :
(total_cost / cost_per_gallon) * miles_per_gallon - (miles_to_mall + miles_to_pet_store + miles_back_home) = 10 := by
  sorry

end NUMINAMATH_GPT_distance_to_grocery_store_l533_53348


namespace NUMINAMATH_GPT_find_M_l533_53371

theorem find_M (p q r s M : ℚ)
  (h1 : p + q + r + s = 100)
  (h2 : p + 10 = M)
  (h3 : q - 5 = M)
  (h4 : 10 * r = M)
  (h5 : s / 2 = M) :
  M = 1050 / 41 :=
by
  sorry

end NUMINAMATH_GPT_find_M_l533_53371
