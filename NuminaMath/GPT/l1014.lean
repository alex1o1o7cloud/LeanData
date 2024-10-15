import Mathlib

namespace NUMINAMATH_GPT_projectile_first_reaches_28_l1014_101462

theorem projectile_first_reaches_28 (t : ℝ) (h_eq : ∀ t, -4.9 * t^2 + 23.8 * t = 28) : 
    t = 2 :=
sorry

end NUMINAMATH_GPT_projectile_first_reaches_28_l1014_101462


namespace NUMINAMATH_GPT_sequence_is_increasing_l1014_101490

theorem sequence_is_increasing :
  ∀ n m : ℕ, n < m → (1 - 2 / (n + 1) : ℝ) < (1 - 2 / (m + 1) : ℝ) :=
by
  intro n m hnm
  have : (2 : ℝ) / (n + 1) > 2 / (m + 1) :=
    sorry
  linarith [this]

end NUMINAMATH_GPT_sequence_is_increasing_l1014_101490


namespace NUMINAMATH_GPT_tail_to_body_ratio_l1014_101465

variables (B : ℝ) (tail : ℝ := 9) (total_length : ℝ := 30)
variables (head_ratio : ℝ := 1/6)

-- Condition: The overall length is 30 inches
def overall_length_eq : Prop := B + B * head_ratio + tail = total_length

-- Theorem: Ratio of tail length to body length is 1:2
theorem tail_to_body_ratio (h : overall_length_eq B) : tail / B = 1 / 2 :=
sorry

end NUMINAMATH_GPT_tail_to_body_ratio_l1014_101465


namespace NUMINAMATH_GPT_relationship_A_B_l1014_101436

variable (x y : ℝ)

noncomputable def A : ℝ := (x + y) / (1 + x + y)

noncomputable def B : ℝ := (x / (1 + x)) + (y / (1 + y))

theorem relationship_A_B (hx : 0 < x) (hy : 0 < y) : A x y < B x y := sorry

end NUMINAMATH_GPT_relationship_A_B_l1014_101436


namespace NUMINAMATH_GPT_least_positive_integer_with_12_factors_l1014_101488

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_with_12_factors_l1014_101488


namespace NUMINAMATH_GPT_least_possible_number_l1014_101433

theorem least_possible_number :
  ∃ x : ℕ, (∃ q r : ℕ, x = 34 * q + r ∧ 0 ≤ r ∧ r < 34) ∧
            (∃ q' : ℕ, x = 5 * q' ∧ q' = r + 8) ∧
            x = 75 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_number_l1014_101433


namespace NUMINAMATH_GPT_average_score_is_7_stddev_is_2_l1014_101442

-- Define the scores list
def scores : List ℝ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]

-- Proof statement for average score
theorem average_score_is_7 : (scores.sum / scores.length) = 7 :=
by
  simp [scores]
  sorry

-- Proof statement for standard deviation
theorem stddev_is_2 : Real.sqrt ((scores.map (λ x => (x - (scores.sum / scores.length))^2)).sum / scores.length) = 2 :=
by
  simp [scores]
  sorry

end NUMINAMATH_GPT_average_score_is_7_stddev_is_2_l1014_101442


namespace NUMINAMATH_GPT_number_times_frac_eq_cube_l1014_101403

theorem number_times_frac_eq_cube (x : ℕ) : x * (1/6)^2 = 6^3 → x = 7776 :=
by
  intro h
  -- skipped proof
  sorry

end NUMINAMATH_GPT_number_times_frac_eq_cube_l1014_101403


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_l1014_101413

theorem asymptotes_of_hyperbola :
  (∀ x y : ℝ, (x^2 / 16 - y^2 / 25 = 1) →
    (y = (5 / 4) * x ∨ y = -(5 / 4) * x)) :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_l1014_101413


namespace NUMINAMATH_GPT_distance_between_centers_l1014_101493

variable (P R r : ℝ)
variable (h_tangent : P = R - r)
variable (h_radius1 : R = 6)
variable (h_radius2 : r = 3)

theorem distance_between_centers : P = 3 := by
  sorry

end NUMINAMATH_GPT_distance_between_centers_l1014_101493


namespace NUMINAMATH_GPT_shuttle_speed_in_km_per_sec_l1014_101461

variable (speed_mph : ℝ) (miles_to_km : ℝ) (hour_to_sec : ℝ)

theorem shuttle_speed_in_km_per_sec
  (h_speed_mph : speed_mph = 18000)
  (h_miles_to_km : miles_to_km = 1.60934)
  (h_hour_to_sec : hour_to_sec = 3600) :
  (speed_mph * miles_to_km) / hour_to_sec = 8.046 := by
sorry

end NUMINAMATH_GPT_shuttle_speed_in_km_per_sec_l1014_101461


namespace NUMINAMATH_GPT_trajectory_is_parabola_l1014_101460

def distance_to_line (p : ℝ × ℝ) (a : ℝ) : ℝ :=
|p.1 - a|

noncomputable def distance_to_point (p q : ℝ × ℝ) : ℝ :=
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def parabola_condition (P : ℝ × ℝ) : Prop :=
distance_to_line P (-1) + 1 = distance_to_point P (2, 0)

theorem trajectory_is_parabola : ∀ (P : ℝ × ℝ), parabola_condition P ↔
(P.1 + 1)^2 = (Real.sqrt ((P.1 - 2)^2 + P.2^2))^2 := 
by 
  sorry

end NUMINAMATH_GPT_trajectory_is_parabola_l1014_101460


namespace NUMINAMATH_GPT_unique_sum_of_squares_l1014_101477

theorem unique_sum_of_squares (p : ℕ) (k : ℕ) (x y a b : ℤ) 
  (hp : Prime p) (h1 : p = 4 * k + 1) (hx : x^2 + y^2 = p) (ha : a^2 + b^2 = p) :
  (x = a ∨ x = -a) ∧ (y = b ∨ y = -b) ∨ (x = b ∨ x = -b) ∧ (y = a ∨ y = -a) :=
sorry

end NUMINAMATH_GPT_unique_sum_of_squares_l1014_101477


namespace NUMINAMATH_GPT_sequence_general_formula_l1014_101476

theorem sequence_general_formula (n : ℕ) (hn : n > 0) 
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (hS : ∀ n, S n = 1 - n * a n) 
  (hpos : ∀ n, a n > 0) : 
  (a n = 1 / (n * (n + 1))) :=
sorry

end NUMINAMATH_GPT_sequence_general_formula_l1014_101476


namespace NUMINAMATH_GPT_length_of_second_platform_l1014_101459

theorem length_of_second_platform (train_length first_platform_length : ℕ) (time_to_cross_first_platform time_to_cross_second_platform : ℕ) 
  (H1 : train_length = 110) (H2 : first_platform_length = 160) (H3 : time_to_cross_first_platform = 15) 
  (H4 : time_to_cross_second_platform = 20) : ∃ second_platform_length, second_platform_length = 250 := 
by
  sorry

end NUMINAMATH_GPT_length_of_second_platform_l1014_101459


namespace NUMINAMATH_GPT_rectangular_plot_width_l1014_101449

theorem rectangular_plot_width :
  ∀ (length width : ℕ), 
    length = 60 → 
    ∀ (poles spacing : ℕ), 
      poles = 44 → 
      spacing = 5 → 
      2 * length + 2 * width = poles * spacing →
      width = 50 :=
by
  intros length width h_length poles spacing h_poles h_spacing h_perimeter
  rw [h_length, h_poles, h_spacing] at h_perimeter
  linarith

end NUMINAMATH_GPT_rectangular_plot_width_l1014_101449


namespace NUMINAMATH_GPT_maximum_value_of_vectors_l1014_101447

open Real EuclideanGeometry

variables (a b c : EuclideanSpace ℝ (Fin 3))

def unit_vector (v : EuclideanSpace ℝ (Fin 3)) : Prop := ‖v‖ = 1

def given_conditions (a b c : EuclideanSpace ℝ (Fin 3)) : Prop :=
  unit_vector a ∧ unit_vector b ∧ ‖3 • a + 4 • b‖ = ‖4 • a - 3 • b‖ ∧ ‖c‖ = 2

theorem maximum_value_of_vectors
  (ha : unit_vector a)
  (hb : unit_vector b)
  (hab : ‖3 • a + 4 • b‖ = ‖4 • a - 3 • b‖)
  (hc : ‖c‖ = 2) :
  ‖a + b - c‖ ≤ sqrt 2 + 2 := 
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_vectors_l1014_101447


namespace NUMINAMATH_GPT_length_of_QR_of_triangle_l1014_101437

def length_of_QR (PQ PR PM : ℝ) : ℝ := sorry

theorem length_of_QR_of_triangle (PQ PR : ℝ) (PM : ℝ) (hPQ : PQ = 4) (hPR : PR = 7) (hPM : PM = 7 / 2) : length_of_QR PQ PR PM = 9 := by
  sorry

end NUMINAMATH_GPT_length_of_QR_of_triangle_l1014_101437


namespace NUMINAMATH_GPT_correct_solution_l1014_101486

variable (x y : ℤ) (a b : ℤ) (h1 : 2 * x + a * y = 6) (h2 : b * x - 7 * y = 16)

theorem correct_solution : 
  (∃ x y : ℤ, 2 * x - 3 * y = 6 ∧ 5 * x - 7 * y = 16 ∧ x = 6 ∧ y = 2) :=
by
  use 6, 2
  constructor
  · exact sorry -- 2 * 6 - 3 * 2 = 6
  constructor
  · exact sorry -- 5 * 6 - 7 * 2 = 16
  constructor
  · exact rfl
  · exact rfl

end NUMINAMATH_GPT_correct_solution_l1014_101486


namespace NUMINAMATH_GPT_length_of_solution_set_l1014_101406

variable {a b : ℝ}

theorem length_of_solution_set (h : ∀ x : ℝ, a ≤ 3 * x + 4 ∧ 3 * x + 4 ≤ b → 12 = (b - a) / 3) : b - a = 36 :=
sorry

end NUMINAMATH_GPT_length_of_solution_set_l1014_101406


namespace NUMINAMATH_GPT_possible_values_of_a2b_b2c_c2a_l1014_101408

theorem possible_values_of_a2b_b2c_c2a (a b c : ℝ) (h : a + b + c = 1) : ∀ x : ℝ, ∃ a b c : ℝ, a + b + c = 1 ∧ a^2 * b + b^2 * c + c^2 * a = x :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_a2b_b2c_c2a_l1014_101408


namespace NUMINAMATH_GPT_tangent_line_at_P0_is_parallel_l1014_101471

noncomputable def curve (x : ℝ) : ℝ := x^3 + x - 2

def tangent_slope (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_line_at_P0_is_parallel (x y : ℝ) (h_curve : y = curve x) (h_slope : tangent_slope x = 4) :
  (x, y) = (-1, -4) :=
sorry

end NUMINAMATH_GPT_tangent_line_at_P0_is_parallel_l1014_101471


namespace NUMINAMATH_GPT_tan_135_eq_neg_one_l1014_101443

theorem tan_135_eq_neg_one : Real.tan (135 * Real.pi / 180) = -1 := by
  sorry

end NUMINAMATH_GPT_tan_135_eq_neg_one_l1014_101443


namespace NUMINAMATH_GPT_largest_common_term_l1014_101402

theorem largest_common_term (b : ℕ) (h1 : b ≡ 1 [MOD 3]) (h2 : b ≡ 2 [MOD 10]) (h3 : b < 300) : b = 290 :=
sorry

end NUMINAMATH_GPT_largest_common_term_l1014_101402


namespace NUMINAMATH_GPT_min_even_integers_least_one_l1014_101452

theorem min_even_integers_least_one (x y a b m n o : ℤ) 
  (h1 : x + y = 29)
  (h2 : x + y + a + b = 47)
  (h3 : x + y + a + b + m + n + o = 66) :
  ∃ e : ℕ, (e = 1) := by
sorry

end NUMINAMATH_GPT_min_even_integers_least_one_l1014_101452


namespace NUMINAMATH_GPT_ice_cream_remaining_l1014_101469

def total_initial_scoops : ℕ := 3 * 10
def ethan_scoops : ℕ := 1 + 1
def lucas_danny_connor_scoops : ℕ := 2 * 3
def olivia_scoops : ℕ := 1 + 1
def shannon_scoops : ℕ := 2 * olivia_scoops
def total_consumed_scoops : ℕ := ethan_scoops + lucas_danny_connor_scoops + olivia_scoops + shannon_scoops
def remaining_scoops : ℕ := total_initial_scoops - total_consumed_scoops

theorem ice_cream_remaining : remaining_scoops = 16 := by
  sorry

end NUMINAMATH_GPT_ice_cream_remaining_l1014_101469


namespace NUMINAMATH_GPT_solve_nested_function_l1014_101440

def f (x : ℝ) : ℝ := x^2 + 12 * x + 30

theorem solve_nested_function :
  ∃ x : ℝ, f (f (f (f (f x)))) = 0 ↔ (x = -6 + 6^(1/32) ∨ x = -6 - 6^(1/32)) :=
by sorry

end NUMINAMATH_GPT_solve_nested_function_l1014_101440


namespace NUMINAMATH_GPT_base_form_exists_l1014_101474

-- Definitions for three-digit number and its reverse in base g
def N (a b c g : ℕ) : ℕ := a * g^2 + b * g + c
def N_reverse (a b c g : ℕ) : ℕ := c * g^2 + b * g + a

-- The problem statement in Lean
theorem base_form_exists (a b c g : ℕ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) (h₄ : 0 < g)
    (h₅ : N a b c g = 2 * N_reverse a b c g) : ∃ k : ℕ, g = 3 * k + 2 ∧ k > 0 :=
by
  sorry

end NUMINAMATH_GPT_base_form_exists_l1014_101474


namespace NUMINAMATH_GPT_tangent_line_eq_monotonic_intervals_l1014_101470

noncomputable def f (x : ℝ) (a : ℝ) := x - a * Real.log x
noncomputable def f' (x : ℝ) (a : ℝ) := 1 - (a / x)

theorem tangent_line_eq (x y : ℝ) (h : x = 1 ∧ a = 2) :
  y = f 1 2 → (x - 1) + (y - 1) - 2 * ((x - 1) + (y - 1)) = 0 := by sorry

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x > 0, f' x a > 0) ∧
  (a > 0 → ∀ x > 0, (x < a → f' x a < 0) ∧ (x > a → f' x a > 0)) := by sorry

end NUMINAMATH_GPT_tangent_line_eq_monotonic_intervals_l1014_101470


namespace NUMINAMATH_GPT_decimal_to_binary_25_l1014_101497

theorem decimal_to_binary_25 : (25 : Nat) = 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0 :=
by
  sorry

end NUMINAMATH_GPT_decimal_to_binary_25_l1014_101497


namespace NUMINAMATH_GPT_balloons_difference_l1014_101444

-- Define the balloons each person brought
def Allan_red := 150
def Allan_blue_total := 75
def Allan_forgotten_blue := 25
def Allan_green := 30

def Jake_red := 100
def Jake_blue := 50
def Jake_green := 45

-- Calculate the actual balloons Allan brought to the park
def Allan_blue := Allan_blue_total - Allan_forgotten_blue
def Allan_total := Allan_red + Allan_blue + Allan_green

-- Calculate the total number of balloons Jake brought
def Jake_total := Jake_red + Jake_blue + Jake_green

-- State the problem: Prove Allan distributed 35 more balloons than Jake
theorem balloons_difference : Allan_total - Jake_total = 35 := 
by
  sorry

end NUMINAMATH_GPT_balloons_difference_l1014_101444


namespace NUMINAMATH_GPT_students_per_class_l1014_101467

theorem students_per_class (total_cupcakes : ℕ) (num_classes : ℕ) (pe_students : ℕ) 
  (h1 : total_cupcakes = 140) (h2 : num_classes = 3) (h3 : pe_students = 50) : 
  (total_cupcakes - pe_students) / num_classes = 30 :=
by
  sorry

end NUMINAMATH_GPT_students_per_class_l1014_101467


namespace NUMINAMATH_GPT_cuboid_surface_area_500_l1014_101427

def surface_area (w l h : ℝ) : ℝ :=
  2 * l * w + 2 * l * h + 2 * w * h

theorem cuboid_surface_area_500 :
  ∀ (w l h : ℝ), w = 4 → l = w + 6 → h = l + 5 →
  surface_area w l h = 500 :=
by
  intros w l h hw hl hh
  unfold surface_area
  rw [hw, hl, hh]
  norm_num
  sorry

end NUMINAMATH_GPT_cuboid_surface_area_500_l1014_101427


namespace NUMINAMATH_GPT_find_y_positive_monotone_l1014_101480

noncomputable def y (y : ℝ) : Prop :=
  0 < y ∧ y * (⌊y⌋₊ : ℝ) = 132 ∧ y = 12

theorem find_y_positive_monotone : ∃ y : ℝ, 0 < y ∧ y * (⌊y⌋₊ : ℝ) = 132 := by
  sorry

end NUMINAMATH_GPT_find_y_positive_monotone_l1014_101480


namespace NUMINAMATH_GPT_minimum_perimeter_is_12_l1014_101445

noncomputable def minimum_perimeter_upper_base_frustum
  (a b : ℝ) (h : ℝ) (V : ℝ) : ℝ :=
if h = 3 ∧ V = 63 ∧ (a * b = 9) then
  2 * (a + b)
else
  0 -- this case will never be used

theorem minimum_perimeter_is_12 :
  ∃ a b : ℝ, a * b = 9 ∧ 2 * (a + b) = 12 :=
by
  existsi 3
  existsi 3
  sorry

end NUMINAMATH_GPT_minimum_perimeter_is_12_l1014_101445


namespace NUMINAMATH_GPT_spacesMovedBeforeSetback_l1014_101411

-- Let's define the conditions as local constants
def totalSpaces : ℕ := 48
def firstTurnMove : ℕ := 8
def thirdTurnMove : ℕ := 6
def remainingSpacesToWin : ℕ := 37
def setback : ℕ := 5

theorem spacesMovedBeforeSetback (x : ℕ) : 
  (firstTurnMove + thirdTurnMove) + x - setback + remainingSpacesToWin = totalSpaces →
  x = 28 := by
  sorry

end NUMINAMATH_GPT_spacesMovedBeforeSetback_l1014_101411


namespace NUMINAMATH_GPT_direct_variation_exponent_l1014_101498

variable {X Y Z : Type}

theorem direct_variation_exponent (k j : ℝ) (x y z : ℝ) 
  (h1 : x = k * y^4) 
  (h2 : y = j * z^3) : 
  ∃ m : ℝ, x = m * z^12 :=
by
  sorry

end NUMINAMATH_GPT_direct_variation_exponent_l1014_101498


namespace NUMINAMATH_GPT_triangle_sine_value_l1014_101448

-- Define the triangle sides and angles
variables {a b c A B C : ℝ}

-- Main theorem stating the proof problem
theorem triangle_sine_value (h : a^2 = b^2 + c^2 - bc) :
  (a * Real.sin B) / b = Real.sqrt 3 / 2 := sorry

end NUMINAMATH_GPT_triangle_sine_value_l1014_101448


namespace NUMINAMATH_GPT_cat_food_weight_l1014_101423

theorem cat_food_weight (x : ℝ) :
  let bags_of_cat_food := 2
  let bags_of_dog_food := 2
  let ounces_per_pound := 16
  let total_ounces_of_pet_food := 256
  let dog_food_extra_weight := 2
  (ounces_per_pound * (bags_of_cat_food * x + bags_of_dog_food * (x + dog_food_extra_weight))) = total_ounces_of_pet_food
  → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_cat_food_weight_l1014_101423


namespace NUMINAMATH_GPT_fraction_to_decimal_17_625_l1014_101430

def fraction_to_decimal (num : ℕ) (den : ℕ) : ℚ := num / den

theorem fraction_to_decimal_17_625 : fraction_to_decimal 17 625 = 272 / 10000 := by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_17_625_l1014_101430


namespace NUMINAMATH_GPT_triangle_area_DEF_l1014_101431

def point : Type := ℝ × ℝ

def D : point := (-2, 2)
def E : point := (8, 2)
def F : point := (6, -4)

theorem triangle_area_DEF :
  let base : ℝ := abs (D.1 - E.1)
  let height : ℝ := abs (F.2 - 2)
  let area := 1/2 * base * height
  area = 30 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_area_DEF_l1014_101431


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1014_101439

def A : Set ℝ := {x | 1 < x ∧ x < 7}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x ≤ 5} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1014_101439


namespace NUMINAMATH_GPT_range_of_x_in_function_l1014_101405

theorem range_of_x_in_function : ∀ (x : ℝ), (2 - x ≥ 0) ∧ (x + 2 ≠ 0) ↔ (x ≤ 2 ∧ x ≠ -2) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_range_of_x_in_function_l1014_101405


namespace NUMINAMATH_GPT_smallest_benches_l1014_101419

theorem smallest_benches (N : ℕ) (h1 : ∃ n, 8 * n = 40 ∧ 10 * n = 40) : N = 20 :=
sorry

end NUMINAMATH_GPT_smallest_benches_l1014_101419


namespace NUMINAMATH_GPT_total_gulbis_l1014_101429

theorem total_gulbis (dureums fish_per_dureum : ℕ) (h1 : dureums = 156) (h2 : fish_per_dureum = 20) : dureums * fish_per_dureum = 3120 :=
by
  sorry

end NUMINAMATH_GPT_total_gulbis_l1014_101429


namespace NUMINAMATH_GPT_interest_difference_l1014_101441

theorem interest_difference
  (principal : ℕ) (rate : ℚ) (time : ℕ) (interest : ℚ) (difference : ℚ)
  (h1 : principal = 600)
  (h2 : rate = 0.05)
  (h3 : time = 8)
  (h4 : interest = principal * (rate * time))
  (h5 : difference = principal - interest) :
  difference = 360 :=
by sorry

end NUMINAMATH_GPT_interest_difference_l1014_101441


namespace NUMINAMATH_GPT_average_speed_of_the_car_l1014_101410

noncomputable def averageSpeed (d1 d2 d3 d4 t1 t2 t3 t4 : ℝ) : ℝ :=
  let totalDistance := d1 + d2 + d3 + d4
  let totalTime := t1 + t2 + t3 + t4
  totalDistance / totalTime

theorem average_speed_of_the_car :
  averageSpeed 30 35 65 (40 * 0.5) (30 / 45) (35 / 55) 1 0.5 = 54 := 
  by 
    sorry

end NUMINAMATH_GPT_average_speed_of_the_car_l1014_101410


namespace NUMINAMATH_GPT_range_of_a_l1014_101414

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x ^ 2 - |x + 1| + 2 * a ≥ 0) ↔ a ∈ (Set.Ici ((Real.sqrt 3 + 1) / 4)) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1014_101414


namespace NUMINAMATH_GPT_batsman_average_after_17th_inning_l1014_101416

theorem batsman_average_after_17th_inning
  (A : ℕ)  -- average after the 16th inning
  (h1 : 16 * A + 300 = 17 * (A + 10)) :
  A + 10 = 140 :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_after_17th_inning_l1014_101416


namespace NUMINAMATH_GPT_option_C_correct_l1014_101438

theorem option_C_correct : ∀ x : ℝ, x^2 + 1 ≥ 2 * |x| :=
by
  intro x
  sorry

end NUMINAMATH_GPT_option_C_correct_l1014_101438


namespace NUMINAMATH_GPT_evaluate_expression_l1014_101499

theorem evaluate_expression :
  let x := (1 / 4 : ℚ)
  let y := (1 / 3 : ℚ)
  let z := (-2 : ℚ)
  let w := (3 : ℚ)
  (x^3 * y^2 * z^2 * w) = (1 / 48 : ℚ) :=
by
  let x := (1 / 4 : ℚ)
  let y := (1 / 3 : ℚ)
  let z := (-2 : ℚ)
  let w := (3 : ℚ)
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1014_101499


namespace NUMINAMATH_GPT_problem1_problem2_l1014_101491

noncomputable def cos_alpha (α : ℝ) : ℝ := (Real.sqrt 2 + 4) / 6
noncomputable def cos_alpha_plus_half_beta (α β : ℝ) : ℝ := 5 * Real.sqrt 3 / 9

theorem problem1 {α : ℝ} (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
                 (h1 : Real.cos (Real.pi / 4 + α) = 1 / 3) :
  Real.cos α = cos_alpha α :=
sorry

theorem problem2 {α β : ℝ} (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
                 (hβ1 : -Real.pi / 2 < β) (hβ2 : β < 0) 
                 (h1 : Real.cos (Real.pi / 4 + α) = 1 / 3) 
                 (h2 : Real.cos (Real.pi / 4 - β / 2) = Real.sqrt 3 / 3) :
  Real.cos (α + β / 2) = cos_alpha_plus_half_beta α β :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1014_101491


namespace NUMINAMATH_GPT_find_solution_to_inequality_l1014_101412

open Set

noncomputable def inequality_solution : Set ℝ := {x : ℝ | 0.5 ≤ x ∧ x < 2 ∨ 3 ≤ x}

theorem find_solution_to_inequality :
  {x : ℝ | (x^2 + 1) / (x - 2) + (2 * x + 3) / (2 * x - 1) ≥ 4} = inequality_solution := 
sorry

end NUMINAMATH_GPT_find_solution_to_inequality_l1014_101412


namespace NUMINAMATH_GPT_gina_total_cost_l1014_101489

-- Define the constants based on the conditions
def total_credits : ℕ := 18
def reg_credits : ℕ := 12
def reg_cost_per_credit : ℕ := 450
def lab_credits : ℕ := 6
def lab_cost_per_credit : ℕ := 550
def num_textbooks : ℕ := 3
def textbook_cost : ℕ := 150
def num_online_resources : ℕ := 4
def online_resource_cost : ℕ := 95
def facilities_fee : ℕ := 200
def lab_fee_per_credit : ℕ := 75

-- Calculating the total cost
noncomputable def total_cost : ℕ :=
  (reg_credits * reg_cost_per_credit) +
  (lab_credits * lab_cost_per_credit) +
  (num_textbooks * textbook_cost) +
  (num_online_resources * online_resource_cost) +
  facilities_fee +
  (lab_credits * lab_fee_per_credit)

-- The proof problem to show that the total cost is 10180
theorem gina_total_cost : total_cost = 10180 := by
  sorry

end NUMINAMATH_GPT_gina_total_cost_l1014_101489


namespace NUMINAMATH_GPT_number_of_permissible_sandwiches_l1014_101428

theorem number_of_permissible_sandwiches (b m c : ℕ) (h : b = 5) (me : m = 7) (ch : c = 6) 
  (no_ham_cheddar : ∀ bread, ¬(bread = ham ∧ cheese = cheddar))
  (no_turkey_swiss : ∀ bread, ¬(bread = turkey ∧ cheese = swiss)) : 
  5 * 7 * 6 - (5 * 1 * 1) - (5 * 1 * 1) = 200 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_permissible_sandwiches_l1014_101428


namespace NUMINAMATH_GPT_cupcakes_leftover_l1014_101494

-- Definitions based on the conditions
def total_cupcakes : ℕ := 17
def num_children : ℕ := 3

-- Theorem proving the correct answer
theorem cupcakes_leftover : total_cupcakes % num_children = 2 := by
  sorry

end NUMINAMATH_GPT_cupcakes_leftover_l1014_101494


namespace NUMINAMATH_GPT_math_problem_l1014_101478

variables (a b c : ℤ)

theorem math_problem (h1 : a - (b - 2 * c) = 19) (h2 : a - b - 2 * c = 7) : a - b = 13 := by
  sorry

end NUMINAMATH_GPT_math_problem_l1014_101478


namespace NUMINAMATH_GPT_find_value_l1014_101432

-- Given points A(a, 1), B(2, b), and C(3, 4).
variables (a b : ℝ)

-- Given condition from the problem
def condition : Prop := (3 * a + 4 = 6 + 4 * b)

-- The target is to find 3a - 4b
def target : ℝ := 3 * a - 4 * b

theorem find_value (h : condition a b) : target a b = 2 := 
by sorry

end NUMINAMATH_GPT_find_value_l1014_101432


namespace NUMINAMATH_GPT_number_of_adult_female_alligators_l1014_101473

-- Define the conditions
def total_alligators (females males: ℕ) : ℕ := females + males

def male_alligators : ℕ := 25
def female_alligators : ℕ := 25
def juvenile_percentage : ℕ := 40

-- Calculate the number of juveniles
def juvenile_count : ℕ := (juvenile_percentage * female_alligators) / 100

-- Calculate the number of adults
def adult_female_alligators : ℕ := female_alligators - juvenile_count

-- The main theorem statement
theorem number_of_adult_female_alligators : adult_female_alligators = 15 :=
by
    sorry

end NUMINAMATH_GPT_number_of_adult_female_alligators_l1014_101473


namespace NUMINAMATH_GPT_percent_of_g_is_a_l1014_101404

theorem percent_of_g_is_a (a b c d e f g : ℤ) (h1 : (a + b + c + d + e + f + g) / 7 = 9)
: (a / g) * 100 = 50 := 
sorry

end NUMINAMATH_GPT_percent_of_g_is_a_l1014_101404


namespace NUMINAMATH_GPT_simple_interest_years_l1014_101434

theorem simple_interest_years (r1 r2 t2 P1 P2 S : ℝ) (hP1: P1 = 3225) (hP2: P2 = 8000) (hr1: r1 = 0.08) (hr2: r2 = 0.15) (ht2: t2 = 2) (hCI : S = 2580) :
    S / 2 = (P1 * r1 * t) / 100 → t = 5 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_years_l1014_101434


namespace NUMINAMATH_GPT_square_area_is_256_l1014_101454

-- Definitions of the conditions
def rect_width : ℝ := 4
def rect_length : ℝ := 3 * rect_width
def side_of_square : ℝ := rect_length + rect_width

-- Proposition
theorem square_area_is_256 (rect_width : ℝ) (h1 : rect_width = 4) 
                           (rect_length : ℝ) (h2 : rect_length = 3 * rect_width) :
  side_of_square ^ 2 = 256 :=
by 
  sorry

end NUMINAMATH_GPT_square_area_is_256_l1014_101454


namespace NUMINAMATH_GPT_div_5_implies_one_div_5_l1014_101424

theorem div_5_implies_one_div_5 (a b : ℕ) (h : 5 ∣ (a * b)) : (5 ∣ a) ∨ (5 ∣ b) :=
by 
  sorry

end NUMINAMATH_GPT_div_5_implies_one_div_5_l1014_101424


namespace NUMINAMATH_GPT_xy_positive_l1014_101420

theorem xy_positive (x y : ℝ) (h1 : x - y < x) (h2 : x + y > y) : x > 0 ∧ y > 0 :=
sorry

end NUMINAMATH_GPT_xy_positive_l1014_101420


namespace NUMINAMATH_GPT_median_eq_altitude_eq_perp_bisector_eq_l1014_101495

open Real

def point := ℝ × ℝ

def A : point := (1, 3)
def B : point := (3, 1)
def C : point := (-1, 0)

-- Median on BC
theorem median_eq : ∀ (x y : ℝ), (x, y) = A ∨ (x, y) = ((1 + (-1))/2, (1 + 0)/2) → x = 1 :=
by
  intros x y h
  sorry

-- Altitude on BC
theorem altitude_eq : ∀ (x y : ℝ), (x, y) = A ∨ (x - 1) / (y - 3) = -4 → 4*x + y - 7 = 0 :=
by
  intros x y h
  sorry

-- Perpendicular bisector of BC
theorem perp_bisector_eq : ∀ (x y : ℝ), (x = 1 ∧ y = 1/2) ∨ (x - 1) / (y - 1/2) = -4 
                          → 8*x + 2*y - 9 = 0 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_median_eq_altitude_eq_perp_bisector_eq_l1014_101495


namespace NUMINAMATH_GPT_find_X_l1014_101496

theorem find_X : 
  let M := 3012 / 4
  let N := M / 4
  let X := M - N
  X = 564.75 :=
by
  sorry

end NUMINAMATH_GPT_find_X_l1014_101496


namespace NUMINAMATH_GPT_sectorChordLength_correct_l1014_101464

open Real

noncomputable def sectorChordLength (r α : ℝ) : ℝ :=
  2 * r * sin (α / 2)

theorem sectorChordLength_correct :
  ∃ (r α : ℝ), (1/2) * α * r^2 = 1 ∧ 2 * r + α * r = 4 ∧ sectorChordLength r α = 2 * sin 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_sectorChordLength_correct_l1014_101464


namespace NUMINAMATH_GPT_polynomial_value_l1014_101451

theorem polynomial_value
  (x : ℝ)
  (h : x^2 + 2 * x - 2 = 0) :
  4 - 2 * x - x^2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_value_l1014_101451


namespace NUMINAMATH_GPT_round_robin_cycles_l1014_101417

-- Define the conditions
def teams : ℕ := 28
def wins_per_team : ℕ := 13
def losses_per_team : ℕ := 13
def total_teams_games := teams * (teams - 1) / 2
def sets_of_three_teams := (teams * (teams - 1) * (teams - 2)) / 6

-- Define the problem statement
theorem round_robin_cycles :
  -- We need to show that the number of sets of three teams {A, B, C} where A beats B, B beats C, and C beats A is 1092
  (sets_of_three_teams - (teams * (wins_per_team * (wins_per_team - 1)) / 2)) = 1092 :=
by
  sorry

end NUMINAMATH_GPT_round_robin_cycles_l1014_101417


namespace NUMINAMATH_GPT_rotated_angle_l1014_101400

theorem rotated_angle (initial_angle : ℝ) (rotation_angle : ℝ) (final_angle : ℝ) :
  initial_angle = 30 ∧ rotation_angle = 450 → final_angle = 60 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_rotated_angle_l1014_101400


namespace NUMINAMATH_GPT_multiply_abs_value_l1014_101455

theorem multiply_abs_value : -2 * |(-3 : ℤ)| = -6 := by
  sorry

end NUMINAMATH_GPT_multiply_abs_value_l1014_101455


namespace NUMINAMATH_GPT_mike_total_money_l1014_101483

theorem mike_total_money (num_bills : ℕ) (value_per_bill : ℕ) (h1 : num_bills = 9) (h2 : value_per_bill = 5) :
  (num_bills * value_per_bill) = 45 :=
by
  sorry

end NUMINAMATH_GPT_mike_total_money_l1014_101483


namespace NUMINAMATH_GPT_range_of_a_for_inequality_l1014_101472

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1) ↔ a ≥ 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_for_inequality_l1014_101472


namespace NUMINAMATH_GPT_sales_tax_amount_l1014_101456

variable (T : ℝ := 25) -- Total amount spent
variable (y : ℝ := 19.7) -- Cost of tax-free items
variable (r : ℝ := 0.06) -- Tax rate

theorem sales_tax_amount : 
  ∃ t : ℝ, t = 0.3 ∧ (T - y) * r = t :=
by 
  sorry

end NUMINAMATH_GPT_sales_tax_amount_l1014_101456


namespace NUMINAMATH_GPT_correct_polynomial_multiplication_l1014_101457

theorem correct_polynomial_multiplication (a b : ℤ) (x : ℝ)
  (h1 : 2 * b - 3 * a = 11)
  (h2 : 2 * b + a = -9) :
  (2 * x + a) * (3 * x + b) = 6 * x^2 - 19 * x + 10 := by
  sorry

end NUMINAMATH_GPT_correct_polynomial_multiplication_l1014_101457


namespace NUMINAMATH_GPT_solve_prime_equation_l1014_101426

theorem solve_prime_equation (x y : ℕ) (p : ℕ) (hp : Nat.Prime p) :
  x^3 + y^3 - 3 * x * y = p - 1 ↔
  (x = 1 ∧ y = 0 ∧ p = 2) ∨
  (x = 0 ∧ y = 1 ∧ p = 2) ∨
  (x = 2 ∧ y = 2 ∧ p = 5) := 
sorry

end NUMINAMATH_GPT_solve_prime_equation_l1014_101426


namespace NUMINAMATH_GPT_right_triangle_unique_value_l1014_101421

theorem right_triangle_unique_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
(h1 : a + b + c = (1/2) * a * b) (h2 : c^2 = a^2 + b^2) : a + b - c = 4 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_unique_value_l1014_101421


namespace NUMINAMATH_GPT_starting_number_l1014_101425

theorem starting_number (n : ℤ) : 
  (∃ n, (200 - n) / 3 = 33 ∧ (200 % 3 ≠ 0) ∧ (n % 3 = 0 ∧ n ≤ 200)) → n = 102 :=
by
  sorry

end NUMINAMATH_GPT_starting_number_l1014_101425


namespace NUMINAMATH_GPT_determine_number_l1014_101492

def is_divisible_by_9 (n : ℕ) : Prop :=
  (n.digits 10).sum % 9 = 0

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 5

def ten_power (n p : ℕ) : ℕ :=
  n * 10 ^ p

theorem determine_number (a b : ℕ) (h₁ : b = 0 ∨ b = 5)
  (h₂ : is_divisible_by_9 (7 + 2 + a + 3 + b))
  (h₃ : is_divisible_by_5 (7 * 10000 + 2 * 1000 + a * 100 + 3 * 10 + b)) :
  (7 * 10000 + 2 * 1000 + a * 100 + 3 * 10 + b = 72630 ∨ 
   7 * 10000 + 2 * 1000 + a * 100 + 3 * 10 + b = 72135) :=
by sorry

end NUMINAMATH_GPT_determine_number_l1014_101492


namespace NUMINAMATH_GPT_find_coordinates_of_P_l1014_101453

-- Define points N and M with given symmetries.
structure Point where
  x : ℝ
  y : ℝ

def symmetric_about_x (P1 P2 : Point) : Prop :=
  P1.x = P2.x ∧ P1.y = -P2.y

def symmetric_about_y (P1 P2 : Point) : Prop :=
  P1.x = -P2.x ∧ P1.y = P2.y

-- Given conditions
def N : Point := ⟨1, 2⟩
def M : Point := ⟨-1, 2⟩ -- derived from symmetry about y-axis with N
def P : Point := ⟨-1, -2⟩ -- derived from symmetry about x-axis with M

theorem find_coordinates_of_P :
  symmetric_about_x M P ∧ symmetric_about_y N M → P = ⟨-1, -2⟩ :=
by
  sorry

end NUMINAMATH_GPT_find_coordinates_of_P_l1014_101453


namespace NUMINAMATH_GPT_cumulative_revenue_eq_l1014_101481

-- Define the initial box office revenue and growth rate
def initial_revenue : ℝ := 3
def growth_rate (x : ℝ) : ℝ := x

-- Define the cumulative revenue equation after 3 days
def cumulative_revenue (x : ℝ) : ℝ :=
  initial_revenue + initial_revenue * (1 + growth_rate x) + initial_revenue * (1 + growth_rate x) ^ 2

-- State the theorem that proves the equation
theorem cumulative_revenue_eq (x : ℝ) :
  cumulative_revenue x = 10 :=
sorry

end NUMINAMATH_GPT_cumulative_revenue_eq_l1014_101481


namespace NUMINAMATH_GPT_expenditure_on_digging_l1014_101484

noncomputable def volume_of_cylinder (r h : ℝ) := 
  Real.pi * r^2 * h

noncomputable def rate_per_cubic_meter (cost : ℝ) (r h : ℝ) : ℝ := 
  cost / (volume_of_cylinder r h)

theorem expenditure_on_digging (d h : ℝ) (cost : ℝ) (r : ℝ) (π : ℝ) (rate : ℝ)
  (h₀ : d = 3) (h₁ : h = 14) (h₂ : cost = 1682.32) (h₃ : r = d / 2) (h₄ : π = Real.pi) 
  : rate_per_cubic_meter cost r h = 17 := sorry

end NUMINAMATH_GPT_expenditure_on_digging_l1014_101484


namespace NUMINAMATH_GPT_difference_of_cubes_divisible_by_8_l1014_101463

theorem difference_of_cubes_divisible_by_8 (a b : ℤ) : 
  8 ∣ ((2 * a - 1) ^ 3 - (2 * b - 1) ^ 3) := 
by
  sorry

end NUMINAMATH_GPT_difference_of_cubes_divisible_by_8_l1014_101463


namespace NUMINAMATH_GPT_find_third_side_of_triangle_l1014_101418

theorem find_third_side_of_triangle (a b : ℝ) (A : ℝ) (h1 : a = 6) (h2 : b = 10) (h3 : A = 18) (h4 : ∃ C, 0 < C ∧ C < π / 2 ∧ A = 0.5 * a * b * Real.sin C) : 
  ∃ c : ℝ, c = 2 * Real.sqrt 22 :=
by
  sorry

end NUMINAMATH_GPT_find_third_side_of_triangle_l1014_101418


namespace NUMINAMATH_GPT_find_n_l1014_101446

-- Define the conditions as hypothesis
variables (A B n : ℕ)

-- Hypothesis 1: This year, Ana's age is the square of Bonita's age.
-- A = B^2
#check (A = B^2) 

-- Hypothesis 2: Last year Ana was 5 times as old as Bonita.
-- A - 1 = 5 * (B - 1)
#check (A - 1 = 5 * (B - 1))

-- Hypothesis 3: Ana and Bonita were born n years apart.
-- A = B + n
#check (A = B + n)

-- Goal: The difference in their ages, n, should be 12.
theorem find_n (A B n : ℕ) (h1 : A = B^2) (h2 : A - 1 = 5 * (B - 1)) (h3 : A = B + n) : n = 12 :=
sorry

end NUMINAMATH_GPT_find_n_l1014_101446


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1014_101458

theorem solution_set_of_inequality (x : ℝ) : 
  abs ((x + 2) / x) < 1 ↔ x < -1 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1014_101458


namespace NUMINAMATH_GPT_marseille_hairs_l1014_101409

theorem marseille_hairs (N : ℕ) (M : ℕ) (hN : N = 2000000) (hM : M = 300001) :
  ∃ k, k ≥ 7 ∧ ∃ b : ℕ, b ≤ M ∧ b > 0 ∧ ∀ i ≤ M, ∃ l : ℕ, l ≥ k → l ≤ (N / M + 1) :=
by
  sorry

end NUMINAMATH_GPT_marseille_hairs_l1014_101409


namespace NUMINAMATH_GPT_find_m_of_parallel_lines_l1014_101482

theorem find_m_of_parallel_lines
  (m : ℝ) 
  (parallel : ∀ x y, (x - 2 * y + 5 = 0 → 2 * x + m * y - 5 = 0)) :
  m = -4 :=
sorry

end NUMINAMATH_GPT_find_m_of_parallel_lines_l1014_101482


namespace NUMINAMATH_GPT_plants_per_row_l1014_101422

theorem plants_per_row (P : ℕ) (rows : ℕ) (yield_per_plant : ℕ) (total_yield : ℕ) 
  (h1 : rows = 30)
  (h2 : yield_per_plant = 20)
  (h3 : total_yield = 6000)
  (h4 : rows * yield_per_plant * P = total_yield) : 
  P = 10 :=
by 
  sorry

end NUMINAMATH_GPT_plants_per_row_l1014_101422


namespace NUMINAMATH_GPT_dante_age_l1014_101466

def combined_age (D : ℕ) : ℕ := D + D / 2 + (D + 1)

theorem dante_age :
  ∃ D : ℕ, combined_age D = 31 ∧ D = 12 :=
by
  sorry

end NUMINAMATH_GPT_dante_age_l1014_101466


namespace NUMINAMATH_GPT_partial_fractions_sum_zero_l1014_101479

theorem partial_fractions_sum_zero (A B C D E : ℚ) :
  (∀ x : ℚ, 
     x ≠ 0 → x ≠ -1 → x ≠ -2 → x ≠ -3 → x ≠ -4 →
     1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4)) = 
     A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4)) →
  A + B + C + D + E = 0 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_partial_fractions_sum_zero_l1014_101479


namespace NUMINAMATH_GPT_unique_prime_p_l1014_101401

theorem unique_prime_p (p : ℕ) (hp : Nat.Prime p) (h : Nat.Prime (p^2 + 2)) : p = 3 := 
by 
  sorry

end NUMINAMATH_GPT_unique_prime_p_l1014_101401


namespace NUMINAMATH_GPT_difference_of_squares_l1014_101475

theorem difference_of_squares (a b : ℕ) (h1: a = 630) (h2: b = 570) : a^2 - b^2 = 72000 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l1014_101475


namespace NUMINAMATH_GPT_factorize_x_squared_minus_1_l1014_101407

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_factorize_x_squared_minus_1_l1014_101407


namespace NUMINAMATH_GPT_original_price_l1014_101468

theorem original_price (total_payment : ℝ) (num_units : ℕ) (discount_rate : ℝ) 
(h1 : total_payment = 500) (h2 : num_units = 18) (h3 : discount_rate = 0.20) : 
  (total_payment / (1 - discount_rate) * num_units) = 625.05 :=
by
  sorry

end NUMINAMATH_GPT_original_price_l1014_101468


namespace NUMINAMATH_GPT_number_of_digits_in_x_l1014_101485

open Real

theorem number_of_digits_in_x
  (x y : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (hxy_inequality : x > y)
  (hxy_prod : x * y = 490)
  (hlog_cond : (log x - log 7) * (log y - log 7) = -143/4) :
  ∃ n : ℕ, n = 8 ∧ (10^(n - 1) ≤ x ∧ x < 10^n) :=
by
  sorry

end NUMINAMATH_GPT_number_of_digits_in_x_l1014_101485


namespace NUMINAMATH_GPT_distance_from_A_to_B_l1014_101435

theorem distance_from_A_to_B (d C1A C1B C2A C2B : ℝ) (h1 : C1A + C1B = d)
  (h2 : C2A + C2B = d) (h3 : (C1A = 2 * C1B) ∨ (C1B = 2 * C1A)) 
  (h4 : (C2A = 3 * C2B) ∨ (C2B = 3 * C2A))
  (h5 : |C2A - C1A| = 10) : d = 120 ∨ d = 24 :=
sorry

end NUMINAMATH_GPT_distance_from_A_to_B_l1014_101435


namespace NUMINAMATH_GPT_A_leaves_after_2_days_l1014_101450

noncomputable def A_work_rate : ℚ := 1 / 20
noncomputable def B_work_rate : ℚ := 1 / 30
noncomputable def C_work_rate : ℚ := 1 / 10
noncomputable def C_days_work : ℚ := 4
noncomputable def total_days_work : ℚ := 15

theorem A_leaves_after_2_days (x : ℚ) : 
  2 / 5 + x / 12 + (15 - x) / 30 = 1 → x = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_A_leaves_after_2_days_l1014_101450


namespace NUMINAMATH_GPT_find_m_value_l1014_101415

noncomputable def pyramid_property (m : ℕ) : Prop :=
  let n1 := 3
  let n2 := 9
  let n3 := 6
  let r2_1 := m + n1
  let r2_2 := n1 + n2
  let r2_3 := n2 + n3
  let r3_1 := r2_1 + r2_2
  let r3_2 := r2_2 + r2_3
  let top := r3_1 + r3_2
  top = 54

theorem find_m_value : ∃ m : ℕ, pyramid_property m ∧ m = 12 := by
  sorry

end NUMINAMATH_GPT_find_m_value_l1014_101415


namespace NUMINAMATH_GPT_sum_m_n_l1014_101487

-- We define the conditions and problem
variables (m n : ℕ)

-- Conditions
def conditions := m > 50 ∧ n > 50 ∧ Nat.lcm m n = 480 ∧ Nat.gcd m n = 12

-- Statement to prove
theorem sum_m_n : conditions m n → m + n = 156 := by sorry

end NUMINAMATH_GPT_sum_m_n_l1014_101487
