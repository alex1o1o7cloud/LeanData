import Mathlib

namespace NUMINAMATH_GPT_monday_dressing_time_l2366_236683

theorem monday_dressing_time 
  (Tuesday_time Wednesday_time Thursday_time Friday_time Old_average_time : ℕ)
  (H_tuesday : Tuesday_time = 4)
  (H_wednesday : Wednesday_time = 3)
  (H_thursday : Thursday_time = 4)
  (H_friday : Friday_time = 2)
  (H_average : Old_average_time = 3) :
  ∃ Monday_time : ℕ, Monday_time = 2 :=
by
  let Total_time_5_days := Old_average_time * 5
  let Total_time := 4 + 3 + 4 + 2
  let Monday_time := Total_time_5_days - Total_time
  exact ⟨Monday_time, sorry⟩

end NUMINAMATH_GPT_monday_dressing_time_l2366_236683


namespace NUMINAMATH_GPT_problem_statement_l2366_236627

theorem problem_statement (k x₁ x₂ : ℝ) (hx₁x₂ : x₁ < x₂)
  (h_eq : ∀ x : ℝ, x^2 - (k - 3) * x + (k + 4) = 0) 
  (P : ℝ) (hP : P ≠ 0) 
  (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  (hacute : ∀ A B : ℝ, A = x₁ ∧ B = x₂ ∧ A < 0 ∧ B > 0) :
  k < -4 ∧ α ≠ β ∧ α < β := 
sorry

end NUMINAMATH_GPT_problem_statement_l2366_236627


namespace NUMINAMATH_GPT_adam_age_l2366_236615

variable (E A : ℕ)

namespace AgeProof

theorem adam_age (h1 : A = E - 5) (h2 : E + 1 = 3 * (A - 4)) : A = 9 :=
by
  sorry
end AgeProof

end NUMINAMATH_GPT_adam_age_l2366_236615


namespace NUMINAMATH_GPT_complete_square_form_l2366_236621

theorem complete_square_form {a h k : ℝ} :
  ∀ x, (x^2 - 5 * x) = a * (x - h)^2 + k → k = -25 / 4 :=
by
  intro x
  intro h_eq
  sorry

end NUMINAMATH_GPT_complete_square_form_l2366_236621


namespace NUMINAMATH_GPT_wheel_distance_l2366_236675

noncomputable def diameter : ℝ := 9
noncomputable def revolutions : ℝ := 18.683651804670912
noncomputable def pi_approx : ℝ := 3.14159
noncomputable def circumference (d : ℝ) : ℝ := pi_approx * d
noncomputable def distance (r : ℝ) (c : ℝ) : ℝ := r * c

theorem wheel_distance : distance revolutions (circumference diameter) = 528.219 :=
by
  unfold distance circumference diameter revolutions pi_approx
  -- Here we would perform the calculation and show that the result is approximately 528.219
  sorry

end NUMINAMATH_GPT_wheel_distance_l2366_236675


namespace NUMINAMATH_GPT_tennis_ball_ratio_problem_solution_l2366_236614

def tennis_ball_ratio_problem (total_balls ordered_white ordered_yellow dispatched_yellow extra_yellow : ℕ) : Prop :=
  total_balls = 114 ∧ 
  ordered_white = total_balls / 2 ∧ 
  ordered_yellow = total_balls / 2 ∧ 
  dispatched_yellow = ordered_yellow + extra_yellow → 
  (ordered_white / dispatched_yellow = 57 / 107)

theorem tennis_ball_ratio_problem_solution :
  tennis_ball_ratio_problem 114 57 57 107 50 := by 
  sorry

end NUMINAMATH_GPT_tennis_ball_ratio_problem_solution_l2366_236614


namespace NUMINAMATH_GPT_dog_food_bags_count_l2366_236681

-- Define the constants based on the problem statement
def CatFoodBags := 327
def DogFoodMore := 273

-- Define the total number of dog food bags based on the given conditions
def DogFoodBags : ℤ := CatFoodBags + DogFoodMore

-- State the theorem we want to prove
theorem dog_food_bags_count : DogFoodBags = 600 := by
  sorry

end NUMINAMATH_GPT_dog_food_bags_count_l2366_236681


namespace NUMINAMATH_GPT_binom_10_2_eq_45_l2366_236653

-- Definitions used in the conditions
def binom (n k : ℕ) := n.choose k

-- The statement that needs to be proven
theorem binom_10_2_eq_45 : binom 10 2 = 45 :=
by
  sorry

end NUMINAMATH_GPT_binom_10_2_eq_45_l2366_236653


namespace NUMINAMATH_GPT_math_problem_l2366_236669

theorem math_problem (n : ℕ) (h : n > 0) : 
  1957 ∣ (1721^(2*n) - 73^(2*n) - 521^(2*n) + 212^(2*n)) :=
sorry

end NUMINAMATH_GPT_math_problem_l2366_236669


namespace NUMINAMATH_GPT_sequence_pattern_l2366_236620

theorem sequence_pattern (a b c : ℝ) (h1 : a = 19.8) (h2 : b = 18.6) (h3 : c = 17.4) 
  (h4 : ∀ n, n = a ∨ n = b ∨ n = c ∨ n = 16.2 ∨ n = 15) 
  (H : ∀ x y, (y = x - 1.2) → 
    (x = a ∨ x = b ∨ x = c ∨ y = 16.2 ∨ y = 15)) :
  (16.2 = c - 1.2) ∧ (15 = (c - 1.2) - 1.2) :=
by
  sorry

end NUMINAMATH_GPT_sequence_pattern_l2366_236620


namespace NUMINAMATH_GPT_time_per_mask_after_first_hour_l2366_236686

-- Define the conditions as given in the problem
def rate_in_first_hour := 1 / 4 -- Manolo makes one face-mask every four minutes
def total_face_masks := 45 -- Manolo makes 45 face-masks in four hours
def first_hour_duration := 60 -- The duration of the first hour in minutes
def total_duration := 4 * 60 -- The total duration in minutes (4 hours)

-- Define the number of face-masks made in the first hour
def face_masks_first_hour := first_hour_duration / 4 -- 60 minutes / 4 minutes per face-mask = 15 face-masks

-- Calculate the number of face-masks made in the remaining time
def face_masks_remaining_hours := total_face_masks - face_masks_first_hour -- 45 - 15 = 30 face-masks

-- Define the duration of the remaining hours
def remaining_duration := total_duration - first_hour_duration -- 180 minutes (3 hours)

-- The target is to prove that the rate after the first hour is 6 minutes per face-mask
theorem time_per_mask_after_first_hour : remaining_duration / face_masks_remaining_hours = 6 := by
  sorry

end NUMINAMATH_GPT_time_per_mask_after_first_hour_l2366_236686


namespace NUMINAMATH_GPT_Felix_distance_proof_l2366_236628

def average_speed : ℕ := 66
def twice_speed : ℕ := 2 * average_speed
def driving_hours : ℕ := 4
def distance_covered : ℕ := twice_speed * driving_hours

theorem Felix_distance_proof : distance_covered = 528 := by
  sorry

end NUMINAMATH_GPT_Felix_distance_proof_l2366_236628


namespace NUMINAMATH_GPT_lcm_16_35_l2366_236684

theorem lcm_16_35 : Nat.lcm 16 35 = 560 := by
  sorry

end NUMINAMATH_GPT_lcm_16_35_l2366_236684


namespace NUMINAMATH_GPT_triangle_inequality_squared_l2366_236661

theorem triangle_inequality_squared {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (habc : a + b > c) (hbca : b + c > a) (hcab : c + a > b) :
    a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a) := sorry

end NUMINAMATH_GPT_triangle_inequality_squared_l2366_236661


namespace NUMINAMATH_GPT_cube_face_area_l2366_236600

-- Definition for the condition of the cube's surface area
def cube_surface_area (s : ℝ) : Prop := s = 36

-- Definition stating a cube has 6 faces
def cube_faces : ℝ := 6

-- The target proposition to prove
theorem cube_face_area (s : ℝ) (area_of_one_face : ℝ) (h1 : cube_surface_area s) (h2 : cube_faces = 6) : area_of_one_face = s / 6 :=
by
  sorry

end NUMINAMATH_GPT_cube_face_area_l2366_236600


namespace NUMINAMATH_GPT_bacterium_descendants_in_range_l2366_236698

theorem bacterium_descendants_in_range (total_bacteria : ℕ) (initial : ℕ) 
  (h_total : total_bacteria = 1000) (h_initial : initial = total_bacteria) 
  (descendants : ℕ → ℕ)
  (h_step : ∀ k, descendants (k+1) ≤ descendants k / 2) :
  ∃ k, 334 ≤ descendants k ∧ descendants k ≤ 667 :=
by
  sorry

end NUMINAMATH_GPT_bacterium_descendants_in_range_l2366_236698


namespace NUMINAMATH_GPT_max_area_trapezoid_l2366_236663

theorem max_area_trapezoid :
  ∀ {AB CD : ℝ}, 
    AB = 6 → CD = 14 → 
    (∃ (r1 r2 : ℝ), r1 = AB / 2 ∧ r2 = CD / 2 ∧ r1 + r2 = 10) → 
    (1 / 2 * (AB + CD) * 10 = 100) :=
by
  intros AB CD hAB hCD hExist
  sorry

end NUMINAMATH_GPT_max_area_trapezoid_l2366_236663


namespace NUMINAMATH_GPT_final_surface_area_l2366_236603

noncomputable def surface_area (total_cubes remaining_cubes cube_surface removed_internal_surface : ℕ) : ℕ :=
  (remaining_cubes * cube_surface) + (remaining_cubes * removed_internal_surface)

theorem final_surface_area :
  surface_area 64 55 54 6 = 3300 :=
by
  sorry

end NUMINAMATH_GPT_final_surface_area_l2366_236603


namespace NUMINAMATH_GPT_min_value_of_quadratic_l2366_236633

theorem min_value_of_quadratic (x y s : ℝ) (h : x + y = s) : 
  ∃ x y, 3 * x^2 + 2 * y^2 = 6 * s^2 / 5 := sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l2366_236633


namespace NUMINAMATH_GPT_cement_total_l2366_236660

-- Defining variables for the weights of cement
def weight_self : ℕ := 215
def weight_son : ℕ := 137

-- Defining the function that calculates the total weight of the cement
def total_weight (a b : ℕ) : ℕ := a + b

-- Theorem statement: Proving the total cement weight is 352 lbs
theorem cement_total : total_weight weight_self weight_son = 352 :=
by
  sorry

end NUMINAMATH_GPT_cement_total_l2366_236660


namespace NUMINAMATH_GPT_expansion_of_a_plus_b_pow_4_expansion_of_a_plus_b_pow_5_computation_of_formula_l2366_236695

section
variables (a b : ℚ)

theorem expansion_of_a_plus_b_pow_4 :
  (a + b) ^ 4 = a ^ 4 + 4 * a ^ 3 * b + 6 * a ^ 2 * b ^ 2 + 4 * a * b ^ 3 + b ^ 4 :=
sorry

theorem expansion_of_a_plus_b_pow_5 :
  (a + b) ^ 5 = a ^ 5 + 5 * a ^ 4 * b + 10 * a ^ 3 * b ^ 2 + 10 * a ^ 2 * b ^ 3 + 5 * a * b ^ 4 + b ^ 5 :=
sorry

theorem computation_of_formula :
  2^4 + 4*2^3*(-1/3) + 6*2^2*(-1/3)^2 + 4*2*(-1/3)^3 + (-1/3)^4 = 625 / 81 :=
sorry
end

end NUMINAMATH_GPT_expansion_of_a_plus_b_pow_4_expansion_of_a_plus_b_pow_5_computation_of_formula_l2366_236695


namespace NUMINAMATH_GPT_problem_l2366_236606

noncomputable def f (x : ℝ) : ℝ := Real.sin x + x - Real.pi / 4
noncomputable def g (x : ℝ) : ℝ := Real.cos x - x + Real.pi / 4

theorem problem (x1 x2 : ℝ) (hx1 : 0 < x1 ∧ x1 < Real.pi / 2) (hx2 : 0 < x2 ∧ x2 < Real.pi / 2) :
  (∃! x, 0 < x ∧ x < Real.pi / 2 ∧ f x = 0) ∧ (∃! x, 0 < x ∧ x < Real.pi / 2 ∧ g x = 0) →
  x1 + x2 = Real.pi / 2 :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_problem_l2366_236606


namespace NUMINAMATH_GPT_reporters_local_politics_percentage_l2366_236634

theorem reporters_local_politics_percentage
  (T : ℕ) -- Total number of reporters
  (P : ℝ) -- Percentage of reporters covering politics
  (h1 : 30 / 100 * (P / 100) * T = (P / 100 - 0.7 * (P / 100)) * T)
  (h2 : 92.85714285714286 / 100 * T = (1 - P / 100) * T):
  (0.7 * (P / 100) * T) / T = 5 / 100 :=
by
  sorry

end NUMINAMATH_GPT_reporters_local_politics_percentage_l2366_236634


namespace NUMINAMATH_GPT_find_a_value_l2366_236623

theorem find_a_value :
  (∀ (x y : ℝ), (x = 1.5 → y = 8 → x * y = 12) ∧ 
               (x = 2 → y = 6 → x * y = 12) ∧ 
               (x = 3 → y = 4 → x * y = 12)) →
  ∃ (a : ℝ), (5 * a = 12 ∧ a = 2.4) :=
by
  sorry

end NUMINAMATH_GPT_find_a_value_l2366_236623


namespace NUMINAMATH_GPT_percentage_of_males_l2366_236611

theorem percentage_of_males (total_employees males_below_50 males_percentage : ℕ) (h1 : total_employees = 800) (h2 : males_below_50 = 120) (h3 : 40 * males_percentage / 100 = 60 * males_below_50):
  males_percentage = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_males_l2366_236611


namespace NUMINAMATH_GPT_eggs_in_each_basket_l2366_236689

theorem eggs_in_each_basket :
  ∃ (n : ℕ), (n ∣ 30) ∧ (n ∣ 45) ∧ (n ≥ 5) ∧
    (∀ m : ℕ, (m ∣ 30) ∧ (m ∣ 45) ∧ (m ≥ 5) → m ≤ n) ∧ n = 15 :=
by
  -- Condition 1: n divides 30
  -- Condition 2: n divides 45
  -- Condition 3: n is greater than or equal to 5
  -- Condition 4: n is the largest such divisor
  -- Therefore, n = 15
  sorry

end NUMINAMATH_GPT_eggs_in_each_basket_l2366_236689


namespace NUMINAMATH_GPT_range_of_a_for_inequality_l2366_236618

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, a * x^2 + a * x + 1 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 4) :=
by sorry

end NUMINAMATH_GPT_range_of_a_for_inequality_l2366_236618


namespace NUMINAMATH_GPT_total_edge_length_of_parallelepiped_l2366_236645

/-- Kolya has 440 identical cubes with a side length of 1 cm.
Kolya constructs a rectangular parallelepiped from these cubes 
and all edges have lengths of at least 5 cm. Prove 
that the total length of all edges of the rectangular parallelepiped is 96 cm. -/
theorem total_edge_length_of_parallelepiped {a b c : ℕ} 
  (h1 : a * b * c = 440) 
  (h2 : a ≥ 5) 
  (h3 : b ≥ 5) 
  (h4 : c ≥ 5) : 
  4 * (a + b + c) = 96 :=
sorry

end NUMINAMATH_GPT_total_edge_length_of_parallelepiped_l2366_236645


namespace NUMINAMATH_GPT_combined_total_capacity_l2366_236655

theorem combined_total_capacity (A B C : ℝ) 
  (hA : 0.35 * A + 48 = 3 / 4 * A)
  (hB : 0.45 * B + 36 = 0.95 * B)
  (hC : 0.20 * C - 24 = 0.10 * C) :
  A + B + C = 432 := 
by 
  sorry

end NUMINAMATH_GPT_combined_total_capacity_l2366_236655


namespace NUMINAMATH_GPT_equation_solutions_l2366_236641

theorem equation_solutions (m n x y : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  x^n + y^n = 3^m ↔ (x = 1 ∧ y = 2 ∧ n = 3 ∧ m = 2) ∨ (x = 2 ∧ y = 1 ∧ n = 3 ∧ m = 2) :=
by
  sorry -- proof to be implemented

end NUMINAMATH_GPT_equation_solutions_l2366_236641


namespace NUMINAMATH_GPT_race_total_people_l2366_236605

theorem race_total_people (b t : ℕ) 
(h1 : b = t + 15) 
(h2 : 3 * t = 2 * b + 15) : 
b + t = 105 := 
sorry

end NUMINAMATH_GPT_race_total_people_l2366_236605


namespace NUMINAMATH_GPT_range_satisfying_f_inequality_l2366_236696

noncomputable def f (x : ℝ) : ℝ :=
  Real.log (1 + |x|) - (1 / (1 + x^2))

theorem range_satisfying_f_inequality : 
  ∀ x : ℝ, (1 / 3) < x ∧ x < 1 → f x > f (2 * x - 1) :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_range_satisfying_f_inequality_l2366_236696


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2366_236691

noncomputable def A : Set ℝ := { x | -1 < x - 3 ∧ x - 3 ≤ 2 }
noncomputable def B : Set ℝ := { x | 3 ≤ x ∧ x < 6 }

theorem intersection_of_A_and_B : A ∩ B = { x | 3 ≤ x ∧ x ≤ 5 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2366_236691


namespace NUMINAMATH_GPT_candy_problem_l2366_236692

theorem candy_problem (N S a : ℕ) (h1 : a = S - a - 7) (h2 : a > 1) : S = 21 := 
sorry

end NUMINAMATH_GPT_candy_problem_l2366_236692


namespace NUMINAMATH_GPT_ellipse_properties_l2366_236632

noncomputable def a_square : ℝ := 2
noncomputable def b_square : ℝ := 9 / 8
noncomputable def c_square : ℝ := a_square - b_square
noncomputable def c : ℝ := Real.sqrt c_square
noncomputable def distance_between_foci : ℝ := 2 * c
noncomputable def eccentricity : ℝ := c / Real.sqrt a_square

theorem ellipse_properties :
  (distance_between_foci = Real.sqrt 14) ∧ (eccentricity = Real.sqrt 7 / 4) := by
  sorry

end NUMINAMATH_GPT_ellipse_properties_l2366_236632


namespace NUMINAMATH_GPT_russia_is_one_third_bigger_l2366_236609

theorem russia_is_one_third_bigger (U : ℝ) (Canada Russia : ℝ) 
  (h1 : Canada = 1.5 * U) (h2 : Russia = 2 * U) : 
  (Russia - Canada) / Canada = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_russia_is_one_third_bigger_l2366_236609


namespace NUMINAMATH_GPT_integer_implies_perfect_square_l2366_236638

theorem integer_implies_perfect_square (n : ℕ) (h : ∃ m : ℤ, 2 + 2 * Real.sqrt (28 * (n ^ 2) + 1) = m) :
  ∃ k : ℤ, 2 + 2 * Real.sqrt (28 * (n ^ 2) + 1) = (k ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_integer_implies_perfect_square_l2366_236638


namespace NUMINAMATH_GPT_arrangement_count_SUCCESS_l2366_236626

-- Define the conditions for the problem
def letters : Finset String := {"S", "U", "C", "C", "E", "S", "S"}
def occurrences_S : Nat := 3
def occurrences_C : Nat := 2
def occurrences_other : Nat := 1 -- For 'U' and 'E'

-- State the theorem using these conditions
theorem arrangement_count_SUCCESS : 
  let N := letters.card
  N = 7 →
  occurrences_S = 3 →
  occurrences_C = 2 →
  occurrences_other = 1 →
  Nat.factorial N / (Nat.factorial occurrences_S * Nat.factorial occurrences_C * Nat.factorial occurrences_other * Nat.factorial occurrences_other) = 420 :=
by
  sorry

end NUMINAMATH_GPT_arrangement_count_SUCCESS_l2366_236626


namespace NUMINAMATH_GPT_problem_l2366_236685

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.sin x + a * Real.cos x

theorem problem (a : ℝ) (h₀ : a < 0) (h₁ : ∀ x : ℝ, f x a ≤ 2) : f (π / 6) a = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_l2366_236685


namespace NUMINAMATH_GPT_distance_between_sasha_and_kolya_when_sasha_finished_l2366_236673

-- Definitions based on the problem conditions
def distance_sasha : ℝ := 100
def distance_lesha_when_sasha_finished : ℝ := 90
def distance_kolya_when_lesha_finished : ℝ := 90

def velocity_lesha (v_s : ℝ) : ℝ := 0.9 * v_s
def velocity_kolya (v_s : ℝ) : ℝ := 0.81 * v_s

-- Theorem statement
theorem distance_between_sasha_and_kolya_when_sasha_finished (v_s : ℝ) :
  distance_sasha - (velocity_kolya v_s * (distance_sasha / v_s)) = 19 :=
  by sorry

end NUMINAMATH_GPT_distance_between_sasha_and_kolya_when_sasha_finished_l2366_236673


namespace NUMINAMATH_GPT_inequality_proof_l2366_236613

theorem inequality_proof (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) : a^2 > -a * b ∧ -a * b > b^2 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2366_236613


namespace NUMINAMATH_GPT_inside_circle_implies_line_intersects_circle_on_circle_implies_line_tangent_to_circle_outside_circle_implies_line_does_not_intersect_circle_l2366_236697

-- Definitions for the conditions
def inside_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1^2 + M.2^2 < r^2 ∧ (M.1 ≠ 0 ∨ M.2 ≠ 0)

def on_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1^2 + M.2^2 = r^2

def outside_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1^2 + M.2^2 > r^2

def line_l_intersects_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1 * M.1 + M.2 * M.2 < r^2 ∨ M.1 * M.1 + M.2 * M.2 = r^2

def line_l_tangent_to_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1 * M.1 + M.2 * M.2 = r^2

def line_l_does_not_intersect_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1 * M.1 + M.2 * M.2 > r^2

-- Propositions
theorem inside_circle_implies_line_intersects_circle (M : ℝ × ℝ) (r : ℝ) : 
  inside_circle M r → line_l_intersects_circle M r := 
sorry

theorem on_circle_implies_line_tangent_to_circle (M : ℝ × ℝ) (r : ℝ) :
  on_circle M r → line_l_tangent_to_circle M r :=
sorry

theorem outside_circle_implies_line_does_not_intersect_circle (M : ℝ × ℝ) (r : ℝ) :
  outside_circle M r → line_l_does_not_intersect_circle M r :=
sorry

end NUMINAMATH_GPT_inside_circle_implies_line_intersects_circle_on_circle_implies_line_tangent_to_circle_outside_circle_implies_line_does_not_intersect_circle_l2366_236697


namespace NUMINAMATH_GPT_net_pay_is_correct_l2366_236616

-- Define the gross pay and taxes paid as constants
def gross_pay : ℕ := 450
def taxes_paid : ℕ := 135

-- Define net pay as a function of gross pay and taxes paid
def net_pay (gross : ℕ) (taxes : ℕ) : ℕ := gross - taxes

-- The proof statement
theorem net_pay_is_correct : net_pay gross_pay taxes_paid = 315 := by
  sorry -- The proof goes here

end NUMINAMATH_GPT_net_pay_is_correct_l2366_236616


namespace NUMINAMATH_GPT_ratio_of_AC_to_BD_l2366_236637

theorem ratio_of_AC_to_BD (A B C D : ℝ) (AB BC AD AC BD : ℝ) 
  (h1 : AB = 2) (h2 : BC = 5) (h3 : AD = 14) (h4 : AC = AB + BC) (h5 : BD = AD - AB) :
  AC / BD = 7 / 12 := by
  sorry

end NUMINAMATH_GPT_ratio_of_AC_to_BD_l2366_236637


namespace NUMINAMATH_GPT_number_of_correct_propositions_l2366_236624

variable (Ω : Type) (R : Type) [Nonempty Ω] [Nonempty R]

-- Definitions of the conditions
def carsPassingIntersection (t : ℝ) : Ω → ℕ := sorry
def passengersInWaitingRoom (t : ℝ) : Ω → ℕ := sorry
def maximumFlowRiverEachYear : Ω → ℝ := sorry
def peopleExitingTheater (t : ℝ) : Ω → ℕ := sorry

-- Statement to prove the number of correct propositions
theorem number_of_correct_propositions : 4 = 4 := sorry

end NUMINAMATH_GPT_number_of_correct_propositions_l2366_236624


namespace NUMINAMATH_GPT_find_a_l2366_236667

theorem find_a (a : ℝ) (h_pos : 0 < a) 
  (prob : (2 / a) = (1 / 3)) : a = 6 :=
by sorry

end NUMINAMATH_GPT_find_a_l2366_236667


namespace NUMINAMATH_GPT_H2CO3_formation_l2366_236699

-- Define the given conditions
def one_to_one_reaction (a b : ℕ) := a = b

-- Define the reaction
theorem H2CO3_formation (m_CO2 m_H2O : ℕ) 
  (h : one_to_one_reaction m_CO2 m_H2O) : 
  m_CO2 = 2 → m_H2O = 2 → m_CO2 = 2 ∧ m_H2O = 2 := 
by 
  intros h1 h2
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_H2CO3_formation_l2366_236699


namespace NUMINAMATH_GPT_smallest_possible_sum_l2366_236690

theorem smallest_possible_sum :
  ∃ (B : ℕ) (c : ℕ), B + c = 34 ∧ 
    (B ≥ 0 ∧ B < 5) ∧ 
    (c > 7) ∧ 
    (31 * B = 4 * c + 4) := 
by
  sorry

end NUMINAMATH_GPT_smallest_possible_sum_l2366_236690


namespace NUMINAMATH_GPT_difference_between_percent_and_fraction_l2366_236672

-- Define the number
def num : ℕ := 140

-- Define the percentage and fraction calculations
def percent_65 (n : ℕ) : ℕ := (65 * n) / 100
def fraction_4_5 (n : ℕ) : ℕ := (4 * n) / 5

-- Define the problem's conditions and the required proof
theorem difference_between_percent_and_fraction : 
  percent_65 num ≤ fraction_4_5 num ∧ (fraction_4_5 num - percent_65 num = 21) :=
by
  sorry

end NUMINAMATH_GPT_difference_between_percent_and_fraction_l2366_236672


namespace NUMINAMATH_GPT_find_sequence_l2366_236639

noncomputable def seq (a : ℕ → ℝ) :=
  a 1 = 0 ∧ (∀ n, a (n + 1) = (n / (n + 1)) * (a n + 1))

theorem find_sequence {a : ℕ → ℝ} (h : seq a) :
  ∀ n, a n = (n - 1) / 2 :=
sorry

end NUMINAMATH_GPT_find_sequence_l2366_236639


namespace NUMINAMATH_GPT_seventh_term_of_arithmetic_sequence_l2366_236671

variable (a d : ℕ)

theorem seventh_term_of_arithmetic_sequence (h1 : 5 * a + 10 * d = 15) (h2 : a + 3 * d = 4) : a + 6 * d = 7 := 
by
  sorry

end NUMINAMATH_GPT_seventh_term_of_arithmetic_sequence_l2366_236671


namespace NUMINAMATH_GPT_Kate_relies_on_dumpster_diving_Upscale_stores_discard_items_Kate_frugal_habits_l2366_236657

structure Person :=
  (name : String)
  (age : Nat)
  (location : String)
  (occupation : String)

def kate : Person := {name := "Kate Hashimoto", age := 30, location := "New York", occupation := "CPA"}

-- Conditions
def lives_on_15_dollars_a_month (p : Person) : Prop := p = kate → true
def dumpster_diving (p : Person) : Prop := p = kate → true
def upscale_stores_discard_good_items : Prop := true
def frugal_habits (p : Person) : Prop := p = kate → true

-- Proof
theorem Kate_relies_on_dumpster_diving : lives_on_15_dollars_a_month kate ∧ dumpster_diving kate → true := 
by sorry

theorem Upscale_stores_discard_items : upscale_stores_discard_good_items → true := 
by sorry

theorem Kate_frugal_habits : frugal_habits kate → true := 
by sorry

end NUMINAMATH_GPT_Kate_relies_on_dumpster_diving_Upscale_stores_discard_items_Kate_frugal_habits_l2366_236657


namespace NUMINAMATH_GPT_simple_interest_principal_l2366_236676

theorem simple_interest_principal (R : ℝ) (P : ℝ) (h : P * 7 * (R + 2) / 100 = P * 7 * R / 100 + 140) : P = 1000 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_principal_l2366_236676


namespace NUMINAMATH_GPT_value_of_expression_l2366_236677

theorem value_of_expression (p q : ℚ) (h : p / q = 4 / 5) : 4 / 7 + (2 * q - p) / (2 * q + p) = 1 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2366_236677


namespace NUMINAMATH_GPT_number_of_correct_propositions_l2366_236646

def f (x b c : ℝ) := x * |x| + b * x + c

def proposition1 (b : ℝ) : Prop :=
  ∀ (x : ℝ), f x b 0 = -f (-x) b 0

def proposition2 (c : ℝ) : Prop :=
  c > 0 → ∃ (x : ℝ), ∀ (y : ℝ), f y 0 c = 0 → y = x

def proposition3 (b c : ℝ) : Prop :=
  ∀ (x : ℝ), f x b c = f (-x) b c + 2 * c

def proposition4 (b c : ℝ) : Prop :=
  ∀ (x₁ x₂ x₃ : ℝ), f x₁ b c = 0 → f x₂ b c = 0 → f x₃ b c = 0 → x₁ = x₂ ∨ x₂ = x₃ ∨ x₁ = x₃

theorem number_of_correct_propositions (b c : ℝ) : 
  1 + (if c > 0 then 1 else 0) + 1 + 0 = 3 :=
  sorry

end NUMINAMATH_GPT_number_of_correct_propositions_l2366_236646


namespace NUMINAMATH_GPT_solve_inequality_l2366_236642

theorem solve_inequality (x: ℝ) : (25 - 5 * Real.sqrt 3) ≤ x ∧ x ≤ (25 + 5 * Real.sqrt 3) ↔ x ^ 2 - 50 * x + 575 ≤ 25 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2366_236642


namespace NUMINAMATH_GPT_four_people_fill_pool_together_in_12_minutes_l2366_236674

def combined_pool_time (j s t e : ℕ) : ℕ := 
  1 / ((1 / j) + (1 / s) + (1 / t) + (1 / e))

theorem four_people_fill_pool_together_in_12_minutes : 
  ∀ (j s t e : ℕ), j = 30 → s = 45 → t = 90 → e = 60 → combined_pool_time j s t e = 12 := 
by 
  intros j s t e h_j h_s h_t h_e
  unfold combined_pool_time
  rw [h_j, h_s, h_t, h_e]
  have r1 : 1 / 30 = 1 / 30 := rfl
  have r2 : 1 / 45 = 1 / 45 := rfl
  have r3 : 1 / 90 = 1 / 90 := rfl
  have r4 : 1 / 60 = 1 / 60 := rfl
  rw [r1, r2, r3, r4]
  norm_num
  sorry

end NUMINAMATH_GPT_four_people_fill_pool_together_in_12_minutes_l2366_236674


namespace NUMINAMATH_GPT_rectangle_x_value_l2366_236644

theorem rectangle_x_value (x : ℝ) (h : (4 * x) * (x + 7) = 2 * (4 * x) + 2 * (x + 7)) : x = 0.675 := 
sorry

end NUMINAMATH_GPT_rectangle_x_value_l2366_236644


namespace NUMINAMATH_GPT_divisor_of_form_4k_minus_1_l2366_236665

theorem divisor_of_form_4k_minus_1
  (n : ℕ) (hn1 : Odd n) (hn_pos : 0 < n)
  (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h_eq : (1 / (x : ℚ) + 1 / (y : ℚ) = 4 / n)) :
  ∃ k : ℕ, ∃ d, d ∣ n ∧ d = 4 * k - 1 ∧ k ∈ Set.Ici 1 :=
sorry

end NUMINAMATH_GPT_divisor_of_form_4k_minus_1_l2366_236665


namespace NUMINAMATH_GPT_Mike_additional_money_needed_proof_l2366_236659

-- Definitions of conditions
def phone_cost : ℝ := 1300
def smartwatch_cost : ℝ := 500
def phone_discount : ℝ := 0.10
def smartwatch_discount : ℝ := 0.15
def sales_tax : ℝ := 0.07
def mike_has_percentage : ℝ := 0.40

-- Definitions of intermediate calculations
def discounted_phone_cost : ℝ := phone_cost * (1 - phone_discount)
def discounted_smartwatch_cost : ℝ := smartwatch_cost * (1 - smartwatch_discount)
def total_cost_before_tax : ℝ := discounted_phone_cost + discounted_smartwatch_cost
def total_tax : ℝ := total_cost_before_tax * sales_tax
def total_cost_after_tax : ℝ := total_cost_before_tax + total_tax
def mike_has_amount : ℝ := total_cost_after_tax * mike_has_percentage
def additional_money_needed : ℝ := total_cost_after_tax - mike_has_amount

-- Theorem statement
theorem Mike_additional_money_needed_proof :
  additional_money_needed = 1023.99 :=
by sorry

end NUMINAMATH_GPT_Mike_additional_money_needed_proof_l2366_236659


namespace NUMINAMATH_GPT_measure_of_angle_l2366_236629

theorem measure_of_angle (x : ℝ) 
  (h₁ : 180 - x = 3 * x - 10) : x = 47.5 :=
by 
  sorry

end NUMINAMATH_GPT_measure_of_angle_l2366_236629


namespace NUMINAMATH_GPT_simple_interest_two_years_l2366_236682
-- Import the necessary Lean library for mathematical concepts

-- Define the problem conditions and the proof statement
theorem simple_interest_two_years (P r t : ℝ) (CI SI : ℝ)
  (hP : P = 17000) (ht : t = 2) (hCI : CI = 11730) : SI = 5100 :=
by
  -- Principal (P), Rate (r), and Time (t) definitions
  let P := 17000
  let t := 2

  -- Given Compound Interest (CI)
  let CI := 11730

  -- Correct value for Simple Interest (SI) that we need to prove
  let SI := 5100

  -- Formalize the assumptions
  have h1 : P = 17000 := rfl
  have h2 : t = 2 := rfl
  have h3 : CI = 11730 := rfl

  -- Crucial parts of the problem are used here
  sorry  -- This is a placeholder for the actual proof steps

end NUMINAMATH_GPT_simple_interest_two_years_l2366_236682


namespace NUMINAMATH_GPT_area_of_paper_is_500_l2366_236670

-- Define the width and length of the rectangular drawing paper
def width := 25
def length := 20

-- Define the formula for the area of a rectangle
def area (w : Nat) (l : Nat) : Nat := w * l

-- Prove that the area of the paper is 500 square centimeters
theorem area_of_paper_is_500 : area width length = 500 := by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_area_of_paper_is_500_l2366_236670


namespace NUMINAMATH_GPT_train_speed_kmph_l2366_236678

def train_length : ℝ := 360
def bridge_length : ℝ := 140
def time_to_pass : ℝ := 40
def mps_to_kmph (speed : ℝ) : ℝ := speed * 3.6

theorem train_speed_kmph : mps_to_kmph ((train_length + bridge_length) / time_to_pass) = 45 := 
by {
  sorry
}

end NUMINAMATH_GPT_train_speed_kmph_l2366_236678


namespace NUMINAMATH_GPT_fraction_to_decimal_l2366_236662

theorem fraction_to_decimal : (58 : ℚ) / 125 = 0.464 := by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l2366_236662


namespace NUMINAMATH_GPT_prime_square_remainder_l2366_236687

theorem prime_square_remainder (p : ℕ) (hp : Nat.Prime p) (h5 : p > 5) : 
  ∃! r : ℕ, r < 180 ∧ (p^2 ≡ r [MOD 180]) := 
by
  sorry

end NUMINAMATH_GPT_prime_square_remainder_l2366_236687


namespace NUMINAMATH_GPT_wall_width_8_l2366_236694

theorem wall_width_8 (w h l : ℝ) (V : ℝ) 
  (h_eq : h = 6 * w) 
  (l_eq : l = 7 * h) 
  (vol_eq : w * h * l = 129024) : 
  w = 8 := 
by 
  sorry

end NUMINAMATH_GPT_wall_width_8_l2366_236694


namespace NUMINAMATH_GPT_screws_weight_l2366_236648

theorem screws_weight (x y : ℕ) 
  (h1 : 3 * x + 2 * y = 319) 
  (h2 : 2 * x + 3 * y = 351) : 
  x = 51 ∧ y = 83 :=
by 
  sorry

end NUMINAMATH_GPT_screws_weight_l2366_236648


namespace NUMINAMATH_GPT_delores_money_left_l2366_236651

def initial : ℕ := 450
def computer_cost : ℕ := 400
def printer_cost : ℕ := 40
def money_left (initial computer_cost printer_cost : ℕ) : ℕ := initial - (computer_cost + printer_cost)

theorem delores_money_left : money_left initial computer_cost printer_cost = 10 := by
  sorry

end NUMINAMATH_GPT_delores_money_left_l2366_236651


namespace NUMINAMATH_GPT_cube_root_simplification_l2366_236640

theorem cube_root_simplification (c d : ℕ) (h1 : c = 3) (h2 : d = 100) : c + d = 103 :=
by
  sorry

end NUMINAMATH_GPT_cube_root_simplification_l2366_236640


namespace NUMINAMATH_GPT_B_pow_2021_eq_B_l2366_236680

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![1 / 2, 0, -Real.sqrt 3 / 2],
  ![0, -1, 0],
  ![Real.sqrt 3 / 2, 0, 1 / 2]
]

theorem B_pow_2021_eq_B : B ^ 2021 = B := 
by sorry

end NUMINAMATH_GPT_B_pow_2021_eq_B_l2366_236680


namespace NUMINAMATH_GPT_sum_of_perimeters_of_squares_l2366_236652

theorem sum_of_perimeters_of_squares
  (x y : ℝ)
  (h1 : x^2 + y^2 = 130)
  (h2 : x^2 / y^2 = 4) :
  4*x + 4*y = 12*Real.sqrt 26 := by
  sorry

end NUMINAMATH_GPT_sum_of_perimeters_of_squares_l2366_236652


namespace NUMINAMATH_GPT_weight_of_person_replaced_l2366_236654

theorem weight_of_person_replaced (W : ℝ) (old_avg_weight : ℝ) (new_avg_weight : ℝ)
  (h_avg_increase : new_avg_weight = old_avg_weight + 1.5) (new_person_weight : ℝ) :
  ∃ (person_replaced_weight : ℝ), new_person_weight = 77 ∧ old_avg_weight = W / 8 ∧
  new_avg_weight = (W - person_replaced_weight + 77) / 8 ∧ person_replaced_weight = 65 := by
    sorry

end NUMINAMATH_GPT_weight_of_person_replaced_l2366_236654


namespace NUMINAMATH_GPT_machine_A_produces_40_percent_l2366_236625

theorem machine_A_produces_40_percent (p : ℝ) : 
  (0 < p ∧ p < 1 ∧
  (0.0156 = p * 0.009 + (1 - p) * 0.02)) → 
  p = 0.4 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_machine_A_produces_40_percent_l2366_236625


namespace NUMINAMATH_GPT_octagon_diagonals_l2366_236647

def num_sides := 8

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem octagon_diagonals : num_diagonals num_sides = 20 :=
by
  sorry

end NUMINAMATH_GPT_octagon_diagonals_l2366_236647


namespace NUMINAMATH_GPT_division_decimal_l2366_236607

theorem division_decimal (x : ℝ) (h : x = 0.3333): 12 / x = 36 :=
  by
    sorry

end NUMINAMATH_GPT_division_decimal_l2366_236607


namespace NUMINAMATH_GPT_area_of_fourth_rectangle_l2366_236658

theorem area_of_fourth_rectangle (a b c d : ℕ) (h1 : a = 18) (h2 : b = 27) (h3 : c = 12) :
d = 93 :=
by
  -- Problem reduces to showing that d equals 93 using the given h1, h2, h3
  sorry

end NUMINAMATH_GPT_area_of_fourth_rectangle_l2366_236658


namespace NUMINAMATH_GPT_max_bricks_truck_can_carry_l2366_236601

-- Define the truck's capacity in terms of bags of sand and bricks
def max_sand_bags := 50
def max_bricks := 400
def sand_to_bricks_ratio := 8

-- Define the current number of sand bags already on the truck
def current_sand_bags := 32

-- Define the number of bricks equivalent to a given number of sand bags
def equivalent_bricks (sand_bags: ℕ) := sand_bags * sand_to_bricks_ratio

-- Define the remaining capacity in terms of bags of sand
def remaining_sand_bags := max_sand_bags - current_sand_bags

-- Define the maximum number of additional bricks the truck can carry
def max_additional_bricks := equivalent_bricks remaining_sand_bags

-- Prove the number of additional bricks the truck can carry is 144
theorem max_bricks_truck_can_carry : max_additional_bricks = 144 := by
  sorry

end NUMINAMATH_GPT_max_bricks_truck_can_carry_l2366_236601


namespace NUMINAMATH_GPT_max_profit_l2366_236604

noncomputable def fixed_cost := 20000
noncomputable def variable_cost (x : ℝ) : ℝ :=
  if x < 8 then (1/3) * x^2 + 2 * x else 7 * x + 100 / x - 37
noncomputable def sales_price_per_unit : ℝ := 6
noncomputable def profit (x : ℝ) : ℝ :=
  let revenue := sales_price_per_unit * x
  let cost := fixed_cost / 10000 + variable_cost x
  revenue - cost

theorem max_profit : ∃ x : ℝ, (0 < x) ∧ (15 = profit 10) :=
by {
  sorry
}

end NUMINAMATH_GPT_max_profit_l2366_236604


namespace NUMINAMATH_GPT_matrix_product_is_zero_l2366_236635

-- Define the two matrices
def A (b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, d, -c], ![-d, 0, b], ![c, -b, 0]]

def B (b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![d^2, b * d, c * d], ![b * d, b^2, b * c], ![c * d, b * c, c^2]]

-- Define the zero matrix
def zero_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 0, 0], ![0, 0, 0], ![0, 0, 0]]

-- The theorem to prove
theorem matrix_product_is_zero (b c d : ℝ) : A b c d * B b c d = zero_matrix :=
by sorry

end NUMINAMATH_GPT_matrix_product_is_zero_l2366_236635


namespace NUMINAMATH_GPT_valid_transformation_b_l2366_236608

theorem valid_transformation_b (a b : ℚ) : ((-a - b) / (a + b) = -1) := sorry

end NUMINAMATH_GPT_valid_transformation_b_l2366_236608


namespace NUMINAMATH_GPT_emma_additional_miles_l2366_236612

theorem emma_additional_miles :
  ∀ (initial_distance : ℝ) (initial_speed : ℝ) (additional_speed : ℝ) (desired_avg_speed : ℝ) (total_distance : ℝ) (additional_distance : ℝ),
    initial_distance = 20 →
    initial_speed = 40 →
    additional_speed = 70 →
    desired_avg_speed = 60 →
    total_distance = initial_distance + additional_distance →
    (total_distance / ((initial_distance / initial_speed) + (additional_distance / additional_speed))) = desired_avg_speed →
    additional_distance = 70 :=
by
  intros initial_distance initial_speed additional_speed desired_avg_speed total_distance additional_distance
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_emma_additional_miles_l2366_236612


namespace NUMINAMATH_GPT_nat_divisible_by_five_l2366_236650

theorem nat_divisible_by_five (a b : ℕ) (h : 5 ∣ (a * b)) : (5 ∣ a) ∨ (5 ∣ b) :=
by
  have h₀ : ¬ ((5 ∣ a) ∨ (5 ∣ b)) → ¬ (5 ∣ (a * b)) := sorry
  -- Proof by contradiction steps go here
  sorry

end NUMINAMATH_GPT_nat_divisible_by_five_l2366_236650


namespace NUMINAMATH_GPT_value_of_fraction_l2366_236643

variables {a_1 q : ℝ}

-- Define the conditions and the mathematical equivalent of the problem.
def geometric_sequence (a_1 q : ℝ) (h_pos : a_1 > 0 ∧ q > 0) :=
  2 * a_1 + a_1 * q = a_1 * q^2

theorem value_of_fraction (h_pos : a_1 > 0 ∧ q > 0) (h_geom : geometric_sequence a_1 q h_pos) :
  (a_1 * q^3 + a_1 * q^4) / (a_1 * q^2 + a_1 * q^3) = 2 :=
sorry

end NUMINAMATH_GPT_value_of_fraction_l2366_236643


namespace NUMINAMATH_GPT_train_crossing_time_l2366_236656

def speed := 60 -- in km/hr
def length := 300 -- in meters
def speed_in_m_per_s := (60 * 1000) / 3600 -- converting speed from km/hr to m/s
def expected_time := 18 -- in seconds

theorem train_crossing_time :
  (300 / (speed_in_m_per_s)) = expected_time :=
sorry

end NUMINAMATH_GPT_train_crossing_time_l2366_236656


namespace NUMINAMATH_GPT_isosceles_perimeter_l2366_236636

noncomputable def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem isosceles_perimeter
  (k : ℝ)
  (a b : ℝ)
  (h1 : 4 = a)
  (h2 : k * b^2 - (k + 8) * b + 8 = 0)
  (h3 : k ≠ 0)
  (h4 : is_triangle 4 a a) : a + 4 + a = 9 :=
sorry

end NUMINAMATH_GPT_isosceles_perimeter_l2366_236636


namespace NUMINAMATH_GPT_unique_real_solution_l2366_236668

theorem unique_real_solution : ∃ x : ℝ, (∀ t : ℝ, x^2 - t * x + 36 = 0 ∧ x^2 - 8 * x + t = 0) ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_unique_real_solution_l2366_236668


namespace NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l2366_236630

-- Define the conditions p and q
def p (x : ℝ) : Prop := |x - 1| < 2
def q (x : ℝ) : Prop := x^2 - 5*x - 6 < 0

-- State the theorem that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q (x : ℝ) :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) :=
by
  sorry

end NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l2366_236630


namespace NUMINAMATH_GPT_expression_zero_iff_x_eq_three_l2366_236619

theorem expression_zero_iff_x_eq_three (x : ℝ) :
  (4 * x - 8 ≠ 0) → ((x^2 - 6 * x + 9 = 0) ↔ (x = 3)) :=
by
  sorry

end NUMINAMATH_GPT_expression_zero_iff_x_eq_three_l2366_236619


namespace NUMINAMATH_GPT_find_present_age_of_eldest_l2366_236631

noncomputable def eldest_present_age (x : ℕ) : ℕ :=
  8 * x

theorem find_present_age_of_eldest :
  ∃ x : ℕ, 20 * x - 21 = 59 ∧ eldest_present_age x = 32 :=
by
  sorry

end NUMINAMATH_GPT_find_present_age_of_eldest_l2366_236631


namespace NUMINAMATH_GPT_special_four_digit_numbers_l2366_236666

noncomputable def count_special_four_digit_numbers : Nat :=
  -- The task is to define the number of four-digit numbers formed using the digits {0, 1, 2, 3, 4}
  -- that contain the digit 0 and have exactly two digits repeating
  144

theorem special_four_digit_numbers : count_special_four_digit_numbers = 144 := by
  sorry

end NUMINAMATH_GPT_special_four_digit_numbers_l2366_236666


namespace NUMINAMATH_GPT_cos_alpha_given_tan_alpha_and_quadrant_l2366_236679

theorem cos_alpha_given_tan_alpha_and_quadrant 
  (α : ℝ) 
  (h1 : Real.tan α = -1/3)
  (h2 : π/2 < α ∧ α < π) : 
  Real.cos α = -3*Real.sqrt 10 / 10 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_given_tan_alpha_and_quadrant_l2366_236679


namespace NUMINAMATH_GPT_jill_water_jars_l2366_236693

theorem jill_water_jars (x : ℕ) (h : x * (1 / 4 + 1 / 2 + 1) = 28) : 3 * x = 48 :=
by
  sorry

end NUMINAMATH_GPT_jill_water_jars_l2366_236693


namespace NUMINAMATH_GPT_find_real_x_l2366_236622

theorem find_real_x (x : ℝ) : 
  (2 < x / (3 * x - 7) ∧ x / (3 * x - 7) ≤ 6) ↔ (7 / 3 < x ∧ x ≤ 14 / 5) :=
by sorry

end NUMINAMATH_GPT_find_real_x_l2366_236622


namespace NUMINAMATH_GPT_passing_percentage_l2366_236610

theorem passing_percentage
  (marks_obtained : ℕ)
  (marks_failed_by : ℕ)
  (max_marks : ℕ)
  (h_marks_obtained : marks_obtained = 92)
  (h_marks_failed_by : marks_failed_by = 40)
  (h_max_marks : max_marks = 400) :
  (marks_obtained + marks_failed_by) / max_marks * 100 = 33 := 
by
  sorry

end NUMINAMATH_GPT_passing_percentage_l2366_236610


namespace NUMINAMATH_GPT_total_time_to_complete_work_l2366_236602

-- Definitions based on conditions
variable (W : ℝ) -- W is the total work
variable (Mahesh_days : ℝ := 35) -- Mahesh can complete the work in 35 days
variable (Mahesh_working_days : ℝ := 20) -- Mahesh works for 20 days
variable (Rajesh_days : ℝ := 30) -- Rajesh finishes the remaining work in 30 days

-- Proof statement
theorem total_time_to_complete_work : Mahesh_working_days + Rajesh_days = 50 :=
by
  sorry

end NUMINAMATH_GPT_total_time_to_complete_work_l2366_236602


namespace NUMINAMATH_GPT_iced_coffee_cost_correct_l2366_236664

-- Definitions based on the conditions 
def coffee_cost_per_day (iced_coffee_cost : ℝ) : ℝ := 3 + iced_coffee_cost
def total_spent (days : ℕ) (iced_coffee_cost : ℝ) : ℝ := days * coffee_cost_per_day iced_coffee_cost

-- Proof statement
theorem iced_coffee_cost_correct (iced_coffee_cost : ℝ) (h : total_spent 20 iced_coffee_cost = 110) : iced_coffee_cost = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_iced_coffee_cost_correct_l2366_236664


namespace NUMINAMATH_GPT_platform_length_l2366_236617

noncomputable def train_length := 420 -- length of the train in meters
noncomputable def time_to_cross_platform := 60 -- time to cross the platform in seconds
noncomputable def time_to_cross_pole := 30 -- time to cross the signal pole in seconds

theorem platform_length :
  ∃ L, L = 420 ∧ train_length / time_to_cross_pole = train_length / time_to_cross_platform * (train_length + L) / time_to_cross_platform :=
by
  use 420
  sorry

end NUMINAMATH_GPT_platform_length_l2366_236617


namespace NUMINAMATH_GPT_remaining_files_l2366_236688

def initial_music_files : ℕ := 16
def initial_video_files : ℕ := 48
def deleted_files : ℕ := 30

theorem remaining_files :
  initial_music_files + initial_video_files - deleted_files = 34 := 
by
  sorry

end NUMINAMATH_GPT_remaining_files_l2366_236688


namespace NUMINAMATH_GPT_solve_for_x_l2366_236649

theorem solve_for_x (x : ℕ) : (8^3 + 8^3 + 8^3 + 8^3 = 2^x) → x = 11 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l2366_236649
