import Mathlib

namespace NUMINAMATH_GPT_equation_of_line_l1864_186487

variable {a b k T : ℝ}

theorem equation_of_line (h_b_ne_zero : b ≠ 0)
  (h_line_passing_through : ∃ (line : ℝ → ℝ), line (-a) = b)
  (h_triangle_area : ∃ (h : ℝ), T = 1 / 2 * ka * (h - b))
  (h_base_length : ∃ (base : ℝ), base = ka) :
  ∃ (x y : ℝ), 2 * T * x - k * a^2 * y + k * a^2 * b + 2 * a * T = 0 :=
sorry

end NUMINAMATH_GPT_equation_of_line_l1864_186487


namespace NUMINAMATH_GPT_ratio_comparison_l1864_186424

-- Define the ratios in the standard and sport formulations
def ratio_flavor_corn_standard : ℚ := 1 / 12
def ratio_flavor_water_standard : ℚ := 1 / 30
def ratio_flavor_water_sport : ℚ := 1 / 60

-- Define the amounts of corn syrup and water in the sport formulation
def corn_syrup_sport : ℚ := 2
def water_sport : ℚ := 30

-- Calculate the amount of flavoring in the sport formulation
def flavoring_sport : ℚ := water_sport / 60

-- Calculate the ratio of flavoring to corn syrup in the sport formulation
def ratio_flavor_corn_sport : ℚ := flavoring_sport / corn_syrup_sport

-- Define the theorem to prove the ratio comparison
theorem ratio_comparison :
  (ratio_flavor_corn_sport / ratio_flavor_corn_standard) = 3 :=
by
  -- Using the given conditions and definitions, prove the theorem
  sorry

end NUMINAMATH_GPT_ratio_comparison_l1864_186424


namespace NUMINAMATH_GPT_inequality_transitivity_l1864_186466

theorem inequality_transitivity (a b c : ℝ) (h : a > b) : 
  a + c > b + c :=
sorry

end NUMINAMATH_GPT_inequality_transitivity_l1864_186466


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l1864_186435

theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 / 4 - y^2 = 1) → (y = x / 2 ∨ y = -x / 2) :=
sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l1864_186435


namespace NUMINAMATH_GPT_smallest_positive_debt_resolvable_l1864_186467

/-- The value of a sheep in dollars -/
def sheep_value : ℕ := 250

/-- The value of a lamb in dollars -/
def lamb_value : ℕ := 150

/-- Given a debt D that can be expressed in the form of 250s + 150l for integers s and l,
prove that the smallest positive amount of D is 50 dollars -/
theorem smallest_positive_debt_resolvable : 
  ∃ (s l : ℤ), sheep_value * s + lamb_value * l = 50 :=
sorry

end NUMINAMATH_GPT_smallest_positive_debt_resolvable_l1864_186467


namespace NUMINAMATH_GPT_find_m_l1864_186426

def vec_a (m : ℝ) : ℝ × ℝ := (1, 2 * m)
def vec_b (m : ℝ) : ℝ × ℝ := (m + 1, 1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_m (m : ℝ) : dot_product (vec_a m) (vec_b m) = 0 ↔ m = -1/3 := by 
  sorry

end NUMINAMATH_GPT_find_m_l1864_186426


namespace NUMINAMATH_GPT_gretchen_charge_per_drawing_l1864_186432

-- Given conditions
def sold_on_Saturday : ℕ := 24
def sold_on_Sunday : ℕ := 16
def total_amount : ℝ := 800
def total_drawings := sold_on_Saturday + sold_on_Sunday

-- Assertion to prove
theorem gretchen_charge_per_drawing (x : ℝ) (h : total_drawings * x = total_amount) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_gretchen_charge_per_drawing_l1864_186432


namespace NUMINAMATH_GPT_pet_food_weight_in_ounces_l1864_186495

-- Define the given conditions
def cat_food_bags := 2
def cat_food_weight_per_bag := 3 -- in pounds
def dog_food_bags := 2
def additional_dog_food_weight := 2 -- additional weight per bag compared to cat food
def pounds_to_ounces := 16

-- Calculate the total weight of cat food in pounds
def total_cat_food_weight := cat_food_bags * cat_food_weight_per_bag

-- Calculate the weight of each bag of dog food in pounds
def dog_food_weight_per_bag := cat_food_weight_per_bag + additional_dog_food_weight

-- Calculate the total weight of dog food in pounds
def total_dog_food_weight := dog_food_bags * dog_food_weight_per_bag

-- Calculate the total weight of pet food in pounds
def total_pet_food_weight_pounds := total_cat_food_weight + total_dog_food_weight

-- Convert the total weight to ounces
def total_pet_food_weight_ounces := total_pet_food_weight_pounds * pounds_to_ounces

-- Statement of the problem in Lean 4
theorem pet_food_weight_in_ounces : total_pet_food_weight_ounces = 256 := by
  sorry

end NUMINAMATH_GPT_pet_food_weight_in_ounces_l1864_186495


namespace NUMINAMATH_GPT_cos_pi_over_2_plus_2theta_l1864_186439

theorem cos_pi_over_2_plus_2theta (θ : ℝ) (hcos : Real.cos θ = 1 / 3) (hθ : 0 < θ ∧ θ < Real.pi) :
    Real.cos (Real.pi / 2 + 2 * θ) = - (4 * Real.sqrt 2) / 9 := 
sorry

end NUMINAMATH_GPT_cos_pi_over_2_plus_2theta_l1864_186439


namespace NUMINAMATH_GPT_maximum_value_x_2y_2z_l1864_186423

noncomputable def max_sum (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) : ℝ :=
  x + 2*y + 2*z

theorem maximum_value_x_2y_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) : 
  max_sum x y z h ≤ 15 :=
sorry

end NUMINAMATH_GPT_maximum_value_x_2y_2z_l1864_186423


namespace NUMINAMATH_GPT_quarters_for_chips_l1864_186411

def total_quarters : ℕ := 16
def quarters_for_soda : ℕ := 12

theorem quarters_for_chips : (total_quarters - quarters_for_soda) = 4 :=
  by 
    sorry

end NUMINAMATH_GPT_quarters_for_chips_l1864_186411


namespace NUMINAMATH_GPT_find_h_l1864_186433

-- Define the polynomial f(x)
def f (x : ℤ) := x^4 - 2 * x^3 + x - 1

-- Define the condition that f(x) + h(x) = 3x^2 + 5x - 4
def condition (f h : ℤ → ℤ) := ∀ x, f x + h x = 3 * x^2 + 5 * x - 4

-- Define the solution for h(x) to be proved
def h_solution (x : ℤ) := -x^4 + 2 * x^3 + 3 * x^2 + 4 * x - 3

-- State the theorem to be proved
theorem find_h (h : ℤ → ℤ) (H : condition f h) : h = h_solution :=
by
  sorry

end NUMINAMATH_GPT_find_h_l1864_186433


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1864_186445

variable (α : ℝ)
variable (tan_alpha_two : Real.tan α = 2)

theorem problem_1 : (2 * Real.cos α + 3 * Real.sin α) / (3 * Real.cos α + Real.sin α) = 8 / 5 :=
by
  sorry

theorem problem_2 : (Real.cos α ^ 2 + Real.sin α * Real.cos α) / (2 * Real.sin α * Real.cos α + Real.sin α ^ 2) = 3 / 8 :=
by
  sorry

theorem problem_3 : (Real.sin α ^ 2 - Real.sin α * Real.cos α + 2) = 12 / 5 :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1864_186445


namespace NUMINAMATH_GPT_range_a_l1864_186431

noncomputable def A (a : ℝ) : Set ℝ := {x | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5}

noncomputable def B : Set ℝ := {x | x < -1 ∨ x > 16}

theorem range_a (a : ℝ) : (A a ∩ B = A a) → (a < 6 ∨ a > 7.5) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_a_l1864_186431


namespace NUMINAMATH_GPT_emily_small_gardens_l1864_186456

theorem emily_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) (num_small_gardens : ℕ) :
  total_seeds = 41 →
  big_garden_seeds = 29 →
  seeds_per_small_garden = 4 →
  num_small_gardens = (total_seeds - big_garden_seeds) / seeds_per_small_garden →
  num_small_gardens = 3 :=
by
  intros h_total h_big h_seeds_per_small h_num_small
  rw [h_total, h_big, h_seeds_per_small] at h_num_small
  exact h_num_small

end NUMINAMATH_GPT_emily_small_gardens_l1864_186456


namespace NUMINAMATH_GPT_circumcircle_radius_l1864_186407

-- Here we define the necessary conditions and prove the radius.
theorem circumcircle_radius
  (A B C : Type)
  (AB : ℝ)
  (angle_B : ℝ)
  (angle_A : ℝ)
  (h_AB : AB = 2)
  (h_angle_B : angle_B = 120)
  (h_angle_A : angle_A = 30) :
  ∃ R, R = 2 :=
by
  -- We will skip the proof using sorry
  sorry

end NUMINAMATH_GPT_circumcircle_radius_l1864_186407


namespace NUMINAMATH_GPT_mason_grandmother_age_l1864_186473

theorem mason_grandmother_age (mason_age: ℕ) (sydney_age: ℕ) (father_age: ℕ) (grandmother_age: ℕ)
  (h1: mason_age = 20)
  (h2: mason_age * 3 = sydney_age)
  (h3: sydney_age + 6 = father_age)
  (h4: father_age * 2 = grandmother_age) : 
  grandmother_age = 132 :=
by
  sorry

end NUMINAMATH_GPT_mason_grandmother_age_l1864_186473


namespace NUMINAMATH_GPT_min_value_of_quadratic_l1864_186449

theorem min_value_of_quadratic : ∀ x : ℝ, z = x^2 + 16*x + 20 → ∃ m : ℝ, m ≤ z :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l1864_186449


namespace NUMINAMATH_GPT_unique_solution_values_a_l1864_186436

theorem unique_solution_values_a (a : ℝ) : 
  (∃ x y : ℝ, |x| + |y - 1| = 1 ∧ y = a * x + 2012) ∧ 
  (∀ x1 y1 x2 y2 : ℝ, (|x1| + |y1 - 1| = 1 ∧ y1 = a * x1 + 2012) ∧ 
                      (|x2| + |y2 - 1| = 1 ∧ y2 = a * x2 + 2012) → 
                      (x1 = x2 ∧ y1 = y2)) ↔ 
  a = 2011 ∨ a = -2011 := 
sorry

end NUMINAMATH_GPT_unique_solution_values_a_l1864_186436


namespace NUMINAMATH_GPT_least_number_to_subtract_l1864_186496

theorem least_number_to_subtract (n : ℕ) (p : ℕ) (hdiv : p = 47) (hn : n = 929) 
: ∃ k, n - 44 = k * p := by
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_l1864_186496


namespace NUMINAMATH_GPT_quadrilateral_side_length_l1864_186406

-- Definitions
def inscribed_quadrilateral (a b c d r : ℝ) : Prop :=
  ∃ (O : ℝ) (A B C D : ℝ), 
    O = r ∧ 
    A = a ∧ B = b ∧ C = c ∧ 
    (r^2 + r^2 = (a^2 + b^2) / 2) ∧
    (r^2 + r^2 = (b^2 + c^2) / 2) ∧
    (r^2 + r^2 = (c^2 + d^2) / 2)

-- Theorem statement
theorem quadrilateral_side_length :
  inscribed_quadrilateral 250 250 100 200 250 :=
sorry

end NUMINAMATH_GPT_quadrilateral_side_length_l1864_186406


namespace NUMINAMATH_GPT_find_radius_l1864_186408

noncomputable def radius (π : ℝ) : Prop :=
  ∃ r : ℝ, π * r^2 + 2 * r - 2 * π * r = 12 ∧ r = Real.sqrt (12 / π)

theorem find_radius (π : ℝ) (hπ : π > 0) : 
  radius π :=
sorry

end NUMINAMATH_GPT_find_radius_l1864_186408


namespace NUMINAMATH_GPT_flight_duration_l1864_186425

noncomputable def departure_time_pst := 9 * 60 + 15 -- in minutes
noncomputable def arrival_time_est := 17 * 60 + 40 -- in minutes
noncomputable def time_difference := 3 * 60 -- in minutes

theorem flight_duration (h m : ℕ) 
  (h_cond : 0 < m ∧ m < 60) 
  (total_flight_time : (arrival_time_est - (departure_time_pst + time_difference)) = h * 60 + m) : 
  h + m = 30 :=
sorry

end NUMINAMATH_GPT_flight_duration_l1864_186425


namespace NUMINAMATH_GPT_angle_no_complement_greater_than_90_l1864_186479

-- Definition of angle
def angle (A : ℝ) : Prop := 
  A = 100 + (15 / 60)

-- Definition of complement
def has_complement (A : ℝ) : Prop :=
  A < 90

-- Theorem: Angles greater than 90 degrees do not have complements
theorem angle_no_complement_greater_than_90 {A : ℝ} (h: angle A) : ¬ has_complement A :=
by sorry

end NUMINAMATH_GPT_angle_no_complement_greater_than_90_l1864_186479


namespace NUMINAMATH_GPT_sqrt_of_16_l1864_186434

theorem sqrt_of_16 (x : ℝ) (hx : x^2 = 16) : x = 4 ∨ x = -4 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_of_16_l1864_186434


namespace NUMINAMATH_GPT_total_distance_after_fourth_bounce_l1864_186464

noncomputable def total_distance_traveled (initial_height : ℝ) (bounce_ratio : ℝ) (num_bounces : ℕ) : ℝ :=
  let fall_distances := (List.range (num_bounces + 1)).map (λ n => initial_height * bounce_ratio^n)
  let rise_distances := (List.range num_bounces).map (λ n => initial_height * bounce_ratio^(n+1))
  fall_distances.sum + rise_distances.sum

theorem total_distance_after_fourth_bounce :
  total_distance_traveled 25 (5/6 : ℝ) 4 = 154.42 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_after_fourth_bounce_l1864_186464


namespace NUMINAMATH_GPT_muffin_banana_ratio_l1864_186440

variables (m b : ℝ)

theorem muffin_banana_ratio (h1 : 4 * m + 3 * b = x) 
                            (h2 : 2 * (4 * m + 3 * b) = 2 * m + 16 * b) : 
                            m / b = 5 / 3 :=
by sorry

end NUMINAMATH_GPT_muffin_banana_ratio_l1864_186440


namespace NUMINAMATH_GPT_range_of_a_l1864_186461

variable {a x : ℝ}

theorem range_of_a (h_eq : 2 * (x + a) = x + 3) (h_ineq : 2 * x - 10 > 8 * a) : a < -1 / 3 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1864_186461


namespace NUMINAMATH_GPT_pow_mod_eq_residue_l1864_186454

theorem pow_mod_eq_residue :
  (3 : ℤ)^(2048) % 11 = 5 :=
sorry

end NUMINAMATH_GPT_pow_mod_eq_residue_l1864_186454


namespace NUMINAMATH_GPT_relationship_t_s_l1864_186405

theorem relationship_t_s (a b : ℝ) : 
  let t := a + 2 * b
  let s := a + b^2 + 1
  t <= s :=
by
  sorry

end NUMINAMATH_GPT_relationship_t_s_l1864_186405


namespace NUMINAMATH_GPT_total_students_is_45_l1864_186477

def num_students_in_class 
  (excellent_chinese : ℕ) 
  (excellent_math : ℕ) 
  (excellent_both : ℕ) 
  (no_excellent : ℕ) : ℕ :=
  excellent_chinese + excellent_math - excellent_both + no_excellent

theorem total_students_is_45 
  (h1 : excellent_chinese = 15)
  (h2 : excellent_math = 18)
  (h3 : excellent_both = 8)
  (h4 : no_excellent = 20) : 
  num_students_in_class excellent_chinese excellent_math excellent_both no_excellent = 45 := 
  by 
    sorry

end NUMINAMATH_GPT_total_students_is_45_l1864_186477


namespace NUMINAMATH_GPT_unique_solution_system_eqns_l1864_186451

theorem unique_solution_system_eqns (a b c : ℕ) :
  (a^3 - b^3 - c^3 = 3 * a * b * c) ∧ (a^2 = 2 * (b + c)) ↔ (a = 2 ∧ b = 1 ∧ c = 1) := by 
  sorry

end NUMINAMATH_GPT_unique_solution_system_eqns_l1864_186451


namespace NUMINAMATH_GPT_range_of_a_l1864_186415

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → 2 * x > x^2 + a) → a < -8 :=
by
  intro h
  -- Complete the proof by showing that 2x - x^2 has a minimum value of -8 on [-2, 3] and hence proving a < -8.
  sorry

end NUMINAMATH_GPT_range_of_a_l1864_186415


namespace NUMINAMATH_GPT_middle_number_of_consecutive_sum_30_l1864_186401

theorem middle_number_of_consecutive_sum_30 (n : ℕ) (h : n + (n + 1) + (n + 2) = 30) : n + 1 = 10 :=
by
  sorry

end NUMINAMATH_GPT_middle_number_of_consecutive_sum_30_l1864_186401


namespace NUMINAMATH_GPT_son_l1864_186497

noncomputable def my_age_in_years : ℕ := 84
noncomputable def total_age_in_years : ℕ := 140
noncomputable def months_in_a_year : ℕ := 12
noncomputable def weeks_in_a_year : ℕ := 52

theorem son's_age_in_weeks (G_d S_m G_m S_y : ℕ) (G_y : ℚ) :
  G_d = S_m →
  G_m = my_age_in_years * months_in_a_year →
  G_y = (G_m : ℚ) / months_in_a_year →
  G_y + S_y + my_age_in_years = total_age_in_years →
  S_y * weeks_in_a_year = 2548 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_son_l1864_186497


namespace NUMINAMATH_GPT_geom_seq_308th_term_l1864_186448

def geom_seq (a : ℤ) (r : ℤ) (n : ℕ) : ℤ :=
  a * r ^ n

-- Given conditions
def a := 10
def r := -1

theorem geom_seq_308th_term : geom_seq a r 307 = -10 := by
  sorry

end NUMINAMATH_GPT_geom_seq_308th_term_l1864_186448


namespace NUMINAMATH_GPT_wages_problem_l1864_186491

variable {S W_y W_x : ℝ}
variable {D_x : ℝ}

theorem wages_problem
  (h1 : S = 45 * W_y)
  (h2 : S = 20 * (W_x + W_y))
  (h3 : S = D_x * W_x) :
  D_x = 36 :=
sorry

end NUMINAMATH_GPT_wages_problem_l1864_186491


namespace NUMINAMATH_GPT_area_enclosed_by_trajectory_of_P_l1864_186483

-- Definitions of points
structure Point where
  x : ℝ
  y : ℝ

-- Definition of fixed points A and B
def A : Point := { x := -3, y := 0 }
def B : Point := { x := 3, y := 0 }

-- Condition for the ratio of distances
def ratio_condition (P : Point) : Prop :=
  ((P.x + 3)^2 + P.y^2) / ((P.x - 3)^2 + P.y^2) = 1 / 4

-- Definition of a circle based on the derived condition in the solution
def circle_eq (P : Point) : Prop :=
  (P.x + 5)^2 + P.y^2 = 16

-- Theorem stating the area enclosed by the trajectory of point P is 16π
theorem area_enclosed_by_trajectory_of_P : 
  (∀ P : Point, ratio_condition P → circle_eq P) →
  ∃ A : ℝ, A = 16 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_enclosed_by_trajectory_of_P_l1864_186483


namespace NUMINAMATH_GPT_common_ratio_of_increasing_geometric_sequence_l1864_186400

theorem common_ratio_of_increasing_geometric_sequence 
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : ∀ n, a (n + 1) = q * a n)
  (h_inc : ∀ n, a n < a (n + 1))
  (h_a2 : a 2 = 2)
  (h_a4_a3 : a 4 - a 3 = 4) : 
  q = 2 :=
by
  -- sorry - placeholder for proof
  sorry

end NUMINAMATH_GPT_common_ratio_of_increasing_geometric_sequence_l1864_186400


namespace NUMINAMATH_GPT_sequence_length_l1864_186414

theorem sequence_length :
  ∀ (n : ℕ), 
    (2 + 4 * (n - 1) = 2010) → n = 503 :=
by
    intro n
    intro h
    sorry

end NUMINAMATH_GPT_sequence_length_l1864_186414


namespace NUMINAMATH_GPT_bus_driver_earnings_l1864_186476

variables (rate : ℝ) (regular_hours overtime_hours : ℕ) (regular_rate overtime_rate : ℝ)

def calculate_regular_earnings (regular_rate : ℝ) (regular_hours : ℕ) : ℝ :=
  regular_rate * regular_hours

def calculate_overtime_earnings (overtime_rate : ℝ) (overtime_hours : ℕ) : ℝ :=
  overtime_rate * overtime_hours

def total_compensation (regular_rate overtime_rate : ℝ) (regular_hours overtime_hours : ℕ) : ℝ :=
  calculate_regular_earnings regular_rate regular_hours + calculate_overtime_earnings overtime_rate overtime_hours

theorem bus_driver_earnings :
  let regular_rate := 16
  let overtime_rate := regular_rate * 1.75
  let regular_hours := 40
  let total_hours := 44
  let overtime_hours := total_hours - regular_hours
  total_compensation regular_rate overtime_rate regular_hours overtime_hours = 752 :=
by
  sorry

end NUMINAMATH_GPT_bus_driver_earnings_l1864_186476


namespace NUMINAMATH_GPT_three_digit_number_l1864_186471

theorem three_digit_number (x y z : ℕ) 
  (h1: z^2 = x * y)
  (h2: y = (x + z) / 6)
  (h3: x - z = 4) :
  100 * x + 10 * y + z = 824 := 
by sorry

end NUMINAMATH_GPT_three_digit_number_l1864_186471


namespace NUMINAMATH_GPT_star_eq_122_l1864_186480

noncomputable def solveForStar (star : ℕ) : Prop :=
  45 - (28 - (37 - (15 - star))) = 56

theorem star_eq_122 : solveForStar 122 :=
by
  -- proof
  sorry

end NUMINAMATH_GPT_star_eq_122_l1864_186480


namespace NUMINAMATH_GPT_find_y_values_l1864_186463

theorem find_y_values (x : ℝ) (h1 : x^2 + 4 * ( (x + 1) / (x - 3) )^2 = 50)
  (y := ( (x - 3)^2 * (x + 4) ) / (2 * x - 4)) :
  y = -32 / 7 ∨ y = 2 :=
sorry

end NUMINAMATH_GPT_find_y_values_l1864_186463


namespace NUMINAMATH_GPT_triangle_equilateral_of_constraints_l1864_186410

theorem triangle_equilateral_of_constraints {a b c : ℝ}
  (h1 : a^4 = b^4 + c^4 - b^2 * c^2)
  (h2 : b^4 = c^4 + a^4 - a^2 * c^2) : 
  a = b ∧ b = c :=
by 
  sorry

end NUMINAMATH_GPT_triangle_equilateral_of_constraints_l1864_186410


namespace NUMINAMATH_GPT_gcd_fa_fb_l1864_186413

def f (x : ℤ) : ℤ := x * x - x + 2008

def a : ℤ := 102
def b : ℤ := 103

theorem gcd_fa_fb : Int.gcd (f a) (f b) = 2 := by
  sorry

end NUMINAMATH_GPT_gcd_fa_fb_l1864_186413


namespace NUMINAMATH_GPT_not_polynomial_option_B_l1864_186418

-- Definitions
def is_polynomial (expr : String) : Prop :=
  -- Assuming we have a function that determines if a given string expression is a polynomial.
  sorry

def option_A : String := "m+n"
def option_B : String := "x=1"
def option_C : String := "xy"
def option_D : String := "0"

-- Problem Statement
theorem not_polynomial_option_B : ¬ is_polynomial option_B := 
sorry

end NUMINAMATH_GPT_not_polynomial_option_B_l1864_186418


namespace NUMINAMATH_GPT_simplify_sqrt_l1864_186465

noncomputable def simplify_expression : ℝ :=
  Real.sqrt (8 + 6 * Real.sqrt 2) + Real.sqrt (8 - 6 * Real.sqrt 2)

theorem simplify_sqrt (h : simplify_expression = 2 * Real.sqrt 6) : 
    Real.sqrt (8 + 6 * Real.sqrt 2) + Real.sqrt (8 - 6 * Real.sqrt 2) = 2 * Real.sqrt 6 :=
  by sorry

end NUMINAMATH_GPT_simplify_sqrt_l1864_186465


namespace NUMINAMATH_GPT_perimeter_of_square_C_l1864_186447

theorem perimeter_of_square_C (a b : ℝ) 
  (hA : 4 * a = 16) 
  (hB : 4 * b = 32) : 
  4 * (a + b) = 48 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_square_C_l1864_186447


namespace NUMINAMATH_GPT_cube_minus_self_divisible_by_6_l1864_186450

theorem cube_minus_self_divisible_by_6 (n : ℕ) : 6 ∣ (n^3 - n) :=
sorry

end NUMINAMATH_GPT_cube_minus_self_divisible_by_6_l1864_186450


namespace NUMINAMATH_GPT_melissa_points_per_game_l1864_186457

theorem melissa_points_per_game (total_points : ℕ) (games_played : ℕ) (h1 : total_points = 1200) (h2 : games_played = 10) : (total_points / games_played) = 120 := 
by
  -- Here we would insert the proof steps, but we use sorry to represent the omission
  sorry

end NUMINAMATH_GPT_melissa_points_per_game_l1864_186457


namespace NUMINAMATH_GPT_total_money_l1864_186422

-- Define the variables A, B, and C as real numbers.
variables (A B C : ℝ)

-- Define the conditions as hypotheses.
def conditions : Prop :=
  A + C = 300 ∧ B + C = 150 ∧ C = 50

-- State the theorem to prove the total amount of money A, B, and C have.
theorem total_money (h : conditions A B C) : A + B + C = 400 :=
by {
  -- This proof is currently omitted.
  sorry
}

end NUMINAMATH_GPT_total_money_l1864_186422


namespace NUMINAMATH_GPT_range_of_m_l1864_186441

theorem range_of_m (m : ℝ) :
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ 2 ∧ 0 ≤ x2 ∧ x2 ≤ 2 → |((x2^2 - m * x2) - (x1^2 - m * x1))| ≤ 9) →
  -5 / 2 ≤ m ∧ m ≤ 13 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1864_186441


namespace NUMINAMATH_GPT_correct_overestimation_l1864_186462

theorem correct_overestimation (y : ℕ) : 
  25 * y + 4 * y = 29 * y := 
by 
  sorry

end NUMINAMATH_GPT_correct_overestimation_l1864_186462


namespace NUMINAMATH_GPT_same_solution_m_iff_m_eq_2_l1864_186493

theorem same_solution_m_iff_m_eq_2 (m y : ℝ) (h1 : my - 2 = 4) (h2 : y - 2 = 1) : m = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_same_solution_m_iff_m_eq_2_l1864_186493


namespace NUMINAMATH_GPT_expand_expression_l1864_186468

variable (x y : ℝ)

theorem expand_expression (x y : ℝ) : 12 * (3 * x + 4 - 2 * y) = 36 * x + 48 - 24 * y :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l1864_186468


namespace NUMINAMATH_GPT_problem1_problem2_l1864_186420

-- Problem 1: Proving the range of m values for the given inequality
theorem problem1 (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - m| ≥ 3) ↔ (m ≤ -4 ∨ m ≥ 2) :=
sorry

-- Problem 2: Proving the range of m values given a non-empty solution set for the inequality
theorem problem2 (m : ℝ) : (∃ x : ℝ, |m + 1| - 2 * m ≥ x^2 - x) ↔ (m ≤ 5/4) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1864_186420


namespace NUMINAMATH_GPT_no_outliers_in_dataset_l1864_186446

theorem no_outliers_in_dataset :
  let D := [7, 20, 34, 34, 40, 42, 42, 44, 52, 58]
  let Q1 := 34
  let Q3 := 44
  let IQR := Q3 - Q1
  let lower_threshold := Q1 - 1.5 * IQR
  let upper_threshold := Q3 + 1.5 * IQR
  (∀ x ∈ D, x ≥ lower_threshold) ∧ (∀ x ∈ D, x ≤ upper_threshold) →
  ∀ x ∈ D, ¬(x < lower_threshold ∨ x > upper_threshold) :=
by 
  sorry

end NUMINAMATH_GPT_no_outliers_in_dataset_l1864_186446


namespace NUMINAMATH_GPT_aquatic_reserve_total_fishes_l1864_186416

-- Define the number of bodies of water
def bodies_of_water : ℕ := 6

-- Define the number of fishes per body of water
def fishes_per_body : ℕ := 175

-- Define the total number of fishes
def total_fishes : ℕ := bodies_of_water * fishes_per_body

theorem aquatic_reserve_total_fishes : bodies_of_water * fishes_per_body = 1050 := by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_aquatic_reserve_total_fishes_l1864_186416


namespace NUMINAMATH_GPT_find_a_l1864_186472

theorem find_a (a : ℝ) (h : (1 / Real.log 2 / Real.log a) + (1 / Real.log 3 / Real.log a) + (1 / Real.log 5 / Real.log a) = 2) : a = Real.sqrt 30 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_l1864_186472


namespace NUMINAMATH_GPT_relationship_between_M_and_N_l1864_186403

variable (a : ℝ)

def M : ℝ := 2 * a * (a - 2) + 4
def N : ℝ := (a - 1) * (a - 3)

theorem relationship_between_M_and_N : M a > N a :=
by sorry

end NUMINAMATH_GPT_relationship_between_M_and_N_l1864_186403


namespace NUMINAMATH_GPT_workers_problem_l1864_186486

theorem workers_problem (W : ℕ) (A : ℕ) :
  (W * 45 = A) ∧ ((W + 10) * 35 = A) → W = 35 :=
by
  sorry

end NUMINAMATH_GPT_workers_problem_l1864_186486


namespace NUMINAMATH_GPT_cost_per_night_l1864_186428

variable (x : ℕ)

theorem cost_per_night (h : 3 * x - 100 = 650) : x = 250 :=
sorry

end NUMINAMATH_GPT_cost_per_night_l1864_186428


namespace NUMINAMATH_GPT_number_of_members_l1864_186481

theorem number_of_members
  (headband_cost : ℕ := 3)
  (jersey_cost : ℕ := 10)
  (total_cost : ℕ := 2700)
  (cost_per_member : ℕ := 26) :
  total_cost / cost_per_member = 103 := by
  sorry

end NUMINAMATH_GPT_number_of_members_l1864_186481


namespace NUMINAMATH_GPT_last_recess_break_duration_l1864_186498

-- Definitions based on the conditions
def first_recess_break : ℕ := 15
def second_recess_break : ℕ := 15
def lunch_break : ℕ := 30
def total_outside_class_time : ℕ := 80

-- The theorem we need to prove
theorem last_recess_break_duration :
  total_outside_class_time = first_recess_break + second_recess_break + lunch_break + 20 :=
sorry

end NUMINAMATH_GPT_last_recess_break_duration_l1864_186498


namespace NUMINAMATH_GPT_find_number_l1864_186488

theorem find_number :
  ∃ n : ℕ, n * (1 / 7)^2 = 7^3 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1864_186488


namespace NUMINAMATH_GPT_problem_l1864_186470

theorem problem 
  (x : ℝ) 
  (h1 : x ∈ Set.Icc (-3 : ℝ) 3) 
  (h2 : x ≠ -5/3) : 
  (4 * x ^ 2 + 2) / (5 + 3 * x) ≥ 1 ↔ x ∈ (Set.Icc (-3) (-3/4) ∪ Set.Icc 1 3) :=
sorry

end NUMINAMATH_GPT_problem_l1864_186470


namespace NUMINAMATH_GPT_find_m_l1864_186475

theorem find_m (m : ℝ) (h : 2 / m = (m + 1) / 3) : m = -3 := by
  sorry

end NUMINAMATH_GPT_find_m_l1864_186475


namespace NUMINAMATH_GPT_max_value_of_expression_l1864_186469

open Real

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  a + sqrt (a * b) + (a * b * c) ^ (1 / 4) ≤ 10 / 3 := sorry

end NUMINAMATH_GPT_max_value_of_expression_l1864_186469


namespace NUMINAMATH_GPT_sum_of_cubes_divisible_by_middle_integer_l1864_186453

theorem sum_of_cubes_divisible_by_middle_integer (a : ℤ) : 
  (a - 1)^3 + a^3 + (a + 1)^3 ∣ 3 * a :=
sorry

end NUMINAMATH_GPT_sum_of_cubes_divisible_by_middle_integer_l1864_186453


namespace NUMINAMATH_GPT_unit_prices_max_books_l1864_186444

-- Definitions based on conditions 1 and 2
def unit_price_A (x : ℝ) : Prop :=
  x > 5 ∧ (1200 / x = 900 / (x - 5))

-- Definitions based on conditions 3, 4, and 5
def max_books_A (y : ℝ) : Prop :=
  0 ≤ y ∧ y ≤ 300 ∧ 0.9 * 20 * y + 15 * (300 - y) ≤ 5100

theorem unit_prices
  (x : ℝ)
  (h : unit_price_A x) :
  x = 20 ∧ x - 5 = 15 :=
sorry

theorem max_books
  (y : ℝ)
  (hy : max_books_A y) :
  y ≤ 200 :=
sorry

end NUMINAMATH_GPT_unit_prices_max_books_l1864_186444


namespace NUMINAMATH_GPT_constant_difference_of_equal_derivatives_l1864_186404

theorem constant_difference_of_equal_derivatives
  {f g : ℝ → ℝ}
  (h : ∀ x, deriv f x = deriv g x) :
  ∃ C : ℝ, ∀ x, f x - g x = C := 
sorry

end NUMINAMATH_GPT_constant_difference_of_equal_derivatives_l1864_186404


namespace NUMINAMATH_GPT_max_distinct_tangent_counts_l1864_186494

-- Define the types and conditions for our circles and tangents
structure Circle where
  radius : ℝ

def circle1 : Circle := { radius := 3 }
def circle2 : Circle := { radius := 4 }

-- Define the statement to be proved
theorem max_distinct_tangent_counts :
  ∃ (k : ℕ), k = 5 :=
sorry

end NUMINAMATH_GPT_max_distinct_tangent_counts_l1864_186494


namespace NUMINAMATH_GPT_brooke_kent_ratio_l1864_186455

theorem brooke_kent_ratio :
  ∀ (alison brooke brittany kent : ℕ),
  (kent = 1000) →
  (alison = 4000) →
  (alison = brittany / 2) →
  (brittany = 4 * brooke) →
  brooke / kent = 2 :=
by
  intros alison brooke brittany kent kent_val alison_val alison_brittany brittany_brooke
  sorry

end NUMINAMATH_GPT_brooke_kent_ratio_l1864_186455


namespace NUMINAMATH_GPT_larger_number_l1864_186419

theorem larger_number (x y : ℕ) (h1 : x + y = 40) (h2 : x - y = 4) : x = 22 := by
  sorry

end NUMINAMATH_GPT_larger_number_l1864_186419


namespace NUMINAMATH_GPT_correct_transformation_l1864_186460

theorem correct_transformation (a b c : ℝ) (h : c ≠ 0) (h1 : a / c = b / c) : a = b :=
by 
  -- Actual proof would go here, but we use sorry for the scaffold.
  sorry

end NUMINAMATH_GPT_correct_transformation_l1864_186460


namespace NUMINAMATH_GPT_total_wire_length_l1864_186412

theorem total_wire_length (S : ℕ) (L : ℕ)
  (hS : S = 20) 
  (hL : L = 2 * S) : S + L = 60 :=
by
  sorry

end NUMINAMATH_GPT_total_wire_length_l1864_186412


namespace NUMINAMATH_GPT_minimize_sum_of_squares_of_roots_l1864_186437

theorem minimize_sum_of_squares_of_roots (m : ℝ) (h : 100 - 20 * m ≥ 0) :
  (∀ a b : ℝ, (∀ x : ℝ, 5 * x^2 - 10 * x + m = 0 → x = a ∨ x = b) → (4 - 2 * m / 5) ≥ (4 - 2 * 5 / 5)) :=
by
  sorry

end NUMINAMATH_GPT_minimize_sum_of_squares_of_roots_l1864_186437


namespace NUMINAMATH_GPT_polygon_parallel_edges_l1864_186438

theorem polygon_parallel_edges (n : ℕ) (h : n > 2) :
  (∃ i j, i ≠ j ∧ (i + 1) % n = (j + 1) % n) ↔ (∃ k, n = 2 * k) :=
  sorry

end NUMINAMATH_GPT_polygon_parallel_edges_l1864_186438


namespace NUMINAMATH_GPT_constructible_iff_multiple_of_8_l1864_186458

def is_constructible_with_L_tetromino (m n : ℕ) : Prop :=
  ∃ (k : ℕ), 4 * k = m * n

theorem constructible_iff_multiple_of_8 (m n : ℕ) (hm : 1 < m) (hn : 1 < n) : 
  is_constructible_with_L_tetromino m n ↔ 8 ∣ m * n :=
sorry

end NUMINAMATH_GPT_constructible_iff_multiple_of_8_l1864_186458


namespace NUMINAMATH_GPT_alcohol_to_water_ratio_l1864_186402

theorem alcohol_to_water_ratio (alcohol water : ℚ) (h_alcohol : alcohol = 2/7) (h_water : water = 3/7) : alcohol / water = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_alcohol_to_water_ratio_l1864_186402


namespace NUMINAMATH_GPT_inf_div_p_n2n_plus_one_n_div_3_n2n_plus_one_l1864_186409

theorem inf_div_p_n2n_plus_one (p : ℕ) (hp : Nat.Prime p) (h_odd : p % 2 = 1) :
  ∃ᶠ n in at_top, p ∣ (n * 2^n + 1) :=
sorry

theorem n_div_3_n2n_plus_one :
  (∃ k : ℕ, ∀ n, n = 6 * k + 1 ∨ n = 6 * k + 2 → 3 ∣ (n * 2^n + 1)) :=
sorry

end NUMINAMATH_GPT_inf_div_p_n2n_plus_one_n_div_3_n2n_plus_one_l1864_186409


namespace NUMINAMATH_GPT_part1_part2_l1864_186459

noncomputable def f (x : ℝ) : ℝ := |3 * x + 2|

theorem part1 (x : ℝ): f x < 6 - |x - 2| ↔ (-3/2 < x ∧ x < 1) :=
by sorry

theorem part2 (a : ℝ) (m n : ℝ) (h₁ : 0 < m) (h₂ : 0 < n) (h₃ : m + n = 4) (h₄ : 0 < a) (h₅ : ∀ x, |x - a| - f x ≤ 1/m + 1/n) :
    0 < a ∧ a ≤ 1/3 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1864_186459


namespace NUMINAMATH_GPT_suraj_average_l1864_186417

theorem suraj_average : 
  ∀ (A : ℝ), 
    (16 * A + 92 = 17 * (A + 4)) → 
      (A + 4) = 28 :=
by
  sorry

end NUMINAMATH_GPT_suraj_average_l1864_186417


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1864_186485

theorem sufficient_but_not_necessary (a : ℝ) : (a = 2 → (a - 1) * (a - 2) = 0) ∧ 
  (∀ x : ℝ, (x - 1) * (x - 2) = 0 → x ≠ 2 → x = 1) ∧
  (a = 2 → (1 ≠ 2)) :=
by {
  sorry
}

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1864_186485


namespace NUMINAMATH_GPT_find_m_l1864_186452

theorem find_m 
  (m : ℝ)
  (h_pos : 0 < m)
  (asymptote_twice_angle : ∃ l : ℝ, l = 3 ∧ (x - l * y = 0 ∧ m * x^2 - y^2 = m)) :
  m = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1864_186452


namespace NUMINAMATH_GPT_compare_triangle_operations_l1864_186499

def tri_op (a b : ℤ) : ℤ := a * b - a - b + 1

theorem compare_triangle_operations : tri_op (-3) 4 = tri_op 4 (-3) :=
by
  unfold tri_op
  sorry

end NUMINAMATH_GPT_compare_triangle_operations_l1864_186499


namespace NUMINAMATH_GPT_original_faculty_is_287_l1864_186474

noncomputable def original_faculty (F : ℝ) : Prop :=
  (F * 0.85 * 0.80 = 195)

theorem original_faculty_is_287 : ∃ F : ℝ, original_faculty F ∧ F = 287 := 
by 
  use 287
  sorry

end NUMINAMATH_GPT_original_faculty_is_287_l1864_186474


namespace NUMINAMATH_GPT_arithmetic_sqrt_of_sqrt_16_l1864_186484

theorem arithmetic_sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sqrt_of_sqrt_16_l1864_186484


namespace NUMINAMATH_GPT_gina_can_paint_6_rose_cups_an_hour_l1864_186490

def number_of_rose_cups_painted_in_an_hour 
  (R : ℕ) (lily_rate : ℕ) (rose_order : ℕ) (lily_order : ℕ) (total_payment : ℕ) (hourly_rate : ℕ)
  (lily_hours : ℕ) (total_hours : ℕ) (rose_hours : ℕ) : Prop :=
  (lily_rate = 7) ∧
  (rose_order = 6) ∧
  (lily_order = 14) ∧
  (total_payment = 90) ∧
  (hourly_rate = 30) ∧
  (lily_hours = lily_order / lily_rate) ∧
  (total_hours = total_payment / hourly_rate) ∧
  (rose_hours = total_hours - lily_hours) ∧
  (rose_order = R * rose_hours)

theorem gina_can_paint_6_rose_cups_an_hour :
  ∃ R, number_of_rose_cups_painted_in_an_hour 
    R 7 6 14 90 30 (14 / 7) (90 / 30)  (90 / 30 - 14 / 7) ∧ R = 6 :=
by
  -- proof is left out intentionally
  sorry

end NUMINAMATH_GPT_gina_can_paint_6_rose_cups_an_hour_l1864_186490


namespace NUMINAMATH_GPT_product_of_3_point_6_and_0_point_25_l1864_186429

theorem product_of_3_point_6_and_0_point_25 : 3.6 * 0.25 = 0.9 := 
by 
  sorry

end NUMINAMATH_GPT_product_of_3_point_6_and_0_point_25_l1864_186429


namespace NUMINAMATH_GPT_m_plus_n_l1864_186478

theorem m_plus_n (m n : ℕ) (hm : 0 < m) (hn : 1 < n) (h : m ^ n = 2^25 * 3^40) : m + n = 209957 :=
  sorry

end NUMINAMATH_GPT_m_plus_n_l1864_186478


namespace NUMINAMATH_GPT_sin_cos_value_l1864_186492

-- Given function definition
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x^2 + (Real.sin α - 2 * Real.cos α) * x + 1

-- Definitions and proof problem statement
theorem sin_cos_value (α : ℝ) : 
  (∀ x : ℝ, f α x = f α (-x)) → (Real.sin α * Real.cos α = 2 / 5) :=
by
  intro h_even
  sorry

end NUMINAMATH_GPT_sin_cos_value_l1864_186492


namespace NUMINAMATH_GPT_total_students_l1864_186442

theorem total_students (p q r s : ℕ) 
  (h1 : 1 < p)
  (h2 : p < q)
  (h3 : q < r)
  (h4 : r < s)
  (h5 : p * q * r * s = 1365) :
  p + q + r + s = 28 :=
sorry

end NUMINAMATH_GPT_total_students_l1864_186442


namespace NUMINAMATH_GPT_men_in_first_group_l1864_186443

variable (M : ℕ) (daily_wage : ℝ)
variable (h1 : M * 10 * daily_wage = 1200)
variable (h2 : 9 * 6 * daily_wage = 1620)
variable (dw_eq : daily_wage = 30)

theorem men_in_first_group : M = 4 :=
by sorry

end NUMINAMATH_GPT_men_in_first_group_l1864_186443


namespace NUMINAMATH_GPT_solve_tan_equation_l1864_186421

theorem solve_tan_equation (x : ℝ) (k : ℤ) :
  8.456 * (Real.tan x)^2 * (Real.tan (3 * x))^2 * Real.tan (4 * x) = 
  (Real.tan x)^2 - (Real.tan (3 * x))^2 + Real.tan (4 * x) ->
  x = π * k ∨ x = π / 4 * (2 * k + 1) := sorry

end NUMINAMATH_GPT_solve_tan_equation_l1864_186421


namespace NUMINAMATH_GPT_store_revenue_after_sale_l1864_186430

/--
A store has 2000 items, each normally selling for $50. 
They offer an 80% discount and manage to sell 90% of the items. 
The store owes $15,000 to creditors. Prove that the store has $3,000 left after the sale.
-/
theorem store_revenue_after_sale :
  let items := 2000
  let retail_price := 50
  let discount := 0.8
  let sale_percentage := 0.9
  let debt := 15000
  let items_sold := items * sale_percentage
  let discount_amount := retail_price * discount
  let sale_price_per_item := retail_price - discount_amount
  let total_revenue := items_sold * sale_price_per_item
  let money_left := total_revenue - debt
  money_left = 3000 :=
by
  sorry

end NUMINAMATH_GPT_store_revenue_after_sale_l1864_186430


namespace NUMINAMATH_GPT_boat_travels_125_km_downstream_l1864_186489

/-- The speed of the boat in still water is 20 km/hr -/
def boat_speed_still_water : ℝ := 20

/-- The speed of the stream is 5 km/hr -/
def stream_speed : ℝ := 5

/-- The total time taken downstream is 5 hours -/
def total_time_downstream : ℝ := 5

/-- The effective speed of the boat downstream -/
def effective_speed_downstream : ℝ := boat_speed_still_water + stream_speed

/-- The distance the boat travels downstream -/
def distance_downstream : ℝ := effective_speed_downstream * total_time_downstream

/-- The boat travels 125 km downstream -/
theorem boat_travels_125_km_downstream :
  distance_downstream = 125 := 
sorry

end NUMINAMATH_GPT_boat_travels_125_km_downstream_l1864_186489


namespace NUMINAMATH_GPT_evaluate_expression_l1864_186482

theorem evaluate_expression : (-3)^4 / 3^2 - 2^5 + 7^2 = 26 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1864_186482


namespace NUMINAMATH_GPT_triangle_area_l1864_186427

theorem triangle_area {a c : ℝ} (h_a : a = 3 * Real.sqrt 3) (h_c : c = 2) (angle_B : ℝ) (h_B : angle_B = Real.pi / 3) : 
  (1 / 2) * a * c * Real.sin angle_B = 9 / 2 :=
by
  rw [h_a, h_c, h_B]
  sorry

end NUMINAMATH_GPT_triangle_area_l1864_186427
