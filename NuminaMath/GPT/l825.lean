import Mathlib

namespace value_of_x_minus_y_squared_l825_82515

theorem value_of_x_minus_y_squared (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 6) : (x - y) ^ 2 = 1 :=
by
  sorry

end value_of_x_minus_y_squared_l825_82515


namespace cost_of_candy_car_l825_82521

theorem cost_of_candy_car (starting_amount paid_amount change : ℝ) (h1 : starting_amount = 1.80) (h2 : change = 1.35) (h3 : paid_amount = starting_amount - change) : paid_amount = 0.45 := by
  sorry

end cost_of_candy_car_l825_82521


namespace paige_science_problems_l825_82577

variable (S : ℤ)

theorem paige_science_problems (h1 : 43 + S - 44 = 11) : S = 12 :=
by
  sorry

end paige_science_problems_l825_82577


namespace farmer_trees_l825_82505

theorem farmer_trees (x n m : ℕ) 
  (h1 : x + 20 = n^2) 
  (h2 : x - 39 = m^2) : 
  x = 880 := 
by sorry

end farmer_trees_l825_82505


namespace find_k_l825_82592

theorem find_k : ∃ b k : ℝ, (∀ x : ℝ, (x + b)^2 = x^2 - 20 * x + k) ∧ k = 100 := by
  sorry

end find_k_l825_82592


namespace seats_empty_l825_82565

def number_of_people : ℕ := 532
def total_seats : ℕ := 750

theorem seats_empty (n : ℕ) (m : ℕ) : m - n = 218 := by
  have number_of_people : ℕ := 532
  have total_seats : ℕ := 750
  sorry

end seats_empty_l825_82565


namespace polar_coordinates_of_point_l825_82517

noncomputable def point_rectangular_to_polar (x y : ℝ) : ℝ × ℝ := 
  let r := Real.sqrt (x^2 + y^2)
  let θ := if y < 0 then 2 * Real.pi + Real.arctan (y / x) else Real.arctan (y / x)
  (r, θ)

theorem polar_coordinates_of_point :
  point_rectangular_to_polar 1 (-1) = (Real.sqrt 2, 7 * Real.pi / 4) :=
by
  unfold point_rectangular_to_polar
  sorry

end polar_coordinates_of_point_l825_82517


namespace no_prime_solution_l825_82560

theorem no_prime_solution (p : ℕ) (h_prime : Nat.Prime p) : ¬(2^p + p ∣ 3^p + p) := by
  sorry

end no_prime_solution_l825_82560


namespace problem1_part1_problem1_part2_problem2_l825_82593

noncomputable def problem1_condition1 (m : ℕ) (a : ℕ) : Prop := 4^m = a
noncomputable def problem1_condition2 (n : ℕ) (b : ℕ) : Prop := 8^n = b

theorem problem1_part1 (m n a b : ℕ) (h1 : 4^m = a) (h2 : 8^n = b) : 2^(2*m + 3*n) = a * b :=
by sorry

theorem problem1_part2 (m n a b : ℕ) (h1 : 4^m = a) (h2 : 8^n = b) : 2^(4*m - 6*n) = (a^2) / (b^2) :=
by sorry

theorem problem2 (x : ℕ) (h : 2 * 8^x * 16 = 2^23) : x = 6 :=
by sorry

end problem1_part1_problem1_part2_problem2_l825_82593


namespace inscribed_sphere_radius_l825_82584

theorem inscribed_sphere_radius (h1 h2 h3 h4 : ℝ) (S1 S2 S3 S4 V : ℝ)
  (h1_ge : h1 ≥ 1) (h2_ge : h2 ≥ 1) (h3_ge : h3 ≥ 1) (h4_ge : h4 ≥ 1)
  (volume : V = (1/3) * S1 * h1)
  : (∃ r : ℝ, 3 * V = (S1 + S2 + S3 + S4) * r ∧ r = 1 / 4) :=
by
  sorry

end inscribed_sphere_radius_l825_82584


namespace cookies_per_batch_l825_82542

-- Define the necessary conditions
def total_chips : ℕ := 81
def batches : ℕ := 3
def chips_per_cookie : ℕ := 9

-- Theorem stating the number of cookies per batch
theorem cookies_per_batch : (total_chips / batches) / chips_per_cookie = 3 :=
by
  -- Here would be the proof, but we use sorry as placeholder
  sorry

end cookies_per_batch_l825_82542


namespace factorial_power_of_two_l825_82571

theorem factorial_power_of_two solutions (a b c : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_equation : a.factorial + b.factorial = 2^(c.factorial)) :
  solutions = [(1, 1, 1), (2, 2, 2)] :=
sorry

end factorial_power_of_two_l825_82571


namespace meteorite_weight_possibilities_l825_82527

def valid_meteorite_weight_combinations : ℕ :=
  (2 * (Nat.factorial 5 / (Nat.factorial 2 * Nat.factorial 2))) + (Nat.factorial 5)

theorem meteorite_weight_possibilities :
  valid_meteorite_weight_combinations = 180 :=
by
  -- Sorry added to skip the proof.
  sorry

end meteorite_weight_possibilities_l825_82527


namespace factor_expression_l825_82536

theorem factor_expression (a b c : ℝ) :
  a^3 * (b^3 - c^3) + b^3 * (c^3 - a^3) + c^3 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + ab + bc + ca) :=
by
  sorry

end factor_expression_l825_82536


namespace evaluate_expression_l825_82509

theorem evaluate_expression : (733 * 733) - (732 * 734) = 1 :=
by
  sorry

end evaluate_expression_l825_82509


namespace fishermen_total_catch_l825_82551

noncomputable def m : ℕ := 30  -- Mike can catch 30 fish per hour
noncomputable def j : ℕ := 2 * m  -- Jim can catch twice as much as Mike
noncomputable def b : ℕ := j + (j / 2)  -- Bob can catch 50% more than Jim

noncomputable def fish_caught_in_40_minutes : ℕ := (2 * m) / 3 -- Fishermen fish together for 40 minutes (2/3 hour)
noncomputable def fish_caught_by_jim_in_remaining_time : ℕ := j / 3 -- Jim fishes alone for the remaining 20 minutes (1/3 hour)

noncomputable def total_fish_caught : ℕ :=
  fish_caught_in_40_minutes * 3 + fish_caught_by_jim_in_remaining_time

theorem fishermen_total_catch : total_fish_caught = 140 := by
  sorry

end fishermen_total_catch_l825_82551


namespace input_command_is_INPUT_l825_82516

-- Define the commands
def PRINT : String := "PRINT"
def INPUT : String := "INPUT"
def THEN : String := "THEN"
def END : String := "END"

-- Define the properties of each command
def PRINT_is_output (cmd : String) : Prop :=
  cmd = PRINT

def INPUT_is_input (cmd : String) : Prop :=
  cmd = INPUT

def THEN_is_conditional (cmd : String) : Prop :=
  cmd = THEN

def END_is_end (cmd : String) : Prop :=
  cmd = END

-- Theorem stating that INPUT is the command associated with input operation
theorem input_command_is_INPUT : INPUT_is_input INPUT :=
by
  -- Proof goes here
  sorry

end input_command_is_INPUT_l825_82516


namespace largest_common_factor_462_330_l825_82591

-- Define the factors of 462
def factors_462 : Set ℕ := {1, 2, 3, 6, 7, 14, 21, 33, 42, 66, 77, 154, 231, 462}

-- Define the factors of 330
def factors_330 : Set ℕ := {1, 2, 3, 5, 6, 10, 11, 15, 30, 33, 55, 66, 110, 165, 330}

-- Define the statement of the theorem
theorem largest_common_factor_462_330 : 
  (∀ d : ℕ, d ∈ (factors_462 ∩ factors_330) → d ≤ 66) ∧
  66 ∈ (factors_462 ∩ factors_330) :=
sorry

end largest_common_factor_462_330_l825_82591


namespace class_gpa_l825_82548

theorem class_gpa (n : ℕ) (h1 : (n / 3) * 60 + (2 * (n / 3)) * 66 = total_gpa) :
  total_gpa / n = 64 :=
by
  sorry

end class_gpa_l825_82548


namespace parallel_lines_l825_82579

theorem parallel_lines (a : ℝ) : (∀ x y : ℝ, (a-1) * x + 2 * y + 10 = 0) → (∀ x y : ℝ, x + a * y + 3 = 0) → (a = -1 ∨ a = 2) :=
sorry

end parallel_lines_l825_82579


namespace additional_grassy_ground_l825_82507

theorem additional_grassy_ground (r₁ r₂ : ℝ) (π : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 23) :
  π * r₂ ^ 2 - π * r₁ ^ 2 = 385 * π :=
  by
  subst h₁ h₂
  sorry

end additional_grassy_ground_l825_82507


namespace parity_related_to_phi_not_omega_l825_82523

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem parity_related_to_phi_not_omega (ω : ℝ) (φ : ℝ) (h : 0 < ω) :
  (∃ k : ℤ, φ = k * Real.pi → ∀ x : ℝ, f ω φ (-x) = -f ω φ x) ∧
  (∃ k : ℤ, φ = k * Real.pi + Real.pi / 2 → ∀ x : ℝ, f ω φ (-x) = f ω φ x) :=
sorry

end parity_related_to_phi_not_omega_l825_82523


namespace inequality_squares_l825_82534

theorem inequality_squares (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h : a + b + c = 1) :
    (3 / 16) ≤ ( (a / (1 + a))^2 + (b / (1 + b))^2 + (c / (1 + c))^2 ) ∧
    ( (a / (1 + a))^2 + (b / (1 + b))^2 + (c / (1 + c))^2 ) ≤ 1 / 4 :=
by
  sorry

end inequality_squares_l825_82534


namespace system_of_equations_solution_l825_82596

theorem system_of_equations_solution (x y z : ℤ) :
  x^2 - 9 * y^2 - z^2 = 0 ∧ z = x - 3 * y ↔ 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (∃ k : ℤ, x = 3 * k ∧ y = k ∧ z = 0) := 
by
  sorry

end system_of_equations_solution_l825_82596


namespace base_for_four_digit_even_l825_82576

theorem base_for_four_digit_even (b : ℕ) : b^3 ≤ 346 ∧ 346 < b^4 ∧ (346 % b) % 2 = 0 → b = 6 :=
by
  sorry

end base_for_four_digit_even_l825_82576


namespace geometric_sequence_sum_l825_82564

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : a 2 = 1 - a 1)
  (h3 : a 4 = 9 - a 3)
  (h4 : ∀ n, a (n + 1) = a n * q) :
  a 4 + a 5 = 27 :=
sorry

end geometric_sequence_sum_l825_82564


namespace points_on_opposite_sides_of_line_l825_82599

theorem points_on_opposite_sides_of_line (a : ℝ) :
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 ↔ -7 < a ∧ a < 24 :=
by 
  sorry

end points_on_opposite_sides_of_line_l825_82599


namespace difference_of_squares_l825_82574

-- Define the variables x and y as real numbers
variables (x y : ℝ)

-- Define the condition for the expression which should hold
def expression_b := (2 * x + y) * (y - 2 * x)

-- The theorem to prove that this expression fits the formula for the difference of squares
theorem difference_of_squares : 
  ∃ a b : ℝ, expression_b x y = a^2 - b^2 := 
by 
  sorry

end difference_of_squares_l825_82574


namespace evaluate_f_x_plus_3_l825_82598

def f (x : ℝ) : ℝ := x^2

theorem evaluate_f_x_plus_3 (x : ℝ) : f (x + 3) = x^2 + 6 * x + 9 := by
  sorry

end evaluate_f_x_plus_3_l825_82598


namespace john_bike_speed_l825_82567

noncomputable def average_speed_for_bike_ride (swim_distance swim_speed run_distance run_speed bike_distance total_time : ℕ) := 
  let swim_time := swim_distance / swim_speed
  let run_time := run_distance / run_speed
  let remaining_time := total_time - (swim_time + run_time)
  bike_distance / remaining_time

theorem john_bike_speed : average_speed_for_bike_ride 1 5 8 12 (3 / 2) = 18 := by
  sorry

end john_bike_speed_l825_82567


namespace largest_side_of_rectangle_l825_82541

theorem largest_side_of_rectangle (l w : ℝ) 
    (h1 : 2 * l + 2 * w = 240) 
    (h2 : l * w = 1920) : 
    max l w = 101 := 
sorry

end largest_side_of_rectangle_l825_82541


namespace original_cube_volume_l825_82546

theorem original_cube_volume (a : ℕ) (V_cube V_new : ℕ)
  (h1 : V_cube = a^3)
  (h2 : V_new = (a + 2) * a * (a - 2))
  (h3 : V_cube = V_new + 24) :
  V_cube = 216 :=
by
  sorry

end original_cube_volume_l825_82546


namespace shaded_area_of_square_l825_82545

theorem shaded_area_of_square (side_square : ℝ) (leg_triangle : ℝ) (h1 : side_square = 40) (h2 : leg_triangle = 25) :
  let area_square := side_square ^ 2
  let area_triangle := (1 / 2) * leg_triangle * leg_triangle
  let total_area_triangles := 2 * area_triangle
  let shaded_area := area_square - total_area_triangles
  shaded_area = 975 :=
by
  sorry

end shaded_area_of_square_l825_82545


namespace correct_choice_l825_82583

variable (a b : ℝ) (p q : Prop) (x : ℝ)

-- Proposition A: Incorrect because x > 3 is a sufficient condition for x > 2.
def propositionA : Prop := (∀ x : ℝ, x > 3 → x > 2) ∧ ¬ (∀ x : ℝ, x > 2 → x > 3)

-- Proposition B: Incorrect negation form.
def propositionB : Prop := ¬ (¬p → ¬q) ∧ (q → p)

-- Proposition C: Incorrect because it should be 1/a > 1/b given 0 < a < b.
def propositionC : Prop := (a > 0 ∧ b < 0) ∧ ¬ (1/a < 1/b)

-- Proposition D: Correct negation form.
def propositionD_negation_correct : Prop := 
  (¬ ∃ x : ℝ, x^2 = 1) = ( ∀ x : ℝ, x^2 ≠ 1)

theorem correct_choice : propositionD_negation_correct := by
  sorry

end correct_choice_l825_82583


namespace heidi_and_karl_painting_l825_82510

-- Given conditions
def heidi_paint_rate := 1 / 60 -- Rate at which Heidi paints, in walls per minute
def karl_paint_rate := 2 * heidi_paint_rate -- Rate at which Karl paints, in walls per minute
def painting_time := 20 -- Time spent painting, in minutes

-- Prove the amount of each wall painted
theorem heidi_and_karl_painting :
  (heidi_paint_rate * painting_time = 1 / 3) ∧ (karl_paint_rate * painting_time = 2 / 3) :=
sorry

end heidi_and_karl_painting_l825_82510


namespace solve_system_of_equations_l825_82511

theorem solve_system_of_equations :
  ∃ (x y z : ℝ), 
    (2 * y + x - x^2 - y^2 = 0) ∧ 
    (z - x + y - y * (x + z) = 0) ∧ 
    (-2 * y + z - y^2 - z^2 = 0) ∧ 
    ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 0 ∧ z = 1)) :=
by
  sorry

end solve_system_of_equations_l825_82511


namespace find_XY_squared_l825_82595

variables {A B C T X Y : Type}

-- Conditions
variables (is_acute_scalene_triangle : ∀ A B C : Type, Prop) -- Assume scalene and acute properties
variable  (circumcircle : ∀ A B C : Type, Type) -- Circumcircle of the triangle
variable  (tangent_at : ∀ (ω : Type) B C, Type) -- Tangents at B and C
variables (BT CT : ℝ)
variables (BC : ℝ)
variables (projections : ∀ T (line : Type), Type)
variables (TX TY XY : ℝ)

-- Given conditions
axiom BT_value : BT = 18
axiom CT_value : CT = 18
axiom BC_value : BC = 24
axiom final_equation : TX^2 + TY^2 + XY^2 = 1552

-- Goal
theorem find_XY_squared : XY^2 = 884 := by
  sorry

end find_XY_squared_l825_82595


namespace bothStoresSaleSameDate_l825_82556

-- Define the conditions
def isBookstoreSaleDay (d : ℕ) : Prop := d % 4 = 0
def isShoeStoreSaleDay (d : ℕ) : Prop := ∃ k : ℕ, d = 5 + 7 * k
def isJulyDay (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 31

-- Define the problem statement
theorem bothStoresSaleSameDate : 
  (∃ d1 d2 : ℕ, isJulyDay d1 ∧ isBookstoreSaleDay d1 ∧ isShoeStoreSaleDay d1 ∧
                 isJulyDay d2 ∧ isBookstoreSaleDay d2 ∧ isShoeStoreSaleDay d2 ∧ d1 ≠ d2) :=
sorry

end bothStoresSaleSameDate_l825_82556


namespace intersect_parabolas_l825_82524

theorem intersect_parabolas :
  ∀ (x y : ℝ),
    ((y = 2 * x^2 - 7 * x + 1 ∧ y = 8 * x^2 + 5 * x + 1) ↔ 
     ((x = -2 ∧ y = 23) ∨ (x = 0 ∧ y = 1))) :=
by sorry

end intersect_parabolas_l825_82524


namespace sqrt_of_mixed_number_l825_82520

theorem sqrt_of_mixed_number :
  (Real.sqrt (8 + 9 / 16)) = (Real.sqrt 137 / 4) :=
by
  sorry

end sqrt_of_mixed_number_l825_82520


namespace calc_expr_l825_82570

theorem calc_expr : 3000 * (3000 ^ 3000) + 3000 ^ 2 = 3000 ^ 3001 :=
by sorry

end calc_expr_l825_82570


namespace freshman_to_sophomore_ratio_l825_82588

variable (f s : ℕ)

-- Define the participants from freshmen and sophomores
def freshmen_participants : ℕ := (3 * f) / 7
def sophomores_participants : ℕ := (2 * s) / 3

-- Theorem: There are 14/9 times as many freshmen as sophomores
theorem freshman_to_sophomore_ratio (h : freshmen_participants f = sophomores_participants s) : 
  9 * f = 14 * s :=
by
  sorry

end freshman_to_sophomore_ratio_l825_82588


namespace punctures_covered_l825_82568

theorem punctures_covered (P1 P2 P3 : ℝ) (h1 : 0 ≤ P1) (h2 : P1 < P2) (h3 : P2 < P3) (h4 : P3 < 3) :
    ∃ x, x ≤ P1 ∧ x + 2 ≥ P3 := 
sorry

end punctures_covered_l825_82568


namespace rhombus_condition_perimeter_rhombus_given_ab_l825_82544

noncomputable def roots_of_quadratic (m : ℝ) : Set ℝ :=
{ x : ℝ | x^2 - m * x + m / 2 - 1 / 4 = 0 }

theorem rhombus_condition (m : ℝ) : 
  (∃ ab ad : ℝ, ab ∈ roots_of_quadratic m ∧ ad ∈ roots_of_quadratic m ∧ ab = ad) ↔ m = 1 :=
by
  sorry

theorem perimeter_rhombus_given_ab (m : ℝ) (ab : ℝ) (ad : ℝ) : 
  ab = 2 →
  (ab ∈ roots_of_quadratic m) →
  (ad ∈ roots_of_quadratic m) →
  ab ≠ ad →
  m = 5 / 2 →
  2 * (ab + ad) = 5 :=
by
  sorry

end rhombus_condition_perimeter_rhombus_given_ab_l825_82544


namespace prime_sum_product_l825_82597

theorem prime_sum_product (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 101) : p * q = 194 :=
sorry

end prime_sum_product_l825_82597


namespace volume_of_locations_eq_27sqrt6pi_over_8_l825_82533

noncomputable def volumeOfLocationSet : ℝ :=
  let sqrt2_inv := 1 / (2 * Real.sqrt 2)
  let points := [ (sqrt2_inv, sqrt2_inv, sqrt2_inv),
                  (sqrt2_inv, sqrt2_inv, -sqrt2_inv),
                  (sqrt2_inv, -sqrt2_inv, sqrt2_inv),
                  (-sqrt2_inv, sqrt2_inv, sqrt2_inv) ]
  let condition (x y z : ℝ) : Prop :=
    4 * (x^2 + y^2 + z^2) + 3 / 2 ≤ 15
  let r := Real.sqrt (27 / 8)
  let volume := (4/3) * Real.pi * r^3
  volume

theorem volume_of_locations_eq_27sqrt6pi_over_8 :
  volumeOfLocationSet = 27 * Real.sqrt 6 * Real.pi / 8 :=
sorry

end volume_of_locations_eq_27sqrt6pi_over_8_l825_82533


namespace distinct_remainders_l825_82504

theorem distinct_remainders (p : ℕ) (a : Fin p → ℤ) (hp : Nat.Prime p) :
  ∃ k : ℤ, (Finset.univ.image (fun i : Fin p => (a i + i * k) % p)).card ≥ ⌈(p / 2 : ℚ)⌉ :=
sorry

end distinct_remainders_l825_82504


namespace ray_total_grocery_bill_l825_82519

noncomputable def meat_cost : ℝ := 5
noncomputable def crackers_cost : ℝ := 3.50
noncomputable def veg_cost_per_bag : ℝ := 2
noncomputable def veg_bags : ℕ := 4
noncomputable def cheese_cost : ℝ := 3.50
noncomputable def discount_rate : ℝ := 0.10

noncomputable def total_grocery_bill : ℝ :=
  let veg_total := veg_cost_per_bag * (veg_bags:ℝ)
  let total_before_discount := meat_cost + crackers_cost + veg_total + cheese_cost
  let discount := discount_rate * total_before_discount
  total_before_discount - discount

theorem ray_total_grocery_bill : total_grocery_bill = 18 :=
  by
  sorry

end ray_total_grocery_bill_l825_82519


namespace spend_together_is_85_l825_82526

variable (B D : ℝ)

theorem spend_together_is_85 (h1 : D = 0.70 * B) (h2 : B = D + 15) : B + D = 85 := by
  sorry

end spend_together_is_85_l825_82526


namespace Ron_four_times_Maurice_l825_82554

theorem Ron_four_times_Maurice
  (r m : ℕ) (x : ℕ) 
  (h_r : r = 43) 
  (h_m : m = 7) 
  (h_eq : r + x = 4 * (m + x)) : 
  x = 5 := 
by
  sorry

end Ron_four_times_Maurice_l825_82554


namespace cells_at_day_10_l825_82525

-- Define a function to compute the number of cells given initial cells, tripling rate, intervals, and total time.
def number_of_cells (initial_cells : ℕ) (ratio : ℕ) (interval : ℕ) (total_time : ℕ) : ℕ :=
  let n := total_time / interval + 1
  initial_cells * ratio^(n-1)

-- State the main theorem
theorem cells_at_day_10 :
  number_of_cells 5 3 2 10 = 1215 := by
  sorry

end cells_at_day_10_l825_82525


namespace orthodiagonal_quadrilateral_l825_82547

-- Define the quadrilateral sides and their relationships
variables (AB BC CD DA : ℝ)
variables (h1 : AB = 20) (h2 : BC = 70) (h3 : CD = 90)
theorem orthodiagonal_quadrilateral : AB^2 + CD^2 = BC^2 + DA^2 → DA = 60 :=
by
  sorry

end orthodiagonal_quadrilateral_l825_82547


namespace total_amount_for_uniforms_students_in_classes_cost_effective_purchase_plan_l825_82512

-- Define the conditions
def total_people (A B : ℕ) : Prop := A + B = 92
def valid_class_A (A : ℕ) : Prop := 51 < A ∧ A < 55
def total_cost (sets : ℕ) (cost_per_set : ℕ) : ℕ := sets * cost_per_set

-- Prices per set for different ranges of number of sets
def price_per_set (n : ℕ) : ℕ :=
  if n > 90 then 30 else if n > 50 then 40 else 50

-- Question 1
theorem total_amount_for_uniforms (A B : ℕ) (h1 : total_people A B) : total_cost 92 30 = 2760 := sorry

-- Question 2
theorem students_in_classes (A B : ℕ) (h1 : total_people A B) (h2 : valid_class_A A) (h3 : 40 * A + 50 * B = 4080) : A = 52 ∧ B = 40 := sorry

-- Question 3
theorem cost_effective_purchase_plan (A : ℕ) (h1 : 51 < A ∧ A < 55) (B : ℕ) (h2 : 92 - A = B) (h3 : A - 8 + B = 91) :
  ∃ (cost : ℕ), cost = total_cost 91 30 ∧ cost = 2730 := sorry

end total_amount_for_uniforms_students_in_classes_cost_effective_purchase_plan_l825_82512


namespace find_k_find_m_l825_82538

-- Condition definitions
def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (2, 3)

-- Proof problem statements
theorem find_k (k : ℝ) :
  (3 * a.fst - b.fst) / (a.fst + k * b.fst) = (3 * a.snd - b.snd) / (a.snd + k * b.snd) →
  k = -1 / 3 :=
sorry

theorem find_m (m : ℝ) :
  a.fst * (m * a.fst - b.fst) + a.snd * (m * a.snd - b.snd) = 0 →
  m = -4 / 5 :=
sorry

end find_k_find_m_l825_82538


namespace housing_price_growth_l825_82503

theorem housing_price_growth (x : ℝ) (h₁ : (5500 : ℝ) > 0) (h₂ : (7000 : ℝ) > 0) :
  5500 * (1 + x) ^ 2 = 7000 := 
sorry

end housing_price_growth_l825_82503


namespace fewer_blue_than_green_l825_82529

-- Definitions for given conditions
def green_buttons : ℕ := 90
def yellow_buttons : ℕ := green_buttons + 10
def total_buttons : ℕ := 275
def blue_buttons : ℕ := total_buttons - (green_buttons + yellow_buttons)

-- Theorem statement to be proved
theorem fewer_blue_than_green : green_buttons - blue_buttons = 5 :=
by
  -- Proof is omitted as per the instructions
  sorry

end fewer_blue_than_green_l825_82529


namespace area_of_rectangular_field_l825_82552

-- Define the conditions
def length (b : ℕ) : ℕ := b + 30
def perimeter (b : ℕ) (l : ℕ) : ℕ := 2 * (b + l)

-- Define the main theorem to prove
theorem area_of_rectangular_field (b : ℕ) (l : ℕ) (h1 : l = length b) (h2 : perimeter b l = 540) : 
  l * b = 18000 := by
  -- Placeholder for the proof
  sorry

end area_of_rectangular_field_l825_82552


namespace sum_of_x_values_l825_82522

theorem sum_of_x_values :
  (2^(x^2 + 6*x + 9) = 16^(x + 3)) → ∃ x1 x2 : ℝ, x1 + x2 = -2 :=
by
  sorry

end sum_of_x_values_l825_82522


namespace integer_value_expression_l825_82508

theorem integer_value_expression (p q : ℕ) (hp : Prime p) (hq : Prime q) : 
  (p = 2 ∧ q = 2) ∨ (p ≠ 2 ∧ q = 2 ∧ pq + p^p + q^q = 3 * (p + q)) :=
sorry

end integer_value_expression_l825_82508


namespace heartsuit_ratio_l825_82562

def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

theorem heartsuit_ratio :
  (heartsuit 3 5) / (heartsuit 5 3) = 5 / 3 :=
by
  sorry

end heartsuit_ratio_l825_82562


namespace extreme_points_inequality_l825_82587

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 + m * Real.log (1 + x)

-- Given m > 0 and f(x) has extreme points x1 and x2 such that x1 < x2
theorem extreme_points_inequality {m x1 x2 : ℝ} (h_m : m > 0)
    (h_extreme1 : x1 = (-1 - Real.sqrt (1 - 2 * m)) / 2)
    (h_extreme2 : x2 = (-1 + Real.sqrt (1 - 2 * m)) / 2)
    (h_order : x1 < x2) :
    2 * f x2 m > -x1 + 2 * x1 * Real.log 2 := sorry

end extreme_points_inequality_l825_82587


namespace find_y_given_z_25_l825_82594

theorem find_y_given_z_25 (k m x y z : ℝ) 
  (hk : y = k * x) 
  (hm : z = m * x)
  (hy5 : y = 10) 
  (hx5z15 : z = 15) 
  (hz25 : z = 25) : 
  y = 50 / 3 := 
  by sorry

end find_y_given_z_25_l825_82594


namespace combined_height_of_trees_l825_82581

noncomputable def growth_rate_A (weeks : ℝ) : ℝ := (weeks / 2) * 50
noncomputable def growth_rate_B (weeks : ℝ) : ℝ := (weeks / 3) * 70
noncomputable def growth_rate_C (weeks : ℝ) : ℝ := (weeks / 4) * 90
noncomputable def initial_height_A : ℝ := 200
noncomputable def initial_height_B : ℝ := 150
noncomputable def initial_height_C : ℝ := 250
noncomputable def total_weeks : ℝ := 16
noncomputable def total_growth_A := growth_rate_A total_weeks
noncomputable def total_growth_B := growth_rate_B total_weeks
noncomputable def total_growth_C := growth_rate_C total_weeks
noncomputable def final_height_A := initial_height_A + total_growth_A
noncomputable def final_height_B := initial_height_B + total_growth_B
noncomputable def final_height_C := initial_height_C + total_growth_C
noncomputable def final_combined_height := final_height_A + final_height_B + final_height_C

theorem combined_height_of_trees :
  final_combined_height = 1733.33 := by
  sorry

end combined_height_of_trees_l825_82581


namespace a_works_less_than_b_l825_82566

theorem a_works_less_than_b (A B : ℝ) (x y : ℝ)
  (h1 : A = 3 * B)
  (h2 : (A + B) * 22.5 = A * x)
  (h3 : y = 3 * x) :
  y - x = 60 :=
by sorry

end a_works_less_than_b_l825_82566


namespace find_n_l825_82530

noncomputable def satisfies_condition (n d₁ d₂ d₃ d₄ d₅ d₆ d₇ : ℕ) : Prop :=
  1 = d₁ ∧ d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧ d₄ < d₅ ∧ d₅ < d₆ ∧ d₆ < d₇ ∧ d₇ < n ∧
  (∀ d, d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄ ∨ d = d₅ ∨ d = d₆ ∨ d = d₇ ∨ d = n → n % d = 0) ∧
  (∀ d, n % d = 0 → d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄ ∨ d = d₅ ∨ d = d₆ ∨ d = d₇ ∨ d = n)

theorem find_n (n : ℕ) : (∃ d₁ d₂ d₃ d₄ d₅ d₆ d₇, satisfies_condition n d₁ d₂ d₃ d₄ d₅ d₆ d₇ ∧ n = d₆^2 + d₇^2 - 1) → (n = 144 ∨ n = 1984) :=
  by
  sorry

end find_n_l825_82530


namespace find_roots_of_equation_l825_82500

theorem find_roots_of_equation
  (a b c d x : ℝ)
  (h1 : a + d = 2015)
  (h2 : b + c = 2015)
  (h3 : a ≠ c)
  (h4 : (x - a) * (x - b) = (x - c) * (x - d)) :
  x = 1007.5 :=
by
  sorry

end find_roots_of_equation_l825_82500


namespace find_xy_l825_82575

theorem find_xy (x y : ℝ) (k : ℤ) :
  3 * Real.sin x - 4 * Real.cos x = 4 * y^2 + 4 * y + 6 ↔
  (x = -Real.arccos (-4/5) + (2 * k + 1) * Real.pi ∧ y = -1/2) := by
  sorry

end find_xy_l825_82575


namespace find_n_values_l825_82590

-- Define a function that calculates the polynomial expression
def prime_expression (n : ℕ) : ℕ :=
  n^4 - 27 * n^2 + 121

-- State the problem as a theorem
theorem find_n_values (n : ℕ) (h : Nat.Prime (prime_expression n)) : n = 2 ∨ n = 5 :=
  sorry

end find_n_values_l825_82590


namespace smallest_lcm_l825_82502

theorem smallest_lcm (a b : ℕ) (h₁ : 1000 ≤ a ∧ a < 10000) (h₂ : 1000 ≤ b ∧ b < 10000) (h₃ : Nat.gcd a b = 5) : 
  Nat.lcm a b = 201000 :=
sorry

end smallest_lcm_l825_82502


namespace line_canonical_eqn_l825_82518

theorem line_canonical_eqn 
  (x y z : ℝ)
  (h1 : x - y + z - 2 = 0)
  (h2 : x - 2*y - z + 4 = 0) :
  ∃ a : ℝ, ∃ b : ℝ, ∃ c : ℝ,
    (a = (x - 8)/3) ∧ (b = (y - 6)/2) ∧ (c = z/(-1)) ∧ (a = b) ∧ (b = c) ∧ (c = a) :=
by sorry

end line_canonical_eqn_l825_82518


namespace real_solutions_l825_82557

theorem real_solutions :
  ∃ x : ℝ, 
    (x = 9 ∨ x = 5) ∧ 
    (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
     1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 10) := 
by 
  sorry  

end real_solutions_l825_82557


namespace num_children_with_dogs_only_l825_82532

-- Defining the given values and constants
def total_children : ℕ := 30
def children_with_cats : ℕ := 12
def children_with_dogs_and_cats : ℕ := 6

-- Define the required proof statement
theorem num_children_with_dogs_only : 
  ∃ (D : ℕ), D + children_with_dogs_and_cats + (children_with_cats - children_with_dogs_and_cats) = total_children ∧ D = 18 :=
by
  sorry

end num_children_with_dogs_only_l825_82532


namespace mark_bananas_equals_mike_matt_fruits_l825_82506

theorem mark_bananas_equals_mike_matt_fruits :
  (∃ (bananas_mike matt_apples mark_bananas : ℕ),
    bananas_mike = 3 ∧
    matt_apples = 2 * bananas_mike ∧
    mark_bananas = 18 - (bananas_mike + matt_apples) ∧
    mark_bananas = (bananas_mike + matt_apples)) :=
sorry

end mark_bananas_equals_mike_matt_fruits_l825_82506


namespace sign_up_ways_l825_82553

theorem sign_up_ways : 
  let num_ways_A := 2
  let num_ways_B := 2
  let num_ways_C := 2
  num_ways_A * num_ways_B * num_ways_C = 8 := 
by 
  -- show the proof (omitted for simplicity)
  sorry

end sign_up_ways_l825_82553


namespace polynomial_at_neg_one_eq_neg_two_l825_82501

-- Define the polynomial f(x)
def polynomial (x : ℝ) : ℝ := 1 + x + 2 * x^2 + 3 * x^3 + 4 * x^4 + 5 * x^5

-- Define Horner's method process
def horner_method (x : ℝ) : ℝ :=
  let a5 := 5
  let a4 := 4
  let a3 := 3
  let a2 := 2
  let a1 := 1
  let a  := 1
  let u4 := a5 * x + a4
  let u3 := u4 * x + a3
  let u2 := u3 * x + a2
  let u1 := u2 * x + a1
  let u0 := u1 * x + a
  u0

-- Prove that the polynomial evaluated using Horner's method at x := -1 is equal to -2
theorem polynomial_at_neg_one_eq_neg_two : horner_method (-1) = -2 := by
  sorry

end polynomial_at_neg_one_eq_neg_two_l825_82501


namespace rectangle_length_l825_82539

theorem rectangle_length
    (a : ℕ)
    (b : ℕ)
    (area_square : a * a = 81)
    (width_rect : b = 3)
    (area_equal : a * a = b * (27) )
    : b * 27 = 81 :=
by
  sorry

end rectangle_length_l825_82539


namespace two_digit_product_l825_82582

theorem two_digit_product (x y : ℕ) (h₁ : 10 ≤ x) (h₂ : x < 100) (h₃ : 10 ≤ y) (h₄ : y < 100) (h₅ : x * y = 4320) :
  (x = 60 ∧ y = 72) ∨ (x = 72 ∧ y = 60) :=
sorry

end two_digit_product_l825_82582


namespace base_of_power_expr_l825_82559

-- Defining the power expression as a condition
def power_expr : ℤ := (-4 : ℤ) ^ 3

-- The Lean statement for the proof problem
theorem base_of_power_expr : ∃ b : ℤ, (power_expr = b ^ 3) ∧ (b = -4) := 
sorry

end base_of_power_expr_l825_82559


namespace hexagon_angle_E_l825_82514

theorem hexagon_angle_E (A N G L E S : ℝ) 
  (h1 : A = G) 
  (h2 : G = E) 
  (h3 : N + S = 180) 
  (h4 : L = 90) 
  (h_sum : A + N + G + L + E + S = 720) : 
  E = 150 := 
by 
  sorry

end hexagon_angle_E_l825_82514


namespace incorrect_statement_c_l825_82531

-- Define even function
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x

-- Define odd function
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Function definitions
def f1 (x : ℝ) : ℝ := x^4 + x^2
def f2 (x : ℝ) : ℝ := x^3 + x^2

-- Main theorem statement
theorem incorrect_statement_c : ¬ is_odd f2 := sorry

end incorrect_statement_c_l825_82531


namespace intersection_sum_l825_82549

theorem intersection_sum (h j : ℝ → ℝ)
  (H1 : h 3 = 3 ∧ j 3 = 3)
  (H2 : h 6 = 9 ∧ j 6 = 9)
  (H3 : h 9 = 18 ∧ j 9 = 18)
  (H4 : h 12 = 18 ∧ j 12 = 18) :
  ∃ a b : ℕ, h (3 * a) = b ∧ 3 * j a = b ∧ (a + b = 33) :=
by {
  sorry
}

end intersection_sum_l825_82549


namespace solve_for_x_l825_82572

noncomputable def f (x : ℝ) : ℝ := x^3

noncomputable def f_prime (x : ℝ) : ℝ := 3

theorem solve_for_x (x : ℝ) (h : f_prime x = 3) : x = 1 ∨ x = -1 :=
by
  sorry

end solve_for_x_l825_82572


namespace f_is_odd_f_min_value_pos_f_minimum_at_2_f_increasing_intervals_l825_82578

noncomputable def f (x : ℝ) : ℝ := x + 4/x

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

theorem f_min_value_pos : ∀ x : ℝ, x > 0 → f x ≥ 4 :=
by
  sorry

theorem f_minimum_at_2 : f 2 = 4 :=
by
  sorry

theorem f_increasing_intervals : (MonotoneOn f {x | x ≤ -2} ∧ MonotoneOn f {x | x ≥ 2}) :=
by
  sorry

end f_is_odd_f_min_value_pos_f_minimum_at_2_f_increasing_intervals_l825_82578


namespace fleas_difference_l825_82540

-- Define the initial number of fleas and subsequent fleas after each treatment.
def initial_fleas (F : ℝ) := F
def after_first_treatment (F : ℝ) := F * 0.40
def after_second_treatment (F : ℝ) := (after_first_treatment F) * 0.55
def after_third_treatment (F : ℝ) := (after_second_treatment F) * 0.70
def after_fourth_treatment (F : ℝ) := (after_third_treatment F) * 0.80

-- Given condition
axiom final_fleas : initial_fleas 20 = after_fourth_treatment 20

-- Prove the number of fleas before treatment minus the number after treatment is 142
theorem fleas_difference (F : ℝ) (h : initial_fleas F = after_fourth_treatment 20) : 
  F - 20 = 142 :=
by {
  sorry
}

end fleas_difference_l825_82540


namespace find_expression_l825_82550

def B : ℂ := 3 + 2 * Complex.I
def Q : ℂ := -5 * Complex.I
def R : ℂ := 1 + Complex.I
def T : ℂ := 3 - 4 * Complex.I

theorem find_expression : B * R + Q + T = 4 + Complex.I := by
  sorry

end find_expression_l825_82550


namespace initial_price_of_gasoline_l825_82558

theorem initial_price_of_gasoline 
  (P0 : ℝ) 
  (P1 : ℝ := 1.30 * P0)
  (P2 : ℝ := 0.75 * P1)
  (P3 : ℝ := 1.10 * P2)
  (P4 : ℝ := 0.85 * P3)
  (P5 : ℝ := 0.80 * P4)
  (h : P5 = 102.60) : 
  P0 = 140.67 :=
by sorry

end initial_price_of_gasoline_l825_82558


namespace word_count_with_a_l825_82586

-- Defining the constants for the problem
def alphabet_size : ℕ := 26
def no_a_size : ℕ := 25

-- Calculating words that contain 'A' for lengths 1 to 5
def words_with_a (len : ℕ) : ℕ :=
  alphabet_size ^ len - no_a_size ^ len

-- The main theorem statement
theorem word_count_with_a : words_with_a 1 + words_with_a 2 + words_with_a 3 + words_with_a 4 + words_with_a 5 = 2186085 :=
by
  -- Calculations are established in the problem statement
  sorry

end word_count_with_a_l825_82586


namespace fermats_little_theorem_poly_binom_coeff_divisible_by_prime_l825_82585

variable (p : ℕ) [Fact (Nat.Prime p)]

theorem fermats_little_theorem_poly (X : ℤ) :
  (X + 1) ^ p = X ^ p + 1 := by
    sorry

theorem binom_coeff_divisible_by_prime {k : ℕ} (hkp : 1 ≤ k ∧ k < p) :
  p ∣ Nat.choose p k := by
    sorry

end fermats_little_theorem_poly_binom_coeff_divisible_by_prime_l825_82585


namespace sum_of_squares_geometric_progression_theorem_l825_82535

noncomputable def sum_of_squares_geometric_progression (a₁ q : ℝ) (S₁ S₂ : ℝ)
  (h_q : abs q < 1)
  (h_S₁ : S₁ = a₁ / (1 - q))
  (h_S₂ : S₂ = a₁ / (1 + q)) : ℝ :=
  S₁ * S₂

theorem sum_of_squares_geometric_progression_theorem
  (a₁ q S₁ S₂ : ℝ)
  (h_q : abs q < 1)
  (h_S₁ : S₁ = a₁ / (1 - q))
  (h_S₂ : S₂ = a₁ / (1 + q)) :
  sum_of_squares_geometric_progression a₁ q S₁ S₂ h_q h_S₁ h_S₂ = S₁ * S₂ := sorry

end sum_of_squares_geometric_progression_theorem_l825_82535


namespace least_cost_of_grass_seed_l825_82563

-- Definitions of the prices and weights
def price_per_bag (size : Nat) : Float :=
  if size = 5 then 13.85
  else if size = 10 then 20.40
  else if size = 25 then 32.25
  else 0.0

-- The conditions for the weights and costs
def valid_weight_range (total_weight : Nat) : Prop :=
  65 ≤ total_weight ∧ total_weight ≤ 80

-- Calculate the total cost given quantities of each bag size
def total_cost (bag5 : Nat) (bag10 : Nat) (bag25 : Nat) : Float :=
  Float.ofNat bag5 * price_per_bag 5 + Float.ofNat bag10 * price_per_bag 10 + Float.ofNat bag25 * price_per_bag 25

-- Correct cost for the minimum possible cost within the given weight range
def min_possible_cost : Float := 98.75

-- Proof statement to be proven
theorem least_cost_of_grass_seed : ∃ (bag5 bag10 bag25 : Nat), 
  valid_weight_range (bag5 * 5 + bag10 * 10 + bag25 * 25) ∧ total_cost bag5 bag10 bag25 = min_possible_cost :=
sorry

end least_cost_of_grass_seed_l825_82563


namespace solve_equation_l825_82569

theorem solve_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (x / (x + 1) = 2 / (x^2 - 1)) ↔ (x = 2) :=
by
  sorry

end solve_equation_l825_82569


namespace fill_tank_with_only_C_l825_82573

noncomputable def time_to_fill_with_only_C (x y z : ℝ) : ℝ := 
  let eq1 := (1 / z - 1 / x) * 2 = 1
  let eq2 := (1 / z - 1 / y) * 4 = 1
  let eq3 := 1 / z * 5 - (1 / x + 1 / y) * 8 = 0
  z

theorem fill_tank_with_only_C (x y z : ℝ) (h1 : (1 / z - 1 / x) * 2 = 1) 
  (h2 : (1 / z - 1 / y) * 4 = 1) (h3 : 1 / z * 5 - (1 / x + 1 / y) * 8 = 0) : 
  time_to_fill_with_only_C x y z = 11 / 6 :=
by
  sorry

end fill_tank_with_only_C_l825_82573


namespace joe_total_time_l825_82537

variable (r_w t_w : ℝ) 
variable (t_total : ℝ)

-- Given conditions:
def joe_problem_conditions : Prop :=
  (r_w > 0) ∧ 
  (t_w = 9) ∧
  (3 * r_w * (3)) / 2 = r_w * 9 / 2 + 1 / 2

-- The statement to prove:
theorem joe_total_time (h : joe_problem_conditions r_w t_w) : t_total = 13 :=
by { sorry }

end joe_total_time_l825_82537


namespace area_of_square_same_yarn_l825_82513

theorem area_of_square_same_yarn (a : ℕ) (ha : a = 4) :
  let hexagon_perimeter := 6 * a
  let square_side := hexagon_perimeter / 4
  square_side * square_side = 36 :=
by
  sorry

end area_of_square_same_yarn_l825_82513


namespace monotonically_increasing_range_k_l825_82580

noncomputable def f (k x : ℝ) : ℝ := k * x - Real.log x

theorem monotonically_increasing_range_k :
  (∀ x > 1, deriv (f k) x ≥ 0) → k ≥ 1 :=
sorry

end monotonically_increasing_range_k_l825_82580


namespace stair_calculation_l825_82555

def already_climbed : ℕ := 74
def left_to_climb : ℕ := 22
def total_stairs : ℕ := 96

theorem stair_calculation :
  already_climbed + left_to_climb = total_stairs :=
by {
  sorry
}

end stair_calculation_l825_82555


namespace quadratic_equal_roots_iff_l825_82589

theorem quadratic_equal_roots_iff (k : ℝ) :
  (∃ x : ℝ, x^2 - k * x + 9 = 0 ∧ x^2 - k * x + 9 = 0 ∧ x = x) ↔ k^2 = 36 :=
by
  sorry

end quadratic_equal_roots_iff_l825_82589


namespace part1_part2_l825_82528

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part1 (x : ℝ) : f x ≥ 2 :=
by
  sorry

theorem part2 (x : ℝ) : (∀ b : ℝ, b ≠ 0 → f x ≥ (|2 * b + 1| - |1 - b|) / |b|) → (x ≤ -1.5 ∨ x ≥ 1.5) :=
by
  sorry

end part1_part2_l825_82528


namespace real_values_of_a_l825_82561

noncomputable def P (x a b : ℝ) : ℝ := x^2 - 2 * a * x + b

theorem real_values_of_a (a b : ℝ) :
  (P 0 a b ≠ 0) →
  (P 1 a b ≠ 0) →
  (P 2 a b ≠ 0) →
  (P 1 a b / P 0 a b = P 2 a b / P 1 a b) →
  (∃ b, P x 1 b = 0) :=
by
  sorry

end real_values_of_a_l825_82561


namespace total_weight_of_peppers_l825_82543

def green_peppers_weight : Real := 0.3333333333333333
def red_peppers_weight : Real := 0.3333333333333333
def total_peppers_weight : Real := 0.6666666666666666

theorem total_weight_of_peppers :
  green_peppers_weight + red_peppers_weight = total_peppers_weight :=
by
  sorry

end total_weight_of_peppers_l825_82543
