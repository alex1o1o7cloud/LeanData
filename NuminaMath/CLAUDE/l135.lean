import Mathlib

namespace dress_design_count_l135_13578

/-- The number of available fabric colors -/
def num_colors : ℕ := 5

/-- The number of available patterns -/
def num_patterns : ℕ := 4

/-- The number of available sleeve styles -/
def num_sleeve_styles : ℕ := 3

/-- Each dress design requires exactly one color, one pattern, and one sleeve style -/
axiom dress_design_composition : True

/-- The total number of possible dress designs -/
def total_dress_designs : ℕ := num_colors * num_patterns * num_sleeve_styles

theorem dress_design_count : total_dress_designs = 60 := by
  sorry

end dress_design_count_l135_13578


namespace adult_tickets_bought_l135_13509

/-- Proves the number of adult tickets bought given ticket prices and total information -/
theorem adult_tickets_bought (adult_price child_price : ℚ) (total_tickets : ℕ) (total_cost : ℚ) 
  (h1 : adult_price = 5.5)
  (h2 : child_price = 3.5)
  (h3 : total_tickets = 21)
  (h4 : total_cost = 83.5) :
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_cost ∧ 
    adult_tickets = 5 := by
  sorry

end adult_tickets_bought_l135_13509


namespace unique_number_l135_13548

theorem unique_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  n % 13 = 0 ∧ 
  n > 26 ∧ 
  n % 7 = 0 ∧ 
  n % 10 ≠ 6 ∧ 
  n % 10 ≠ 8 := by
  sorry

end unique_number_l135_13548


namespace min_value_expression_l135_13538

theorem min_value_expression (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 3 + 27 / n ≥ 6 ∧
  ((n : ℝ) / 3 + 27 / n = 6 ↔ n = 9) :=
by sorry

end min_value_expression_l135_13538


namespace all_transformations_pass_through_point_l135_13542

def f (x : ℝ) := (x - 2)^2
def g (x : ℝ) := (x - 1)^2 - 1
def h (x : ℝ) := x^2 - 4
def k (x : ℝ) := -x^2 + 4

theorem all_transformations_pass_through_point :
  f 2 = 0 ∧ g 2 = 0 ∧ h 2 = 0 ∧ k 2 = 0 := by
  sorry

end all_transformations_pass_through_point_l135_13542


namespace circle_equation_example_l135_13591

/-- The standard equation of a circle with center (h,k) and radius r is (x-h)^2 + (y-k)^2 = r^2 -/
def standard_circle_equation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Given a circle with center (3,1) and radius 5, its standard equation is (x-3)^2+(y-1)^2=25 -/
theorem circle_equation_example :
  ∀ x y : ℝ, standard_circle_equation 3 1 5 x y ↔ (x - 3)^2 + (y - 1)^2 = 25 := by
  sorry

end circle_equation_example_l135_13591


namespace four_correct_propositions_l135_13535

theorem four_correct_propositions (x y : ℝ) :
  (((x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0) ∧
   ((x^2 + y^2 ≠ 0) → (x ≠ 0 ∨ y ≠ 0)) ∧
   (¬((x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0)) ∧
   ((x^2 + y^2 = 0) → (x = 0 ∧ y = 0))) :=
by sorry

end four_correct_propositions_l135_13535


namespace boat_round_trip_time_l135_13546

/-- Calculate the total time for a round trip by boat, given the boat's speed in standing water,
    the stream's speed, and the distance to the destination. -/
theorem boat_round_trip_time
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (distance : ℝ)
  (h1 : boat_speed = 16)
  (h2 : stream_speed = 2)
  (h3 : distance = 7200)
  : ∃ (total_time : ℝ), abs (total_time - 914.2857) < 0.0001 :=
by
  sorry


end boat_round_trip_time_l135_13546


namespace temperature_conversion_deviation_l135_13544

theorem temperature_conversion_deviation (C : ℝ) : 
  let F_approx := 2 * C + 30
  let F_exact := (9 / 5) * C + 32
  let deviation := (F_approx - F_exact) / F_exact
  (40 / 29 ≤ C ∧ C ≤ 360 / 11) ↔ (abs deviation ≤ 0.05) :=
by sorry

end temperature_conversion_deviation_l135_13544


namespace evaluate_expression_l135_13554

theorem evaluate_expression (x y : ℝ) (hx : x = 4) (hy : y = 5) :
  2 * y * (y - 2 * x) = -30 := by
  sorry

end evaluate_expression_l135_13554


namespace unique_number_with_remainders_and_quotient_condition_l135_13505

theorem unique_number_with_remainders_and_quotient_condition :
  ∃! (n : ℕ),
    n > 0 ∧
    n % 7 = 2 ∧
    n % 8 = 4 ∧
    (n - 2) / 7 = (n - 4) / 8 + 7 ∧
    n = 380 := by
  sorry

end unique_number_with_remainders_and_quotient_condition_l135_13505


namespace equation_solutions_l135_13517

theorem equation_solutions : 
  {x : ℝ | (2 + x)^(2/3) + 3 * (2 - x)^(2/3) = 4 * (4 - x^2)^(1/3)} = {0, 13/7} := by
  sorry

end equation_solutions_l135_13517


namespace number_relationship_l135_13584

theorem number_relationship (A B C : ℕ) : 
  A + B + C = 660 → A = 2 * B → B = 180 → C = A - 240 := by sorry

end number_relationship_l135_13584


namespace students_passed_l135_13516

theorem students_passed (total : ℕ) (fail_freq : ℚ) (h1 : total = 1000) (h2 : fail_freq = 0.4) :
  total - (total * fail_freq).floor = 600 := by
  sorry

end students_passed_l135_13516


namespace final_basketball_count_l135_13575

def initial_count : ℕ := 100

def transactions : List ℤ := [38, -42, 27, -33, -40]

theorem final_basketball_count : 
  initial_count + transactions.sum = 50 := by sorry

end final_basketball_count_l135_13575


namespace log_relation_l135_13541

theorem log_relation (a b : ℝ) (ha : a = Real.log 225 / Real.log 8) (hb : b = Real.log 15 / Real.log 2) : 
  a = (2 * b) / 3 := by
  sorry

end log_relation_l135_13541


namespace intersection_of_A_and_B_l135_13579

def A : Set ℝ := {x | (x - 2) / (x + 3) ≤ 0}
def B : Set ℝ := {x | x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {x | -3 < x ∧ x ≤ 1} := by sorry

end intersection_of_A_and_B_l135_13579


namespace min_abs_phi_l135_13514

/-- Given a function y = 3cos(2x + φ) with its graph symmetric about (2π/3, 0),
    the minimum value of |φ| is π/6 -/
theorem min_abs_phi (φ : ℝ) : 
  (∀ x, 3 * Real.cos (2 * x + φ) = 3 * Real.cos (2 * (4 * π / 3 - x) + φ)) →
  (∃ k : ℤ, φ = k * π - 5 * π / 6) →
  π / 6 ≤ |φ| ∧ (∃ φ₀, |φ₀| = π / 6 ∧ 
    (∀ x, 3 * Real.cos (2 * x + φ₀) = 3 * Real.cos (2 * (4 * π / 3 - x) + φ₀))) :=
by sorry

end min_abs_phi_l135_13514


namespace fractional_equation_solution_l135_13582

theorem fractional_equation_solution :
  ∃ (x : ℝ), x ≠ 1 ∧ (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end fractional_equation_solution_l135_13582


namespace sum_of_solutions_quadratic_l135_13592

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 9*x - 20) → (∃ y : ℝ, y^2 = 9*y - 20 ∧ x + y = 9) :=
by
  sorry

end sum_of_solutions_quadratic_l135_13592


namespace z_is_in_fourth_quadrant_l135_13556

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the sets M and N
def M (z : ℂ) : Set ℂ := {1, z * (1 + i)}
def N : Set ℂ := {3, 4}

-- State the theorem
theorem z_is_in_fourth_quadrant (z : ℂ) :
  M z ∪ N = {1, 2, 3, 4} → z = 1 - i := by
  sorry

end z_is_in_fourth_quadrant_l135_13556


namespace cylinder_volume_l135_13576

/-- The volume of a cylinder whose lateral surface unfolds into a square with side length 4 -/
theorem cylinder_volume (h : ℝ) (r : ℝ) : 
  h = 4 → 2 * Real.pi * r = 4 → Real.pi * r^2 * h = 16 / Real.pi := by
  sorry

end cylinder_volume_l135_13576


namespace shaded_area_in_circumscribed_square_l135_13521

/-- Given a square with side length 20 cm circumscribed around a circle,
    where two of its diagonals form an isosceles triangle with the circle's center,
    the sum of the areas of the two small shaded regions is 100π - 200 square centimeters. -/
theorem shaded_area_in_circumscribed_square (π : ℝ) :
  let square_side : ℝ := 20
  let circle_radius : ℝ := square_side * Real.sqrt 2 / 2
  let sector_area : ℝ := π * circle_radius^2 / 4
  let triangle_area : ℝ := circle_radius^2 / 2
  let shaded_area : ℝ := 2 * (sector_area - triangle_area)
  shaded_area = 100 * π - 200 := by
  sorry

end shaded_area_in_circumscribed_square_l135_13521


namespace hcl_concentration_in_mixed_solution_l135_13569

/-- Calculates the concentration of HCl in a mixed solution -/
theorem hcl_concentration_in_mixed_solution 
  (volume1 : ℝ) (concentration1 : ℝ) 
  (volume2 : ℝ) (concentration2 : ℝ) :
  volume1 = 60 →
  concentration1 = 0.4 →
  volume2 = 90 →
  concentration2 = 0.15 →
  (volume1 * concentration1 + volume2 * concentration2) / (volume1 + volume2) = 0.25 := by
  sorry

end hcl_concentration_in_mixed_solution_l135_13569


namespace journey_distance_ratio_l135_13532

/-- Given a journey where:
  - The initial distance traveled is 20 hours at 30 kilometers per hour
  - After a setback, the traveler is one-third of the way to the destination
  Prove that the ratio of the initial distance to the total distance is 1/3 -/
theorem journey_distance_ratio :
  ∀ (initial_speed : ℝ) (initial_time : ℝ) (total_distance : ℝ),
    initial_speed = 30 →
    initial_time = 20 →
    initial_speed * initial_time = (1/3) * total_distance →
    (initial_speed * initial_time) / total_distance = 1/3 := by
  sorry

end journey_distance_ratio_l135_13532


namespace sum_of_specific_S_l135_13543

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -n else n + 1

theorem sum_of_specific_S : S 18 + S 34 + S 51 = 0 := by sorry

end sum_of_specific_S_l135_13543


namespace nancy_insurance_percentage_l135_13520

/-- Given a monthly insurance cost and an annual payment, 
    calculate the percentage of the total cost being paid. -/
def insurance_percentage (monthly_cost : ℚ) (annual_payment : ℚ) : ℚ :=
  (annual_payment / (monthly_cost * 12)) * 100

/-- Theorem stating that for a monthly cost of $80 and an annual payment of $384,
    the percentage paid is 40% of the total cost. -/
theorem nancy_insurance_percentage :
  insurance_percentage 80 384 = 40 := by
  sorry

end nancy_insurance_percentage_l135_13520


namespace inscribed_circle_path_length_l135_13503

theorem inscribed_circle_path_length (a b c : ℝ) (h_triangle : a = 10 ∧ b = 8 ∧ c = 12) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := area / s
  (a + b + c) - 2 * r = 15 :=
sorry

end inscribed_circle_path_length_l135_13503


namespace uncool_parents_in_two_classes_l135_13562

/-- Represents a math class with information about cool parents -/
structure MathClass where
  total_students : ℕ
  cool_dads : ℕ
  cool_moms : ℕ
  both_cool : ℕ

/-- Calculates the number of students with uncool parents in a class -/
def uncool_parents (c : MathClass) : ℕ :=
  c.total_students - (c.cool_dads + c.cool_moms - c.both_cool)

/-- The problem statement -/
theorem uncool_parents_in_two_classes 
  (class1 : MathClass)
  (class2 : MathClass)
  (h1 : class1.total_students = 45)
  (h2 : class1.cool_dads = 22)
  (h3 : class1.cool_moms = 25)
  (h4 : class1.both_cool = 11)
  (h5 : class2.total_students = 35)
  (h6 : class2.cool_dads = 15)
  (h7 : class2.cool_moms = 18)
  (h8 : class2.both_cool = 7) :
  uncool_parents class1 + uncool_parents class2 = 18 := by
  sorry

end uncool_parents_in_two_classes_l135_13562


namespace max_value_theorem_l135_13518

theorem max_value_theorem (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_constraint : x + y + z = 3) :
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 243/16 :=
by sorry

end max_value_theorem_l135_13518


namespace quadratic_two_roots_l135_13596

theorem quadratic_two_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0) ↔ m < 1 := by
  sorry

end quadratic_two_roots_l135_13596


namespace complete_residue_system_l135_13588

theorem complete_residue_system (m : ℕ) (x : Fin m → ℤ) 
  (h : ∀ i j : Fin m, i ≠ j → x i % m ≠ x j % m) :
  ∀ k : Fin m, ∃ i : Fin m, x i % m = k.val :=
sorry

end complete_residue_system_l135_13588


namespace positive_numbers_inequality_l135_13568

theorem positive_numbers_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end positive_numbers_inequality_l135_13568


namespace equation_solutions_l135_13525

theorem equation_solutions (x : ℝ) : 
  (x - 1)^2 * (x - 5)^2 / (x - 5) = 4 ↔ x = 3 ∨ x = -1 :=
by sorry

end equation_solutions_l135_13525


namespace adrianna_gum_theorem_l135_13566

/-- Calculates the remaining pieces of gum after sharing with friends -/
def remaining_gum (initial : ℕ) (additional : ℕ) (friends : ℕ) : ℕ :=
  initial + additional - friends

/-- Theorem stating that Adrianna's remaining gum pieces is 2 -/
theorem adrianna_gum_theorem (initial : ℕ) (additional : ℕ) (friends : ℕ)
  (h1 : initial = 10)
  (h2 : additional = 3)
  (h3 : friends = 11) :
  remaining_gum initial additional friends = 2 := by
  sorry

#eval remaining_gum 10 3 11

end adrianna_gum_theorem_l135_13566


namespace carrot_weight_problem_l135_13549

/-- Given 30 carrots weighing 5.94 kg, and 27 of these carrots having an average weight of 200 grams,
    the average weight of the remaining 3 carrots is 180 grams. -/
theorem carrot_weight_problem (total_weight : ℝ) (avg_weight_27 : ℝ) :
  total_weight = 5.94 →
  avg_weight_27 = 0.2 →
  (total_weight * 1000 - 27 * avg_weight_27 * 1000) / 3 = 180 :=
by sorry

end carrot_weight_problem_l135_13549


namespace solve_equation_1_solve_equation_2_solve_equation_3_solve_equation_4_l135_13583

-- Problem 1
theorem solve_equation_1 : ∃ x : ℝ, (3 * x^2 - 32 * x - 48 = 0) ↔ (x = 12 ∨ x = -4/3) := by sorry

-- Problem 2
theorem solve_equation_2 : ∃ x : ℝ, (4 * x^2 + x - 3 = 0) ↔ (x = 3/4 ∨ x = -1) := by sorry

-- Problem 3
theorem solve_equation_3 : ∃ x : ℝ, ((3 * x + 1)^2 - 4 = 0) ↔ (x = 1/3 ∨ x = -1) := by sorry

-- Problem 4
theorem solve_equation_4 : ∃ x : ℝ, (9 * (x - 2)^2 = 4 * (x + 1)^2) ↔ (x = 8 ∨ x = 4/5) := by sorry

end solve_equation_1_solve_equation_2_solve_equation_3_solve_equation_4_l135_13583


namespace prob_green_face_is_half_l135_13547

/-- A six-faced dice with colored faces -/
structure ColoredDice :=
  (total_faces : ℕ)
  (green_faces : ℕ)
  (yellow_faces : ℕ)
  (purple_faces : ℕ)
  (h_total : total_faces = 6)
  (h_green : green_faces = 3)
  (h_yellow : yellow_faces = 2)
  (h_purple : purple_faces = 1)
  (h_sum : green_faces + yellow_faces + purple_faces = total_faces)

/-- The probability of rolling a green face on the colored dice -/
def prob_green_face (d : ColoredDice) : ℚ :=
  d.green_faces / d.total_faces

/-- Theorem: The probability of rolling a green face is 1/2 -/
theorem prob_green_face_is_half (d : ColoredDice) :
  prob_green_face d = 1/2 := by
  sorry

end prob_green_face_is_half_l135_13547


namespace B_power_15_minus_3_power_14_l135_13536

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 4; 0, -2] := by sorry

end B_power_15_minus_3_power_14_l135_13536


namespace age_difference_l135_13560

/-- The problem of finding the age difference between A and B -/
theorem age_difference (a b : ℕ) : b = 36 → a + 10 = 2 * (b - 10) → a - b = 6 := by
  sorry

end age_difference_l135_13560


namespace arithmetic_mean_problem_l135_13511

theorem arithmetic_mean_problem (x : ℝ) : 
  ((x + 8) + 15 + (2 * x) + 13 + (2 * x + 4) + (3 * x + 5)) / 6 = 30 → x = 13.5 := by
sorry

end arithmetic_mean_problem_l135_13511


namespace cakes_sold_is_six_l135_13550

/-- The number of cakes sold during dinner today, given the number of cakes
    baked today, yesterday, and the number of cakes left. -/
def cakes_sold_during_dinner (cakes_baked_today cakes_baked_yesterday cakes_left : ℕ) : ℕ :=
  cakes_baked_today + cakes_baked_yesterday - cakes_left

/-- Theorem stating that the number of cakes sold during dinner today is 6,
    given the specific conditions of the problem. -/
theorem cakes_sold_is_six :
  cakes_sold_during_dinner 5 3 2 = 6 := by
  sorry

end cakes_sold_is_six_l135_13550


namespace maria_eggs_l135_13537

/-- The number of eggs Maria has -/
def total_eggs (num_boxes : ℕ) (eggs_per_box : ℕ) : ℕ :=
  num_boxes * eggs_per_box

/-- Theorem: Maria has 21 eggs in total -/
theorem maria_eggs : total_eggs 3 7 = 21 := by
  sorry

end maria_eggs_l135_13537


namespace altitude_difference_l135_13558

theorem altitude_difference (a b c : ℤ) (ha : a = -112) (hb : b = -80) (hc : c = -25) :
  (max a (max b c) - min a (min b c) : ℤ) = 87 := by
  sorry

end altitude_difference_l135_13558


namespace factorization_analysis_l135_13512

theorem factorization_analysis (x y a b : ℝ) :
  (x^4 - y^4 = (x^2 + y^2) * (x + y) * (x - y)) ∧
  (x^3*y - 2*x^2*y^2 + x*y^3 = x*y*(x - y)^2) ∧
  (4*x^2 - 4*x + 1 = (2*x - 1)^2) ∧
  (4*(a - b)^2 + 1 + 4*(a - b) = (2*a - 2*b + 1)^2) :=
by
  sorry

end factorization_analysis_l135_13512


namespace cos_double_angle_special_l135_13530

theorem cos_double_angle_special (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.sin α + Real.cos α = 1/5) : Real.cos (2 * α) = -7/25 := by
  sorry

end cos_double_angle_special_l135_13530


namespace andrew_sandwiches_l135_13572

/-- Given a total number of sandwiches and number of friends, 
    calculate the number of sandwiches per friend -/
def sandwiches_per_friend (total_sandwiches : ℕ) (num_friends : ℕ) : ℕ :=
  total_sandwiches / num_friends

/-- Theorem: Given 12 sandwiches and 4 friends, 
    the number of sandwiches per friend is 3 -/
theorem andrew_sandwiches : 
  sandwiches_per_friend 12 4 = 3 := by
  sorry


end andrew_sandwiches_l135_13572


namespace smallest_positive_shift_is_90_l135_13590

/-- A function with a 30-unit shift property -/
def ShiftFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x - 30) = f x

/-- The smallest positive shift for the scaled function -/
def SmallestPositiveShift (f : ℝ → ℝ) (b : ℝ) : Prop :=
  b > 0 ∧
  (∀ x : ℝ, f ((x - b) / 3) = f (x / 3)) ∧
  (∀ c : ℝ, c > 0 → (∀ x : ℝ, f ((x - c) / 3) = f (x / 3)) → b ≤ c)

theorem smallest_positive_shift_is_90 (f : ℝ → ℝ) (h : ShiftFunction f) :
  SmallestPositiveShift f 90 :=
sorry

end smallest_positive_shift_is_90_l135_13590


namespace triangle_side_length_l135_13515

theorem triangle_side_length (a c area : ℝ) (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
    (h_a : a = 4) (h_c : c = 6) (h_area : area = 6 * Real.sqrt 3) : 
    ∃ (b : ℝ), b^2 = 28 := by
  sorry

end triangle_side_length_l135_13515


namespace arithmetic_sequence_sum_l135_13564

theorem arithmetic_sequence_sum (a₁ d : ℕ) (n : ℕ) :
  a₁ = 3 → d = 3 → n = 10 →
  (n : ℝ) / 2 * (a₁ + (a₁ + (n - 1) * d)) = 165 := by sorry

end arithmetic_sequence_sum_l135_13564


namespace sum_five_consecutive_integers_l135_13551

theorem sum_five_consecutive_integers (n : ℤ) : 
  (n - 2) + (n - 1) + n + (n + 1) + (n + 2) = 5 * n := by
  sorry

end sum_five_consecutive_integers_l135_13551


namespace vector_sum_magnitude_l135_13555

def a : ℝ × ℝ := (2, 0)

theorem vector_sum_magnitude (b : ℝ × ℝ) 
  (h1 : Real.cos (Real.pi / 3) = (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  (h2 : b.1^2 + b.2^2 = 1) : 
  Real.sqrt ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2) = 2 * Real.sqrt 3 := by
sorry

end vector_sum_magnitude_l135_13555


namespace sea_lion_count_l135_13581

/-- Given the ratio of sea lions to penguins and their difference, 
    calculate the number of sea lions -/
theorem sea_lion_count (s p : ℕ) : 
  s * 11 = p * 4 →  -- ratio of sea lions to penguins is 4:11
  p = s + 84 →      -- 84 more penguins than sea lions
  s = 48 :=         -- prove that there are 48 sea lions
by sorry

end sea_lion_count_l135_13581


namespace supermarket_queue_clearing_time_l135_13597

/-- The average number of people lining up to pay per hour -/
def average_customers_per_hour : ℝ := 60

/-- The number of people a single cashier can handle per hour -/
def cashier_capacity_per_hour : ℝ := 80

/-- The number of hours it takes for one cashier to clear the line -/
def hours_for_one_cashier : ℝ := 4

/-- The number of cashiers working in the second scenario -/
def num_cashiers : ℕ := 2

/-- The time it takes for two cashiers to clear the line -/
def time_for_two_cashiers : ℝ := 0.8

theorem supermarket_queue_clearing_time :
  2 * cashier_capacity_per_hour * time_for_two_cashiers = 
  average_customers_per_hour * time_for_two_cashiers + 
  (cashier_capacity_per_hour * hours_for_one_cashier - average_customers_per_hour * hours_for_one_cashier) :=
by sorry

end supermarket_queue_clearing_time_l135_13597


namespace work_ratio_proof_l135_13574

/-- Represents the work rate of a single cat -/
def single_cat_rate : ℝ := 1

/-- Represents the total work to be done -/
def total_work : ℝ := 10

/-- Represents the number of days the initial cats work -/
def initial_days : ℕ := 5

/-- Represents the total number of days to complete the work -/
def total_days : ℕ := 7

/-- Represents the initial number of cats -/
def initial_cats : ℕ := 2

/-- Represents the final number of cats -/
def final_cats : ℕ := 5

theorem work_ratio_proof :
  let initial_work := (initial_cats : ℝ) * single_cat_rate * initial_days
  let remaining_days := total_days - initial_days
  let remaining_work := (final_cats : ℝ) * single_cat_rate * remaining_days
  initial_work / (initial_work + remaining_work) = 1 / 2 := by
sorry


end work_ratio_proof_l135_13574


namespace successive_integers_product_l135_13599

theorem successive_integers_product (n : ℤ) : n * (n + 1) = 9506 → n = 97 := by
  sorry

end successive_integers_product_l135_13599


namespace line_passes_through_fixed_point_l135_13573

/-- The line y - 1 = k(x + 2) passes through the point (-2, 1) for all values of k. -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (1 : ℝ) - 1 = k * ((-2 : ℝ) + 2) := by
  sorry

end line_passes_through_fixed_point_l135_13573


namespace product_trailing_zeros_l135_13531

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of 125 and 960 -/
def product : ℕ := 125 * 960

theorem product_trailing_zeros :
  trailingZeros product = 4 := by sorry

end product_trailing_zeros_l135_13531


namespace cucumber_weight_problem_l135_13557

/-- Given cucumbers that are initially 99% water by weight, then 96% water by weight after
    evaporation with a new weight of 25 pounds, prove that the initial weight was 100 pounds. -/
theorem cucumber_weight_problem (initial_water_percent : ℝ) (final_water_percent : ℝ) (final_weight : ℝ) :
  initial_water_percent = 0.99 →
  final_water_percent = 0.96 →
  final_weight = 25 →
  ∃ (initial_weight : ℝ), initial_weight = 100 ∧
    (1 - initial_water_percent) * initial_weight = (1 - final_water_percent) * final_weight :=
by sorry

end cucumber_weight_problem_l135_13557


namespace num_factors_180_multiple_15_eq_6_l135_13561

/-- The number of positive factors of 180 that are also multiples of 15 -/
def num_factors_180_multiple_15 : ℕ :=
  (Finset.filter (λ x => 180 % x = 0 ∧ x % 15 = 0) (Finset.range 181)).card

/-- Theorem stating that the number of positive factors of 180 that are also multiples of 15 is 6 -/
theorem num_factors_180_multiple_15_eq_6 : num_factors_180_multiple_15 = 6 := by
  sorry

end num_factors_180_multiple_15_eq_6_l135_13561


namespace simplify_expression_l135_13587

theorem simplify_expression (y : ℝ) : 3 * y + 4 * y + 5 * y + 7 = 12 * y + 7 := by
  sorry

end simplify_expression_l135_13587


namespace cube_surface_area_equal_volume_l135_13527

/-- The surface area of a cube with the same volume as a rectangular prism -/
theorem cube_surface_area_equal_volume (l w h : ℝ) (cube_sa : ℝ) : 
  l = 12 → w = 3 → h = 24 → 
  cube_sa = 6 * (l * w * h) ^ (2/3) →
  cube_sa = 545.02 := by
sorry

end cube_surface_area_equal_volume_l135_13527


namespace bills_problem_l135_13528

/-- Represents the bill amount for a person -/
structure Bill where
  amount : ℝ
  tipPercentage : ℝ
  tipAmount : ℝ

/-- The problem statement -/
theorem bills_problem (mike : Bill) (joe : Bill) (bill : Bill)
  (h_mike : mike.tipPercentage = 0.10 ∧ mike.tipAmount = 3)
  (h_joe : joe.tipPercentage = 0.15 ∧ joe.tipAmount = 4.5)
  (h_bill : bill.tipPercentage = 0.25 ∧ bill.tipAmount = 5) :
  bill.amount = 20 := by
  sorry


end bills_problem_l135_13528


namespace factor_implies_b_value_l135_13586

/-- The polynomial Q(x) = x^3 + 3x^2 + bx + 20 -/
def Q (b : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + b*x + 20

/-- Theorem: If x - 4 is a factor of Q(x), then b = -33 -/
theorem factor_implies_b_value (b : ℝ) :
  (∀ x, Q b x = 0 ↔ x = 4) → b = -33 := by
  sorry

end factor_implies_b_value_l135_13586


namespace wire_length_ratio_l135_13598

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the total wire length needed for a cuboid frame -/
def wireLength (d : CuboidDimensions) : ℝ :=
  4 * (d.length + d.width + d.height)

theorem wire_length_ratio : 
  let bonnie := CuboidDimensions.mk 8 10 10
  let roark := CuboidDimensions.mk 1 2 2
  let bonnieVolume := cuboidVolume bonnie
  let roarkVolume := cuboidVolume roark
  let numRoarkCuboids := bonnieVolume / roarkVolume
  let bonnieWire := wireLength bonnie
  let roarkTotalWire := numRoarkCuboids * wireLength roark
  bonnieWire / roarkTotalWire = 9 / 250 := by
  sorry

end wire_length_ratio_l135_13598


namespace largest_square_advertisement_l135_13570

theorem largest_square_advertisement (rectangle_width rectangle_length min_border : Real)
  (h1 : rectangle_width = 9)
  (h2 : rectangle_length = 16)
  (h3 : min_border = 1.5)
  (h4 : rectangle_width ≤ rectangle_length) :
  let max_side := min (rectangle_width - 2 * min_border) (rectangle_length - 2 * min_border)
  (max_side * max_side) = 36 := by
  sorry

#check largest_square_advertisement

end largest_square_advertisement_l135_13570


namespace probability_three_white_balls_l135_13565

def total_balls : ℕ := 11
def white_balls : ℕ := 4
def black_balls : ℕ := 7
def drawn_balls : ℕ := 3

theorem probability_three_white_balls : 
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 4 / 165 := by
  sorry

end probability_three_white_balls_l135_13565


namespace triangle_construction_l135_13510

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define properties
def isAcute (t : Triangle) : Prop := sorry

def onCircumcircle (p : Point) (t : Triangle) : Prop := sorry

def isAltitude (l : Point → Point → Prop) (t : Triangle) : Prop := sorry

def isAngleBisector (l : Point → Point → Prop) (t : Triangle) : Prop := sorry

def intersectsCircumcircle (l : Point → Point → Prop) (t : Triangle) : Point := sorry

-- Main theorem
theorem triangle_construction (t' : Triangle) (h_acute : isAcute t') :
  ∃ t : Triangle,
    (∀ p : Point, onCircumcircle p t' ↔ onCircumcircle p t) ∧
    isAcute t ∧
    (∀ l, isAltitude l t' → onCircumcircle (intersectsCircumcircle l t') t) ∧
    (∀ l, isAngleBisector l t → onCircumcircle (intersectsCircumcircle l t) t) :=
sorry

end triangle_construction_l135_13510


namespace equation_transformation_l135_13500

theorem equation_transformation (x y : ℝ) (hx : x ≠ 0) :
  y = x + 1/x →
  (x^4 + x^3 - 5*x^2 + x + 1 = 0) ↔ (x^2*(y^2 + y - 7) = 0) := by
  sorry

end equation_transformation_l135_13500


namespace A_infinite_l135_13595

/-- τ(n) denotes the number of positive divisors of the positive integer n -/
def tau (n : ℕ+) : ℕ := sorry

/-- The set of positive integers a for which τ(an) = n has no positive integer solutions n -/
def A : Set ℕ+ := {a | ∀ n : ℕ+, tau (a * n) ≠ n}

/-- Theorem: The set A is infinite -/
theorem A_infinite : Set.Infinite A := by sorry

end A_infinite_l135_13595


namespace circumcircle_area_of_triangle_ABP_l135_13553

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define points
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define the condition |AP⃗|cos<AP⃗, AF₂⃗> = |AF₂⃗|
def condition_AP_AF₂ (P : ℝ × ℝ) : Prop :=
  let AP := (P.1 - A.1, P.2 - A.2)
  let AF₂ := (F₂.1 - A.1, F₂.2 - A.2)
  Real.sqrt (AP.1^2 + AP.2^2) * (AP.1 * AF₂.1 + AP.2 * AF₂.2) / 
    (Real.sqrt (AP.1^2 + AP.2^2) * Real.sqrt (AF₂.1^2 + AF₂.2^2)) = 
    Real.sqrt (AF₂.1^2 + AF₂.2^2)

-- Define the theorem
theorem circumcircle_area_of_triangle_ABP (P : ℝ × ℝ) 
  (h₁ : hyperbola P.1 P.2)
  (h₂ : P.1 > B.1)  -- P is on the right branch
  (h₃ : condition_AP_AF₂ P) :
  ∃ (R : ℝ), R > 0 ∧ π * R^2 = 5 * π := by sorry

end circumcircle_area_of_triangle_ABP_l135_13553


namespace symmetric_points_fourth_quadrant_l135_13539

/-- Given points A(a, 3) and B(2, b) are symmetric with respect to the x-axis,
    prove that point M(a, b) is in the fourth quadrant. -/
theorem symmetric_points_fourth_quadrant (a b : ℝ) :
  (a = 2 ∧ b = -3) →  -- Symmetry conditions
  a > 0 ∧ b < 0       -- Fourth quadrant conditions
  := by sorry

end symmetric_points_fourth_quadrant_l135_13539


namespace function_extrema_implies_a_range_l135_13513

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

theorem function_extrema_implies_a_range (a : ℝ) :
  (∃ (x_max x_min : ℝ), ∀ x, f a x ≤ f a x_max ∧ f a x_min ≤ f a x) →
  a < -3 ∨ a > 6 :=
by sorry

end function_extrema_implies_a_range_l135_13513


namespace rabbit_hit_probability_l135_13585

/-- The probability that at least one hunter hits the rabbit. -/
def probability_hit (p1 p2 p3 : ℝ) : ℝ :=
  1 - (1 - p1) * (1 - p2) * (1 - p3)

/-- Theorem: Given three hunters with hit probabilities 0.6, 0.5, and 0.4,
    the probability that the rabbit is hit is 0.88. -/
theorem rabbit_hit_probability :
  probability_hit 0.6 0.5 0.4 = 0.88 := by
  sorry

end rabbit_hit_probability_l135_13585


namespace bathroom_width_l135_13545

/-- The width of a rectangular bathroom with length 4 feet and area 8 square feet is 2 feet. -/
theorem bathroom_width (length : ℝ) (area : ℝ) (width : ℝ) 
    (h1 : length = 4)
    (h2 : area = 8)
    (h3 : area = length * width) : width = 2 := by
  sorry

end bathroom_width_l135_13545


namespace poem_word_count_l135_13524

/-- Given a poem with the specified structure, prove that the total number of words is 1600. -/
theorem poem_word_count (stanzas : ℕ) (lines_per_stanza : ℕ) (words_per_line : ℕ)
  (h1 : stanzas = 20)
  (h2 : lines_per_stanza = 10)
  (h3 : words_per_line = 8) :
  stanzas * lines_per_stanza * words_per_line = 1600 := by
  sorry


end poem_word_count_l135_13524


namespace family_spending_proof_l135_13589

def planned_spending (family_size : ℕ) (orange_cost : ℚ) (savings_percentage : ℚ) : ℚ :=
  (family_size : ℚ) * orange_cost / (savings_percentage / 100)

theorem family_spending_proof :
  let family_size : ℕ := 4
  let orange_cost : ℚ := 3/2
  let savings_percentage : ℚ := 40
  planned_spending family_size orange_cost savings_percentage = 15 := by
sorry

end family_spending_proof_l135_13589


namespace polynomial_division_theorem_l135_13523

theorem polynomial_division_theorem (x : ℝ) : 
  (x^4 + 13) = (x - 1) * (x^3 + x^2 + x + 1) + 14 := by
  sorry

end polynomial_division_theorem_l135_13523


namespace inequality_proof_l135_13594

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1 + a)^4 * (1 + b)^4 ≥ 64 * a * b * (a + b)^2 := by
  sorry

end inequality_proof_l135_13594


namespace complex_magnitude_problem_l135_13571

theorem complex_magnitude_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 30)
  (h2 : Complex.abs (z + 3 * w) = 6)
  (h3 : Complex.abs (z - w) = 3) :
  Complex.abs z = Real.sqrt 132.525 := by sorry

end complex_magnitude_problem_l135_13571


namespace product_units_digit_l135_13529

def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def units_digit (n : ℕ) : ℕ := n % 10

theorem product_units_digit :
  is_composite 4 ∧ is_composite 6 ∧ is_composite 9 →
  units_digit (4 * 6 * 9) = 6 := by sorry

end product_units_digit_l135_13529


namespace correct_scientific_notation_l135_13567

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def number : ℕ := 250000

/-- The scientific notation representation of the number -/
def scientific_representation : ScientificNotation := {
  coefficient := 2.5,
  exponent := 5,
  coefficient_range := by sorry
}

/-- Theorem stating that the scientific notation representation is correct -/
theorem correct_scientific_notation :
  (scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent) = number := by
  sorry

end correct_scientific_notation_l135_13567


namespace three_digit_same_divisible_by_37_l135_13540

theorem three_digit_same_divisible_by_37 (a : ℕ) (h : a ≤ 9) :
  ∃ k : ℕ, 111 * a = 37 * k := by
  sorry

end three_digit_same_divisible_by_37_l135_13540


namespace sum_of_two_numbers_l135_13593

theorem sum_of_two_numbers (x y : ℝ) : 
  (x + y) + (x - y) = 8 → x^2 - y^2 = 160 → x + y = 8 := by
  sorry

end sum_of_two_numbers_l135_13593


namespace jason_initial_cards_l135_13580

theorem jason_initial_cards (initial_cards final_cards bought_cards : ℕ) 
  (h1 : bought_cards = 224)
  (h2 : final_cards = 900)
  (h3 : final_cards = initial_cards + bought_cards) :
  initial_cards = 676 := by
  sorry

end jason_initial_cards_l135_13580


namespace even_function_quadratic_l135_13577

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

theorem even_function_quadratic 
  (a b : ℝ) 
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_domain : Set.Icc (-1 - a) (2 * a) ⊆ Set.range (f a b)) :
  f a b (2 * a - b) = 5 := by
sorry

end even_function_quadratic_l135_13577


namespace largest_integer_a_l135_13507

theorem largest_integer_a : ∃ (a : ℤ), 
  (∀ x : ℝ, -π/2 < x ∧ x < π/2 → 
    a^2 - 15*a - (Real.tan x - 1)*(Real.tan x + 2)*(Real.tan x + 5)*(Real.tan x + 8) < 35) ∧ 
  (∀ b : ℤ, b > a → 
    ∃ x : ℝ, -π/2 < x ∧ x < π/2 ∧ 
      b^2 - 15*b - (Real.tan x - 1)*(Real.tan x + 2)*(Real.tan x + 5)*(Real.tan x + 8) ≥ 35) ∧
  a = 10 :=
sorry

end largest_integer_a_l135_13507


namespace right_triangle_condition_l135_13563

/-- If in a triangle ABC, angle A equals the sum of angles B and C, then angle A is a right angle -/
theorem right_triangle_condition (A B C : Real) (h1 : A + B + C = Real.pi) (h2 : A = B + C) : A = Real.pi / 2 := by
  sorry

end right_triangle_condition_l135_13563


namespace smallest_prime_factor_of_2145_l135_13506

theorem smallest_prime_factor_of_2145 : Nat.minFac 2145 = 3 := by
  sorry

end smallest_prime_factor_of_2145_l135_13506


namespace distribute_six_balls_three_boxes_l135_13522

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ := num_boxes ^ num_balls

/-- Theorem: There are 729 ways to distribute 6 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 729 := by
  sorry

end distribute_six_balls_three_boxes_l135_13522


namespace hyperbola_eccentricity_l135_13533

/-- Given a hyperbola C: mx^2 + ny^2 = 1 (m > 0, n < 0) with one of its asymptotes
    tangent to the circle x^2 + y^2 - 6x - 2y + 9 = 0, 
    the eccentricity of C is 5/4. -/
theorem hyperbola_eccentricity (m n : ℝ) (hm : m > 0) (hn : n < 0) :
  let C := {(x, y) : ℝ × ℝ | m * x^2 + n * y^2 = 1}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 6*x - 2*y + 9 = 0}
  let asymptote := {(x, y) : ℝ × ℝ | Real.sqrt m * x - Real.sqrt (-n) * y = 0}
  (∃ (p : ℝ × ℝ), p ∈ asymptote ∧ p ∈ circle) →
  let a := 1 / Real.sqrt m
  let b := 1 / Real.sqrt (-n)
  let e := Real.sqrt (1 + (b/a)^2)
  e = 5/4 := by
sorry

end hyperbola_eccentricity_l135_13533


namespace f_increasing_on_interval_l135_13552

open Real

noncomputable def f (x : ℝ) : ℝ := -log (x^2 - 3*x + 2)

theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Iio 1) := by sorry

end f_increasing_on_interval_l135_13552


namespace pencil_length_after_sharpening_l135_13504

def initial_length : ℕ := 100
def monday_sharpening : ℕ := 3
def tuesday_sharpening : ℕ := 5
def wednesday_sharpening : ℕ := 7
def thursday_sharpening : ℕ := 11
def friday_sharpening : ℕ := 13

theorem pencil_length_after_sharpening : 
  initial_length - (monday_sharpening + tuesday_sharpening + wednesday_sharpening + thursday_sharpening + friday_sharpening) = 61 := by
  sorry

end pencil_length_after_sharpening_l135_13504


namespace range_of_a_minus_b_l135_13559

-- Define the quadratic function
def f (a b x : ℝ) : ℝ := x^2 + a*x + (b - 2)

-- State the theorem
theorem range_of_a_minus_b (a b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < -1 ∧ -1 < x₂ ∧ x₂ < 0 ∧ f a b x₁ = 0 ∧ f a b x₂ = 0) →
  ∀ y : ℝ, y > -1 → ∃ a' b' : ℝ, a' - b' = y ∧
    ∃ x₁ x₂ : ℝ, x₁ < -1 ∧ -1 < x₂ ∧ x₂ < 0 ∧ f a' b' x₁ = 0 ∧ f a' b' x₂ = 0 :=
by sorry

end range_of_a_minus_b_l135_13559


namespace greatest_divisor_with_remainders_l135_13534

theorem greatest_divisor_with_remainders : Nat.gcd (28572 - 142) (39758 - 84) = 2 := by
  sorry

end greatest_divisor_with_remainders_l135_13534


namespace ratio_b_to_c_l135_13501

theorem ratio_b_to_c (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : c / d = 1 / 2)
  (h3 : d / e = 3)
  (h4 : e / f = 1 / 10)
  (h5 : a * b * c / (d * e * f) = 0.15) :
  b / c = 9 := by
sorry

end ratio_b_to_c_l135_13501


namespace oranges_picked_sum_l135_13502

/-- Given the number of oranges picked by Joan and Sara, prove that their sum
    equals the total number of oranges picked. -/
theorem oranges_picked_sum (joan_oranges sara_oranges total_oranges : ℕ)
  (h1 : joan_oranges = 37)
  (h2 : sara_oranges = 10)
  (h3 : total_oranges = 47) :
  joan_oranges + sara_oranges = total_oranges :=
by sorry

end oranges_picked_sum_l135_13502


namespace marbles_lost_calculation_l135_13508

def initial_marbles : ℕ := 15
def marbles_found : ℕ := 9
def extra_marbles_lost : ℕ := 14

theorem marbles_lost_calculation :
  marbles_found + extra_marbles_lost = 23 :=
by sorry

end marbles_lost_calculation_l135_13508


namespace max_d_value_l135_13526

def a (n : ℕ+) : ℕ := 120 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  (∃ (n : ℕ+), d n = 481) ∧ (∀ (n : ℕ+), d n ≤ 481) :=
sorry

end max_d_value_l135_13526


namespace log_pieces_after_ten_cuts_l135_13519

/-- The number of pieces obtained after cutting a log -/
def numPieces (cuts : ℕ) : ℕ := cuts + 1

/-- Theorem: The number of pieces obtained after 10 cuts on a log is 11 -/
theorem log_pieces_after_ten_cuts : numPieces 10 = 11 := by
  sorry

end log_pieces_after_ten_cuts_l135_13519
