import Mathlib

namespace theater_line_permutations_l1856_185635

theorem theater_line_permutations : Nat.factorial 8 = 40320 := by
  sorry

end theater_line_permutations_l1856_185635


namespace imaginary_part_of_1_minus_2i_l1856_185629

theorem imaginary_part_of_1_minus_2i :
  Complex.im (1 - 2 * Complex.I) = -2 := by sorry

end imaginary_part_of_1_minus_2i_l1856_185629


namespace alice_probability_after_three_turns_l1856_185664

-- Define the probabilities
def alice_to_bob : ℚ := 2/3
def alice_keeps : ℚ := 1/3
def bob_to_alice : ℚ := 1/3
def bob_keeps : ℚ := 2/3

-- Define the game state after three turns
def alice_has_ball_after_three_turns : ℚ :=
  alice_to_bob * bob_to_alice +
  alice_to_bob * bob_keeps * bob_to_alice +
  alice_keeps * alice_to_bob * bob_to_alice +
  alice_keeps * alice_keeps

-- Theorem statement
theorem alice_probability_after_three_turns :
  alice_has_ball_after_three_turns = 5/9 := by
  sorry

end alice_probability_after_three_turns_l1856_185664


namespace complex_equation_solution_l1856_185642

theorem complex_equation_solution :
  ∃ z : ℂ, (4 - 3 * Complex.I * z = 1 + 5 * Complex.I * z) ∧ (z = -3/8 * Complex.I) := by
  sorry

end complex_equation_solution_l1856_185642


namespace number_of_digits_in_x_l1856_185630

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the problem statement
theorem number_of_digits_in_x (x y : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (x_gt_y : x > y)
  (prod_xy : x * y = 490)
  (log_eq : (log10 x - log10 7) * (log10 y - log10 7) = -143/4) :
  ∃ n : ℕ, n = 8 ∧ 10^(n-1) ≤ x ∧ x < 10^n := by
sorry

end number_of_digits_in_x_l1856_185630


namespace olaf_total_cars_l1856_185618

/-- The number of toy cars in Olaf's collection --/
def total_cars (initial : ℕ) (grandpa uncle dad mum auntie : ℕ) : ℕ :=
  initial + grandpa + uncle + dad + mum + auntie

/-- The conditions of Olaf's toy car collection problem --/
def olaf_problem (initial grandpa uncle dad mum auntie : ℕ) : Prop :=
  initial = 150 ∧
  grandpa = 2 * uncle ∧
  dad = 10 ∧
  mum = dad + 5 ∧
  auntie = uncle + 1 ∧
  auntie = 6

/-- Theorem stating that Olaf's total number of cars is 196 --/
theorem olaf_total_cars :
  ∀ initial grandpa uncle dad mum auntie : ℕ,
  olaf_problem initial grandpa uncle dad mum auntie →
  total_cars initial grandpa uncle dad mum auntie = 196 :=
by
  sorry


end olaf_total_cars_l1856_185618


namespace second_train_length_second_train_length_problem_l1856_185637

/-- Calculates the length of the second train given the speeds of two trains moving in opposite directions, the length of the first train, and the time taken to cross each other. -/
theorem second_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (length1 : ℝ) 
  (crossing_time : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let total_distance := relative_speed * crossing_time / 3600
  total_distance - length1

/-- The length of the second train is 0.9 km given the specified conditions -/
theorem second_train_length_problem : 
  second_train_length 60 90 1.10 47.99999999999999 = 0.9 := by
  sorry

end second_train_length_second_train_length_problem_l1856_185637


namespace smallest_m_no_real_solutions_l1856_185689

theorem smallest_m_no_real_solutions : ∃ (m : ℕ), 
  (m > 0) ∧ 
  (∀ (x : ℝ), m * x^2 - 3 * x + 1 ≠ 0) ∧
  (∀ (k : ℕ), k > 0 → k < m → ∃ (x : ℝ), k * x^2 - 3 * x + 1 = 0) ∧
  m = 3 := by
  sorry

end smallest_m_no_real_solutions_l1856_185689


namespace bench_press_increase_factor_l1856_185694

theorem bench_press_increase_factor 
  (initial_weight : ℝ) 
  (injury_decrease_percent : ℝ) 
  (final_weight : ℝ) 
  (h1 : initial_weight = 500) 
  (h2 : injury_decrease_percent = 80) 
  (h3 : final_weight = 300) : 
  final_weight / (initial_weight * (1 - injury_decrease_percent / 100)) = 3 := by
sorry

end bench_press_increase_factor_l1856_185694


namespace inscribed_circle_radius_bounds_l1856_185612

/-- Given a triangle with sides a ≤ b ≤ c and corresponding altitudes ma ≥ mb ≥ mc,
    the radius ρ of the inscribed circle satisfies mc/3 ≤ ρ ≤ ma/3 -/
theorem inscribed_circle_radius_bounds (a b c ma mb mc ρ : ℝ) 
  (h_sides : a ≤ b ∧ b ≤ c)
  (h_altitudes : ma ≥ mb ∧ mb ≥ mc)
  (h_inradius : ρ > 0)
  (h_area : ρ * (a + b + c) = a * ma)
  (h_area_alt : ρ * (a + b + c) = c * mc) :
  mc / 3 ≤ ρ ∧ ρ ≤ ma / 3 := by
  sorry

end inscribed_circle_radius_bounds_l1856_185612


namespace sqrt_32_minus_cos_45_plus_one_minus_sqrt_2_squared_l1856_185638

theorem sqrt_32_minus_cos_45_plus_one_minus_sqrt_2_squared :
  Real.sqrt 32 - Real.cos (π / 4) + (1 - Real.sqrt 2) ^ 2 = 3 + (3 / 2) * Real.sqrt 2 := by
  sorry

end sqrt_32_minus_cos_45_plus_one_minus_sqrt_2_squared_l1856_185638


namespace max_value_sum_of_roots_l1856_185696

theorem max_value_sum_of_roots (a b c : ℝ) (h : a + b + c = 1) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 2 ∧
  (Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) ≤ max) ∧
  (∃ (a₀ b₀ c₀ : ℝ), a₀ + b₀ + c₀ = 1 ∧
    Real.sqrt (3 * a₀ + 1) + Real.sqrt (3 * b₀ + 1) + Real.sqrt (3 * c₀ + 1) = max) :=
by sorry

end max_value_sum_of_roots_l1856_185696


namespace simplify_and_evaluate_l1856_185647

theorem simplify_and_evaluate (x : ℝ) :
  x = -2 →
  (1 - 2 / (2 - x)) / (x / (x^2 - 4*x + 4)) = x - 2 ∧
  x - 2 = -4 :=
by sorry

end simplify_and_evaluate_l1856_185647


namespace remainder_nine_333_mod_50_l1856_185665

theorem remainder_nine_333_mod_50 : 9^333 % 50 = 29 := by
  sorry

end remainder_nine_333_mod_50_l1856_185665


namespace system_solution_l1856_185623

theorem system_solution : 
  ∀ x y : ℝ, x + y = 3 ∧ x^5 + y^5 = 33 → (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 1) := by
  sorry

end system_solution_l1856_185623


namespace g_fixed_points_l1856_185608

def g (x : ℝ) : ℝ := x^2 - 5*x

theorem g_fixed_points (x : ℝ) : g (g x) = g x ↔ x = -1 ∨ x = 0 ∨ x = 5 ∨ x = 6 := by
  sorry

end g_fixed_points_l1856_185608


namespace carla_liquid_consumption_l1856_185653

/-- The amount of water Carla drank in ounces -/
def water : ℝ := 15

/-- The amount of soda Carla drank in ounces -/
def soda : ℝ := 3 * water - 6

/-- The total amount of liquid Carla drank in ounces -/
def total_liquid : ℝ := water + soda

/-- Theorem stating the total amount of liquid Carla drank -/
theorem carla_liquid_consumption : total_liquid = 54 := by
  sorry

end carla_liquid_consumption_l1856_185653


namespace min_overlap_mozart_bach_l1856_185620

theorem min_overlap_mozart_bach (total : ℕ) (mozart : ℕ) (bach : ℕ) 
  (h_total : total = 200)
  (h_mozart : mozart = 160)
  (h_bach : bach = 145)
  : mozart + bach - total ≥ 105 := by
  sorry

end min_overlap_mozart_bach_l1856_185620


namespace cipher_decoding_probabilities_l1856_185660

-- Define the probabilities of success for each person
def p_A : ℝ := 0.4
def p_B : ℝ := 0.35
def p_C : ℝ := 0.3

-- Define the probability of exactly two successes
def prob_two_successes : ℝ :=
  p_A * p_B * (1 - p_C) + p_A * (1 - p_B) * p_C + (1 - p_A) * p_B * p_C

-- Define the probability of at least one success
def prob_at_least_one_success : ℝ :=
  1 - (1 - p_A) * (1 - p_B) * (1 - p_C)

-- Theorem statement
theorem cipher_decoding_probabilities :
  prob_two_successes = 0.239 ∧ prob_at_least_one_success = 0.727 := by
  sorry

end cipher_decoding_probabilities_l1856_185660


namespace repeating_decimal_denominator_l1856_185654

theorem repeating_decimal_denominator : ∃ (n : ℕ) (d : ℕ), d ≠ 0 ∧ (n / d : ℚ) = 0.6666666666666667 ∧ (∀ (n' : ℕ) (d' : ℕ), d' ≠ 0 → (n' / d' : ℚ) = (n / d : ℚ) → d' ≥ d) ∧ d = 3 :=
by sorry

end repeating_decimal_denominator_l1856_185654


namespace ball_diameter_proof_l1856_185651

theorem ball_diameter_proof (h s d : ℝ) (h_pos : h > 0) (s_pos : s > 0) (d_pos : d > 0) :
  h / s = (h / s) / (1 + d / (h / s)) → h / s = 1.25 → s = 1 → d = 0.23 → h / s = 0.23 :=
by sorry

end ball_diameter_proof_l1856_185651


namespace smallest_integer_problem_l1856_185607

theorem smallest_integer_problem (a b c : ℕ+) : 
  (a : ℝ) + b + c = 90 ∧ 
  2 * a = 3 * b ∧ 
  2 * a = 5 * c ∧ 
  (a : ℝ) * b * c < 22000 → 
  a = 18 := by
sorry

end smallest_integer_problem_l1856_185607


namespace divisibility_property_l1856_185658

theorem divisibility_property (n a b c d : ℤ) 
  (hn : n > 0)
  (h1 : n ∣ (a + b + c + d))
  (h2 : n ∣ (a^2 + b^2 + c^2 + d^2)) :
  n ∣ (a^4 + b^4 + c^4 + d^4 + 4*a*b*c*d) := by
  sorry

end divisibility_property_l1856_185658


namespace polynomial_existence_l1856_185685

theorem polynomial_existence : 
  ∃ (P : ℝ → ℝ → ℝ → ℝ), ∀ (t : ℝ), P (t^1993) (t^1994) (t + t^1995) = t := by
  sorry

end polynomial_existence_l1856_185685


namespace unique_divisor_l1856_185666

def sum_even_two_digit : Nat := 2430

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

def digits_sum (n : Nat) : Nat := (n / 10) + (n % 10)

def reverse_digits (n : Nat) : Nat := (n % 10) * 10 + (n / 10)

theorem unique_divisor :
  ∃! n : Nat, is_two_digit n ∧ 
    sum_even_two_digit % n = 0 ∧
    sum_even_two_digit / n = reverse_digits n ∧
    digits_sum (sum_even_two_digit / n) = 9 :=
by sorry

end unique_divisor_l1856_185666


namespace not_p_necessary_not_sufficient_for_not_q_implies_m_ge_11_div_2_l1856_185613

/-- Proposition p: -x^2 + 8x + 20 ≥ 0 -/
def p (x : ℝ) : Prop := -x^2 + 8*x + 20 ≥ 0

/-- Proposition q: x^2 + 2x + 1 - 4m^2 ≤ 0 -/
def q (x m : ℝ) : Prop := x^2 + 2*x + 1 - 4*m^2 ≤ 0

/-- If ¬p is a necessary but not sufficient condition for ¬q when m > 0, then m ≥ 11/2 -/
theorem not_p_necessary_not_sufficient_for_not_q_implies_m_ge_11_div_2 :
  ∀ m : ℝ, m > 0 →
  (∀ x : ℝ, (¬q x m → ¬p x) ∧ (∃ x : ℝ, ¬p x ∧ q x m)) →
  m ≥ 11/2 :=
sorry

end not_p_necessary_not_sufficient_for_not_q_implies_m_ge_11_div_2_l1856_185613


namespace range_of_m_l1856_185604

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x ∈ Set.Icc (-1) m ∧ m > -1 ∧ |x| - 1 > 0) → 
  m ∈ Set.Ioo (-1) 1 := by
sorry

end range_of_m_l1856_185604


namespace tank_dimension_l1856_185650

/-- Given a rectangular tank with dimensions 3 feet, 7 feet, and x feet,
    if the total surface area is 82 square feet, then x = 2 feet. -/
theorem tank_dimension (x : ℝ) : 
  2 * (3 * 7 + 3 * x + 7 * x) = 82 → x = 2 := by
  sorry

end tank_dimension_l1856_185650


namespace sqrt_equation_solution_l1856_185675

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (9 - 2 * x) = 8 → x = -55 / 2 := by
  sorry

end sqrt_equation_solution_l1856_185675


namespace meeting_point_distance_l1856_185602

/-- Proves that the distance between Jack and Jill's meeting point and the hilltop is 35/27 km -/
theorem meeting_point_distance (total_distance : ℝ) (uphill_distance : ℝ)
  (jack_start_earlier : ℝ) (jack_uphill_speed : ℝ) (jack_downhill_speed : ℝ)
  (jill_uphill_speed : ℝ) :
  total_distance = 10 →
  uphill_distance = 5 →
  jack_start_earlier = 1/6 →
  jack_uphill_speed = 15 →
  jack_downhill_speed = 20 →
  jill_uphill_speed = 16 →
  ∃ (meeting_point_distance : ℝ), meeting_point_distance = 35/27 := by
  sorry

end meeting_point_distance_l1856_185602


namespace corner_removed_cube_surface_area_l1856_185684

/-- Represents a cube with corner cubes removed -/
structure CornerRemovedCube where
  side_length : ℝ
  corner_size : ℝ

/-- Calculates the surface area of a cube with corner cubes removed -/
def surface_area (cube : CornerRemovedCube) : ℝ :=
  6 * cube.side_length^2

/-- Theorem stating that a 4x4x4 cube with corner cubes removed has surface area 96 sq.cm -/
theorem corner_removed_cube_surface_area :
  let cube : CornerRemovedCube := ⟨4, 1⟩
  surface_area cube = 96 := by
  sorry

end corner_removed_cube_surface_area_l1856_185684


namespace a1_plus_a3_equals_24_l1856_185670

theorem a1_plus_a3_equals_24 (x : ℝ) (a₀ a₁ a₂ a₃ a₄ : ℝ) 
  (h : (1 - 2/x)^4 = a₀ + a₁*(1/x) + a₂*(1/x)^2 + a₃*(1/x)^3 + a₄*(1/x)^4) :
  a₁ + a₃ = 24 := by
sorry

end a1_plus_a3_equals_24_l1856_185670


namespace students_favoring_both_issues_l1856_185683

/-- The number of students who voted in favor of both issues in a school referendum -/
theorem students_favoring_both_issues 
  (total_students : ℕ) 
  (favor_first : ℕ) 
  (favor_second : ℕ) 
  (against_both : ℕ) 
  (h1 : total_students = 215)
  (h2 : favor_first = 160)
  (h3 : favor_second = 132)
  (h4 : against_both = 40) : 
  favor_first + favor_second - (total_students - against_both) = 117 :=
by sorry

end students_favoring_both_issues_l1856_185683


namespace b_range_l1856_185682

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < -1 then (x + 1) / x^2 else Real.log (x + 2)

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 2*x - 4

-- State the theorem
theorem b_range (b : ℝ) :
  (∃ a : ℝ, f a + g b = 1) → b ∈ Set.Icc (-3/2) (7/2) :=
by sorry

end b_range_l1856_185682


namespace father_age_is_27_l1856_185640

def father_son_ages (father_age son_age : ℕ) : Prop :=
  (father_age = 3 * son_age + 3) ∧
  (father_age + 3 = 2 * (son_age + 3) + 8)

theorem father_age_is_27 :
  ∃ (son_age : ℕ), father_son_ages 27 son_age :=
sorry

end father_age_is_27_l1856_185640


namespace line_parallel_from_perpendicular_to_parallel_planes_l1856_185643

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Define the theorem
theorem line_parallel_from_perpendicular_to_parallel_planes
  (m n : Line) (α β : Plane)
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_m_perp_α : perpendicular m α)
  (h_n_perp_β : perpendicular n β)
  (h_α_parallel_β : plane_parallel α β) :
  parallel m n :=
sorry

end line_parallel_from_perpendicular_to_parallel_planes_l1856_185643


namespace nearest_integer_to_power_l1856_185669

theorem nearest_integer_to_power : 
  ∃ (n : ℤ), n = 2654 ∧ 
  ∀ (m : ℤ), |((3 : ℝ) + Real.sqrt 5)^6 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5)^6 - (m : ℝ)| :=
by sorry

end nearest_integer_to_power_l1856_185669


namespace cubic_inequality_l1856_185631

theorem cubic_inequality (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  (a^3 + b^3 + c^3) / 3 ≥ a * b * c + a + b + c ∧
  ((a^3 + b^3 + c^3) / 3 = a * b * c + a + b + c ↔ 
   (b = a + 1 ∧ c = b + 1) ∨ (a = b + 1 ∧ c = a + 1) ∨ (a = c + 1 ∧ b = a + 1)) :=
sorry

end cubic_inequality_l1856_185631


namespace parabola_properties_l1856_185626

/-- A parabola with focus on a given line -/
structure Parabola where
  p : ℝ
  focus_on_line : (p / 2) + (0 : ℝ) - 2 = 0

/-- The directrix of a parabola -/
def directrix (C : Parabola) : ℝ → Prop :=
  λ x => x = -(C.p / 2)

theorem parabola_properties (C : Parabola) :
  C.p = 4 ∧ directrix C = λ x => x = -2 := by
  sorry

end parabola_properties_l1856_185626


namespace plums_picked_equals_127_l1856_185695

/-- Calculates the total number of plums picked by Alyssa and Jason after three hours -/
def total_plums_picked (alyssa_rate : ℕ) (jason_rate : ℕ) : ℕ :=
  let first_hour := alyssa_rate + jason_rate
  let second_hour := (3 * alyssa_rate) + (jason_rate + (2 * jason_rate / 5))
  let third_hour_before_drop := alyssa_rate + (2 * jason_rate)
  let third_hour_after_drop := third_hour_before_drop - (third_hour_before_drop / 14)
  first_hour + second_hour + third_hour_after_drop

/-- Theorem stating that the total number of plums picked is 127 -/
theorem plums_picked_equals_127 :
  total_plums_picked 17 10 = 127 := by
  sorry

#eval total_plums_picked 17 10

end plums_picked_equals_127_l1856_185695


namespace bus_ticket_solution_l1856_185691

/-- Represents the number and cost of bus tickets -/
structure BusTickets where
  total_tickets : ℕ
  total_cost : ℕ
  one_way_cost : ℕ
  round_trip_cost : ℕ

/-- Theorem stating the correct number of one-way and round-trip tickets -/
theorem bus_ticket_solution (tickets : BusTickets)
  (h1 : tickets.total_tickets = 99)
  (h2 : tickets.total_cost = 280)
  (h3 : tickets.one_way_cost = 2)
  (h4 : tickets.round_trip_cost = 3) :
  ∃ (one_way round_trip : ℕ),
    one_way + round_trip = tickets.total_tickets ∧
    one_way * tickets.one_way_cost + round_trip * tickets.round_trip_cost = tickets.total_cost ∧
    one_way = 17 ∧
    round_trip = 82 := by
  sorry

end bus_ticket_solution_l1856_185691


namespace geometric_sum_remainder_l1856_185673

theorem geometric_sum_remainder (n : ℕ) : 
  (((5^(n+1) - 1) / 4) % 500 = 31) ∧ (n = 1002) := by sorry

end geometric_sum_remainder_l1856_185673


namespace paper_width_is_four_l1856_185634

/-- Given a rectangular paper surrounded by a wall photo, this theorem proves
    that the width of the paper is 4 inches under certain conditions. -/
theorem paper_width_is_four 
  (photo_width : ℝ) 
  (paper_length : ℝ) 
  (photo_area : ℝ) 
  (h1 : photo_width = 2)
  (h2 : paper_length = 8)
  (h3 : photo_area = 96)
  (h4 : photo_area = (paper_length + 2 * photo_width) * (paper_width + 2 * photo_width)) :
  paper_width = 4 :=
by
  sorry

#check paper_width_is_four

end paper_width_is_four_l1856_185634


namespace prob_green_ball_is_five_ninths_l1856_185611

structure Container where
  red : ℕ
  green : ℕ

def containers : List Container := [
  ⟨10, 5⟩,
  ⟨3, 6⟩,
  ⟨4, 8⟩
]

def total_balls (c : Container) : ℕ := c.red + c.green

def prob_green (c : Container) : ℚ :=
  c.green / (total_balls c)

theorem prob_green_ball_is_five_ninths :
  (List.sum (containers.map (λ c => (1 : ℚ) / containers.length * prob_green c))) = 5 / 9 := by
  sorry

end prob_green_ball_is_five_ninths_l1856_185611


namespace fraction_stayed_home_l1856_185649

theorem fraction_stayed_home (total : ℚ) (fun_fraction : ℚ) (youth_fraction : ℚ)
  (h1 : fun_fraction = 5 / 13)
  (h2 : youth_fraction = 4 / 13)
  (h3 : total = 1) :
  total - (fun_fraction + youth_fraction) = 4 / 13 := by
  sorry

end fraction_stayed_home_l1856_185649


namespace smallest_all_ones_divisible_by_d_is_correct_l1856_185632

def d : ℕ := 3 * (10^100 - 1) / 9

def is_all_ones (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (10^k - 1) / 9

def smallest_all_ones_divisible_by_d : ℕ := (10^300 - 1) / 9

theorem smallest_all_ones_divisible_by_d_is_correct :
  is_all_ones smallest_all_ones_divisible_by_d ∧
  smallest_all_ones_divisible_by_d % d = 0 ∧
  ∀ n : ℕ, is_all_ones n → n % d = 0 → n ≥ smallest_all_ones_divisible_by_d :=
by sorry

end smallest_all_ones_divisible_by_d_is_correct_l1856_185632


namespace max_d_value_l1856_185621

def is_valid_number (d e : Nat) : Prop :=
  d ≤ 9 ∧ e ≤ 9 ∧ (808450 + 100000 * d + e) % 45 = 0

theorem max_d_value :
  ∃ (d : Nat), is_valid_number d 2 ∧
  ∀ (d' : Nat), is_valid_number d' 2 → d' ≤ d :=
by sorry

end max_d_value_l1856_185621


namespace cradle_cup_d_score_l1856_185671

/-- Represents the scores of the five participants in the "Cradle Cup" math competition. -/
structure CradleCupScores where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ
  E : ℕ

/-- The conditions of the "Cradle Cup" math competition. -/
def CradleCupConditions (scores : CradleCupScores) : Prop :=
  scores.A = 94 ∧
  scores.B ≥ scores.A ∧ scores.B ≥ scores.C ∧ scores.B ≥ scores.D ∧ scores.B ≥ scores.E ∧
  scores.C = (scores.A + scores.D) / 2 ∧
  5 * scores.D = scores.A + scores.B + scores.C + scores.D + scores.E ∧
  scores.E = scores.C + 2 ∧
  scores.B ≤ 100 ∧ scores.C ≤ 100 ∧ scores.D ≤ 100 ∧ scores.E ≤ 100

/-- The theorem stating that given the conditions of the "Cradle Cup" math competition,
    participant D must have scored 96 points. -/
theorem cradle_cup_d_score (scores : CradleCupScores) :
  CradleCupConditions scores → scores.D = 96 :=
by sorry

end cradle_cup_d_score_l1856_185671


namespace expression_simplification_l1856_185603

theorem expression_simplification : 
  (3 * 5 * 7) / (9 * 11 * 13) * (7 * 9 * 11 * 15) / (3 * 5 * 14) = 15 / 26 := by
  sorry

end expression_simplification_l1856_185603


namespace B_power_200_l1856_185688

def B : Matrix (Fin 4) (Fin 4) ℤ :=
  !![0,0,0,1;
     1,0,0,0;
     0,1,0,0;
     0,0,1,0]

theorem B_power_200 : B ^ 200 = 1 := by sorry

end B_power_200_l1856_185688


namespace apple_baskets_proof_l1856_185668

/-- Given a total number of apples and apples per basket, calculate the number of full baskets -/
def fullBaskets (totalApples applesPerBasket : ℕ) : ℕ :=
  totalApples / applesPerBasket

theorem apple_baskets_proof :
  fullBaskets 495 25 = 19 := by
  sorry

end apple_baskets_proof_l1856_185668


namespace distance_difference_l1856_185652

def house_to_bank : ℕ := 800
def bank_to_pharmacy : ℕ := 1300
def pharmacy_to_school : ℕ := 1700

theorem distance_difference : 
  (house_to_bank + bank_to_pharmacy) - pharmacy_to_school = 400 := by
  sorry

end distance_difference_l1856_185652


namespace gmat_scores_l1856_185690

theorem gmat_scores (u v : ℝ) (hu : u > 0) (hv : v > 0) (h : v / u = 1 / 3) :
  (u + v) / 2 = (2 / 3) * u := by
  sorry

end gmat_scores_l1856_185690


namespace line_perp_to_parallel_planes_l1856_185627

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_to_parallel_planes
  (m : Line) (α β : Plane)
  (h1 : perpendicular m β)
  (h2 : parallel α β) :
  perpendicular m α :=
sorry

end line_perp_to_parallel_planes_l1856_185627


namespace max_intersections_three_circles_one_line_l1856_185615

/-- The maximum number of intersection points between circles -/
def max_circle_intersections (n : ℕ) : ℕ := n.choose 2 * 2

/-- The maximum number of intersection points between circles and a line -/
def max_circle_line_intersections (n : ℕ) : ℕ := n * 2

/-- The maximum number of intersection points for n circles and one line -/
def max_total_intersections (n : ℕ) : ℕ :=
  max_circle_intersections n + max_circle_line_intersections n

theorem max_intersections_three_circles_one_line :
  max_total_intersections 3 = 12 := by sorry

end max_intersections_three_circles_one_line_l1856_185615


namespace new_person_weight_l1856_185641

/-- Proves that if replacing a 65 kg person in a group of 4 people
    increases the average weight by 1.5 kg, then the weight of the new person is 71 kg. -/
theorem new_person_weight (initial_total : ℝ) :
  (initial_total - 65 + new_weight) / 4 = initial_total / 4 + 1.5 →
  new_weight = 71 :=
by
  sorry

end new_person_weight_l1856_185641


namespace orange_bin_problem_l1856_185662

theorem orange_bin_problem (initial : ℕ) (removed : ℕ) (added : ℕ) : 
  initial = 50 → removed = 40 → added = 24 → initial - removed + added = 34 := by
  sorry

end orange_bin_problem_l1856_185662


namespace min_value_reciprocal_sum_l1856_185693

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 3) :
  (1/a + 1/b) ≥ 1 + 2*Real.sqrt 2/3 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 3 ∧ 1/a₀ + 1/b₀ = 1 + 2*Real.sqrt 2/3 :=
sorry

end min_value_reciprocal_sum_l1856_185693


namespace equation_solution_inequalities_solution_l1856_185672

-- Part 1: Equation solution
theorem equation_solution :
  ∃ x : ℚ, (2 / (x + 3) - (x - 3) / (2*x + 6) = 1) ∧ x = 1/3 := by sorry

-- Part 2: System of inequalities solution
theorem inequalities_solution :
  ∀ x : ℚ, (2*x - 1 > 3*(x - 1) ∧ (5 - x)/2 < x + 4) ↔ (-1 < x ∧ x < 2) := by sorry

end equation_solution_inequalities_solution_l1856_185672


namespace min_value_sum_min_value_achievable_l1856_185681

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ 3 / Real.rpow 162 (1/3) :=
sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a / (3 * b) + b / (6 * c) + c / (9 * a) = 3 / Real.rpow 162 (1/3) :=
sorry

end min_value_sum_min_value_achievable_l1856_185681


namespace volunteer_distribution_theorem_l1856_185646

/-- The number of ways to distribute n people into two activities with capacity constraints -/
def distributeVolunteers (n : ℕ) (maxPerActivity : ℕ) : ℕ :=
  -- We don't implement the function here, just declare it
  sorry

/-- Theorem: The number of ways to distribute 6 people into two activities,
    where each activity can accommodate no more than 4 people, is equal to 50 -/
theorem volunteer_distribution_theorem :
  distributeVolunteers 6 4 = 50 := by
  sorry

end volunteer_distribution_theorem_l1856_185646


namespace sqrt_sum_value_l1856_185659

theorem sqrt_sum_value (a b : ℝ) (h : (Real.sqrt a + Real.sqrt b) * (Real.sqrt a + Real.sqrt b - 2) = 3) :
  Real.sqrt a + Real.sqrt b = 3 := by
  sorry

end sqrt_sum_value_l1856_185659


namespace total_is_100_l1856_185625

/-- Represents the shares of money for three individuals -/
structure Shares :=
  (a : ℚ)
  (b : ℚ)
  (c : ℚ)

/-- The conditions of the problem -/
def SatisfiesConditions (s : Shares) : Prop :=
  s.a = (1 / 4) * (s.b + s.c) ∧
  s.b = (3 / 5) * (s.a + s.c) ∧
  s.a = 20

/-- The theorem stating that the total amount is 100 -/
theorem total_is_100 (s : Shares) (h : SatisfiesConditions s) :
  s.a + s.b + s.c = 100 := by
  sorry

end total_is_100_l1856_185625


namespace geometric_mean_proof_l1856_185699

theorem geometric_mean_proof (a b : ℝ) (hb : b ≠ 0) :
  Real.sqrt ((2 * (a^2 - a*b)) / (35*b) * (10*a) / (7*(a*b - b^2))) = 2*a / (7*b) := by
  sorry

end geometric_mean_proof_l1856_185699


namespace set_operation_result_l1856_185614

def A : Set ℕ := {0, 1, 2, 4, 5, 7}
def B : Set ℕ := {1, 3, 6, 8, 9}
def C : Set ℕ := {3, 7, 8}

theorem set_operation_result : (A ∩ B) ∪ C = {1, 3, 7, 8} := by sorry

end set_operation_result_l1856_185614


namespace base_eight_subtraction_l1856_185619

/-- Represents a number in base 8 --/
def BaseEight : Type := ℕ

/-- Convert a base 8 number to decimal --/
def to_decimal (n : BaseEight) : ℕ := sorry

/-- Convert a decimal number to base 8 --/
def to_base_eight (n : ℕ) : BaseEight := sorry

/-- Subtraction in base 8 --/
def base_eight_sub (a b : BaseEight) : BaseEight := 
  to_base_eight (to_decimal a - to_decimal b)

theorem base_eight_subtraction : 
  base_eight_sub (to_base_eight 42) (to_base_eight 25) = to_base_eight 17 := by sorry

end base_eight_subtraction_l1856_185619


namespace planes_parallel_if_perpendicular_to_parallel_lines_l1856_185645

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planeParallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_parallel_lines
  (a b : Line) (α β : Plane)
  (distinct_lines : a ≠ b)
  (distinct_planes : α ≠ β)
  (a_perp_α : perpendicular a α)
  (b_perp_β : perpendicular b β)
  (a_parallel_b : parallel a b) :
  planeParallel α β :=
sorry

end planes_parallel_if_perpendicular_to_parallel_lines_l1856_185645


namespace line_parallel_plane_transitivity_l1856_185644

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallel : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_plane_transitivity 
  (a b : Line) (α : Plane) :
  parallel a α → subset b α → parallel a α :=
by sorry

end line_parallel_plane_transitivity_l1856_185644


namespace real_number_line_bijection_l1856_185606

-- Define the number line as a type
def NumberLine : Type := ℝ

-- Define the bijection between real numbers and points on the number line
def realToPoint : ℝ → NumberLine := id

-- Statement: There is a one-to-one correspondence between real numbers and points on the number line
theorem real_number_line_bijection : Function.Bijective realToPoint := by
  sorry

end real_number_line_bijection_l1856_185606


namespace video_upload_total_l1856_185636

theorem video_upload_total (days_in_month : ℕ) (initial_daily_upload : ℕ) : 
  days_in_month = 30 →
  initial_daily_upload = 10 →
  (days_in_month / 2 * initial_daily_upload) + 
  (days_in_month / 2 * (2 * initial_daily_upload)) = 450 := by
sorry

end video_upload_total_l1856_185636


namespace angela_action_figures_l1856_185622

theorem angela_action_figures (initial : ℕ) : 
  (initial : ℚ) * (3/4) * (2/3) = 12 → initial = 24 := by
  sorry

end angela_action_figures_l1856_185622


namespace equation_solution_l1856_185697

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 5 → (x + 36 / (x - 5) = -9 ↔ x = -9 ∨ x = 5) :=
by sorry

end equation_solution_l1856_185697


namespace inequality_proof_l1856_185667

theorem inequality_proof (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_one : a + b + c = 1) : 
  a * (1 + b - c)^(1/3 : ℝ) + b * (1 + c - a)^(1/3 : ℝ) + c * (1 + a - b)^(1/3 : ℝ) ≤ 1 := by
  sorry

end inequality_proof_l1856_185667


namespace compound_composition_l1856_185698

/-- The atomic weight of aluminum in g/mol -/
def aluminum_weight : ℝ := 26.98

/-- The atomic weight of chlorine in g/mol -/
def chlorine_weight : ℝ := 35.45

/-- The molecular weight of the compound in g/mol -/
def compound_weight : ℝ := 132

/-- The number of chlorine atoms in the compound -/
def chlorine_atoms : ℕ := 3

theorem compound_composition :
  ∃ (n : ℕ), n = chlorine_atoms ∧
  compound_weight = aluminum_weight + n * chlorine_weight :=
sorry

end compound_composition_l1856_185698


namespace spirit_mixture_problem_l1856_185617

/-- Given three vessels a, b, and c with spirit concentrations of 45%, 30%, and 10% respectively,
    and a mixture of x litres from vessel a, 5 litres from vessel b, and 6 litres from vessel c
    resulting in a 26% spirit concentration, prove that x = 4 litres. -/
theorem spirit_mixture_problem (x : ℝ) :
  (0.45 * x + 0.30 * 5 + 0.10 * 6) / (x + 5 + 6) = 0.26 → x = 4 := by
  sorry

#check spirit_mixture_problem

end spirit_mixture_problem_l1856_185617


namespace root_in_interval_l1856_185657

def f (x : ℝ) := x^3 + x - 1

theorem root_in_interval :
  (f 0.5 < 0) → (f 0.75 > 0) →
  ∃ x₀ ∈ Set.Ioo 0.5 0.75, f x₀ = 0 :=
by sorry

end root_in_interval_l1856_185657


namespace perpendicular_lines_from_parallel_planes_l1856_185680

/-- Two lines are distinct if they are not equal -/
def distinct_lines (l m : Line) : Prop := l ≠ m

/-- Two planes are distinct if they are not equal -/
def distinct_planes (α β : Plane) : Prop := α ≠ β

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line) (α : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line) (α : Plane) : Prop := sorry

/-- Two planes are parallel -/
def planes_parallel (α β : Plane) : Prop := sorry

/-- Two lines are perpendicular -/
def lines_perpendicular (l m : Line) : Prop := sorry

theorem perpendicular_lines_from_parallel_planes 
  (l m : Line) (α β : Plane) 
  (h1 : distinct_lines l m)
  (h2 : distinct_planes α β)
  (h3 : planes_parallel α β)
  (h4 : line_perp_plane l α)
  (h5 : line_parallel_plane m β) :
  lines_perpendicular l m := by sorry

end perpendicular_lines_from_parallel_planes_l1856_185680


namespace group_size_proof_l1856_185609

theorem group_size_proof (total_collection : ℚ) (paise_per_rupee : ℕ) : 
  (total_collection = 32.49) →
  (paise_per_rupee = 100) →
  ∃ n : ℕ, (n * n = total_collection * paise_per_rupee) ∧ (n = 57) :=
by sorry

end group_size_proof_l1856_185609


namespace twelfth_term_value_l1856_185628

/-- An arithmetic sequence with a₂ = -8 and common difference d = 2 -/
def arithmetic_seq (n : ℕ) : ℤ :=
  let a₁ : ℤ := -10  -- Derived from a₂ = -8 and d = 2
  a₁ + (n - 1) * 2

theorem twelfth_term_value : arithmetic_seq 12 = 12 := by
  sorry

end twelfth_term_value_l1856_185628


namespace product_sum_theorem_l1856_185655

theorem product_sum_theorem (x y : ℤ) : 
  y = x + 2 → x * y = 20400 → x + y = 286 := by
sorry

end product_sum_theorem_l1856_185655


namespace parallel_condition_l1856_185624

/-- Two lines l₁ and l₂ in the plane -/
structure TwoLines where
  a : ℝ
  l₁ : ℝ → ℝ → ℝ := λ x y => a * x + (a + 2) * y + 1
  l₂ : ℝ → ℝ → ℝ := λ x y => x + a * y + 2

/-- The condition for two lines to be parallel -/
def parallel (lines : TwoLines) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ lines.a = k ∧ lines.a + 2 = k * lines.a

/-- The statement to be proved -/
theorem parallel_condition (lines : TwoLines) :
  (parallel lines → lines.a = -1) ∧ ¬(lines.a = -1 → parallel lines) :=
sorry

end parallel_condition_l1856_185624


namespace problem_1_problem_2_l1856_185610

-- Problem 1
theorem problem_1 (x : ℝ) (h : x > 0) (eq : Real.sqrt x + 1 / Real.sqrt x = 3) : 
  x + 1 / x = 7 := by sorry

-- Problem 2
theorem problem_2 : 
  (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) * (Real.log 2 / Real.log 3 + Real.log 8 / Real.log 9) = 25 / 12 := by sorry

end problem_1_problem_2_l1856_185610


namespace inequality_proof_l1856_185656

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end inequality_proof_l1856_185656


namespace contour_bar_chart_judges_relationship_l1856_185601

/-- Represents a method for judging the relationship between categorical variables -/
inductive IndependenceTestMethod
  | Residuals
  | ContourBarChart
  | HypothesisTesting
  | Other

/-- Defines the property of being able to roughly judge the relationship between categorical variables -/
def can_roughly_judge_relationship (method : IndependenceTestMethod) : Prop :=
  match method with
  | IndependenceTestMethod.ContourBarChart => True
  | _ => False

/-- Theorem stating that a contour bar chart can be used to roughly judge the relationship between categorical variables in an independence test -/
theorem contour_bar_chart_judges_relationship :
  can_roughly_judge_relationship IndependenceTestMethod.ContourBarChart :=
sorry

end contour_bar_chart_judges_relationship_l1856_185601


namespace union_of_a_and_b_l1856_185616

def U : Set Nat := {0, 1, 2, 3, 4}

theorem union_of_a_and_b (A B : Set Nat) 
  (h1 : U = {0, 1, 2, 3, 4})
  (h2 : (U \ A) = {1, 2})
  (h3 : B = {1, 3}) :
  A ∪ B = {0, 1, 3, 4} := by
sorry

end union_of_a_and_b_l1856_185616


namespace weight_loss_challenge_l1856_185692

theorem weight_loss_challenge (original_weight : ℝ) (h : original_weight > 0) :
  let weight_after_loss := 0.87 * original_weight
  let final_measured_weight := 0.8874 * original_weight
  let clothes_weight := final_measured_weight - weight_after_loss
  clothes_weight / weight_after_loss = 0.02 := by
sorry

end weight_loss_challenge_l1856_185692


namespace prob_three_two_digit_l1856_185600

/-- The number of dice being rolled -/
def num_dice : ℕ := 6

/-- The number of sides on each die -/
def num_sides : ℕ := 20

/-- The probability of rolling a two-digit number on a single die -/
def p_two_digit : ℚ := 11 / 20

/-- The probability of rolling a one-digit number on a single die -/
def p_one_digit : ℚ := 9 / 20

/-- The probability of exactly three dice showing a two-digit number when rolling 6 20-sided dice -/
theorem prob_three_two_digit : 
  (num_dice.choose 3 : ℚ) * p_two_digit ^ 3 * p_one_digit ^ 3 = 973971 / 3200000 :=
sorry

end prob_three_two_digit_l1856_185600


namespace statement_1_incorrect_statement_4_incorrect_l1856_185605

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Statement 1
theorem statement_1_incorrect 
  (α : Plane) (l m n : Line) : 
  ¬(∀ (α : Plane) (l m n : Line), 
    contains α m → contains α n → perpendicular l m → perpendicular l n → 
    perpendicularToPlane l α) := 
by sorry

-- Statement 4
theorem statement_4_incorrect 
  (α : Plane) (l m n : Line) : 
  ¬(∀ (α : Plane) (l m n : Line), 
    contains α m → perpendicularToPlane n α → perpendicular l n → 
    parallel l m) := 
by sorry

end statement_1_incorrect_statement_4_incorrect_l1856_185605


namespace arithmetic_mean_change_l1856_185661

theorem arithmetic_mean_change (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 10 →
  b + c + d = 33 →
  a + c + d = 36 →
  a + b + d = 39 →
  (a + b + c) / 3 = 4 :=
by sorry

end arithmetic_mean_change_l1856_185661


namespace tangent_line_to_parabola_l1856_185679

theorem tangent_line_to_parabola (x y : ℝ) :
  y = x^2 →                                    -- Condition: parabola equation
  (∃ k : ℝ, k * x - y + 4 = 0) →               -- Condition: parallel line exists
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧        -- Tangent line equation
               a / b = 2 ∧                     -- Parallel to given line
               (∃ x₀ y₀ : ℝ, y₀ = x₀^2 ∧       -- Point on parabola
                             a * x₀ + b * y₀ + c = 0 ∧  -- Point on tangent line
                             2 * x₀ = (y₀ - y) / (x₀ - x))) →  -- Derivative condition
  2 * x - y - 1 = 0 :=                         -- Conclusion: specific tangent line equation
by sorry

end tangent_line_to_parabola_l1856_185679


namespace pure_imaginary_condition_l1856_185639

-- Define the complex number z as (m+1)(1-i)
def z (m : ℝ) : ℂ := (m + 1) * (1 - Complex.I)

-- Theorem statement
theorem pure_imaginary_condition (m : ℝ) :
  (z m).re = 0 → m = -1 := by
  sorry

end pure_imaginary_condition_l1856_185639


namespace ball_drawing_theorem_l1856_185674

def total_balls : ℕ := 9
def white_balls : ℕ := 4
def black_balls : ℕ := 5
def drawn_balls : ℕ := 3

theorem ball_drawing_theorem :
  -- 1. Total number of ways to choose 3 balls from 9 balls
  Nat.choose total_balls drawn_balls = 84 ∧
  -- 2. Number of ways to choose 2 white and 1 black
  (Nat.choose white_balls 2 * Nat.choose black_balls 1) = 30 ∧
  -- 3. Number of ways to choose at least 2 white balls
  (Nat.choose white_balls 2 * Nat.choose black_balls 1 + Nat.choose white_balls 3) = 34 ∧
  -- 4. Probability of choosing 2 white and 1 black
  (↑(Nat.choose white_balls 2 * Nat.choose black_balls 1) / ↑(Nat.choose total_balls drawn_balls) : ℚ) = 30 / 84 ∧
  -- 5. Probability of choosing at least 2 white balls
  (↑(Nat.choose white_balls 2 * Nat.choose black_balls 1 + Nat.choose white_balls 3) / ↑(Nat.choose total_balls drawn_balls) : ℚ) = 34 / 84 :=
by sorry

end ball_drawing_theorem_l1856_185674


namespace coin_arrangement_l1856_185633

theorem coin_arrangement (n : ℕ) : 
  n ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) → 
  (n * 4 = 12 ↔ n = 3) :=
by sorry

end coin_arrangement_l1856_185633


namespace photo_selection_choices_l1856_185687

theorem photo_selection_choices : ∀ n : ℕ, n = 5 →
  (Nat.choose n 3 + Nat.choose n 4 = 15) :=
by
  sorry

end photo_selection_choices_l1856_185687


namespace bus_capacity_is_193_l1856_185663

/-- Represents the seating capacity of a double-decker bus -/
def double_decker_bus_capacity (lower_left : ℕ) (lower_right : ℕ) (regular_seat_capacity : ℕ)
  (priority_seats : ℕ) (priority_seat_capacity : ℕ) (upper_left : ℕ) (upper_right : ℕ)
  (upper_seat_capacity : ℕ) (upper_back : ℕ) : ℕ :=
  (lower_left + lower_right) * regular_seat_capacity +
  priority_seats * priority_seat_capacity +
  (upper_left + upper_right) * upper_seat_capacity +
  upper_back

/-- Theorem stating the total seating capacity of the given double-decker bus -/
theorem bus_capacity_is_193 :
  double_decker_bus_capacity 15 12 2 4 1 20 20 3 15 = 193 := by
  sorry


end bus_capacity_is_193_l1856_185663


namespace smallest_n_square_and_cube_l1856_185678

theorem smallest_n_square_and_cube : 
  (∃ (n : ℕ), n > 0 ∧ 
   (∃ (k : ℕ), 5 * n = k^2) ∧ 
   (∃ (m : ℕ), 7 * n = m^3) ∧
   (∀ (x : ℕ), x > 0 → 
    (∃ (y : ℕ), 5 * x = y^2) → 
    (∃ (z : ℕ), 7 * x = z^3) → 
    x ≥ 1715)) ∧
  (∃ (k m : ℕ), 5 * 1715 = k^2 ∧ 7 * 1715 = m^3) :=
by sorry

end smallest_n_square_and_cube_l1856_185678


namespace prism_minimum_characteristics_l1856_185686

/-- A prism is a polyhedron with two congruent parallel faces (bases) and all other faces (lateral faces) are parallelograms. -/
structure Prism where
  base_edges : ℕ
  height : ℝ
  height_pos : height > 0

/-- The number of faces in a prism -/
def num_faces (p : Prism) : ℕ := p.base_edges + 2

/-- The number of edges in a prism -/
def num_edges (p : Prism) : ℕ := 3 * p.base_edges

/-- The number of lateral edges in a prism -/
def num_lateral_edges (p : Prism) : ℕ := p.base_edges

/-- The number of vertices in a prism -/
def num_vertices (p : Prism) : ℕ := 2 * p.base_edges

/-- Theorem about the minimum characteristics of a prism -/
theorem prism_minimum_characteristics :
  (∀ p : Prism, num_faces p ≥ 5) ∧
  (∃ p : Prism, num_faces p = 5 ∧
                num_edges p = 9 ∧
                num_lateral_edges p = 3 ∧
                num_vertices p = 6) := by sorry

end prism_minimum_characteristics_l1856_185686


namespace geometric_sequence_problem_l1856_185677

/-- A geometric sequence with positive terms and common ratio greater than 1 -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h_positive : ∀ n, a n > 0
  h_ratio : q > 1
  h_geometric : ∀ n, a (n + 1) = q * a n

/-- Theorem statement for the geometric sequence problem -/
theorem geometric_sequence_problem (seq : GeometricSequence)
  (h1 : seq.a 3 + seq.a 5 = 20)
  (h2 : seq.a 2 * seq.a 6 = 64) :
  seq.a 6 = 32 := by
    sorry

end geometric_sequence_problem_l1856_185677


namespace magic_8_ball_probability_l1856_185676

theorem magic_8_ball_probability :
  let n : ℕ := 7  -- total number of questions
  let k : ℕ := 3  -- number of positive answers
  let p : ℚ := 1/3  -- probability of a positive answer
  Nat.choose n k * p^k * (1-p)^(n-k) = 560/2187 :=
by sorry

end magic_8_ball_probability_l1856_185676


namespace product_equals_square_l1856_185648

theorem product_equals_square : 1000 * 1993 * 0.1993 * 10 = (1993 : ℝ)^2 := by
  sorry

end product_equals_square_l1856_185648
