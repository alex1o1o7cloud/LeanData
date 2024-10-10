import Mathlib

namespace octagon_coloring_count_l2386_238677

/-- The number of disks in the octagonal pattern -/
def num_disks : ℕ := 8

/-- The number of blue disks -/
def num_blue : ℕ := 3

/-- The number of red disks -/
def num_red : ℕ := 3

/-- The number of green disks -/
def num_green : ℕ := 2

/-- The symmetry group of a regular octagon -/
def octagon_symmetry_group_order : ℕ := 16

/-- The number of distinct colorings considering symmetries -/
def distinct_colorings : ℕ := 43

/-- Theorem stating the number of distinct colorings -/
theorem octagon_coloring_count :
  let total_colorings := (Nat.choose num_disks num_blue) * (Nat.choose (num_disks - num_blue) num_red)
  (total_colorings / octagon_symmetry_group_order : ℚ).num = distinct_colorings := by
  sorry

end octagon_coloring_count_l2386_238677


namespace rectangular_prism_layers_l2386_238657

theorem rectangular_prism_layers (prism_volume : ℕ) (block_volume : ℕ) (blocks_per_layer : ℕ) (h1 : prism_volume = 252) (h2 : block_volume = 1) (h3 : blocks_per_layer = 36) : 
  (prism_volume / (blocks_per_layer * block_volume) : ℕ) = 7 := by
sorry

end rectangular_prism_layers_l2386_238657


namespace quadratic_decreasing_implies_h_geq_one_l2386_238615

/-- A quadratic function of the form y = (x - h)^2 + 3 -/
def quadratic_function (h : ℝ) (x : ℝ) : ℝ := (x - h)^2 + 3

/-- The derivative of the quadratic function -/
def quadratic_derivative (h : ℝ) (x : ℝ) : ℝ := 2 * (x - h)

theorem quadratic_decreasing_implies_h_geq_one (h : ℝ) :
  (∀ x < 1, quadratic_derivative h x < 0) → h ≥ 1 := by
  sorry

end quadratic_decreasing_implies_h_geq_one_l2386_238615


namespace crabapple_recipients_count_l2386_238642

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 3

/-- The number of different sequences of crabapple recipients in a week -/
def crabapple_sequences : ℕ := num_students * (num_students - 1) * (num_students - 2)

/-- Theorem stating the number of different sequences of crabapple recipients -/
theorem crabapple_recipients_count :
  crabapple_sequences = 2730 :=
by sorry

end crabapple_recipients_count_l2386_238642


namespace blue_spools_count_l2386_238651

/-- The number of spools needed to make one beret -/
def spools_per_beret : ℕ := 3

/-- The number of red yarn spools -/
def red_spools : ℕ := 12

/-- The number of black yarn spools -/
def black_spools : ℕ := 15

/-- The total number of berets that can be made -/
def total_berets : ℕ := 11

/-- The number of blue yarn spools -/
def blue_spools : ℕ := total_berets * spools_per_beret - (red_spools + black_spools)

theorem blue_spools_count : blue_spools = 6 := by
  sorry

end blue_spools_count_l2386_238651


namespace abs_z_eq_one_l2386_238666

theorem abs_z_eq_one (z : ℂ) (h : (1 - Complex.I) / z = 1 + Complex.I) : Complex.abs z = 1 := by
  sorry

end abs_z_eq_one_l2386_238666


namespace no_integer_solution_implies_k_range_l2386_238668

theorem no_integer_solution_implies_k_range (k : ℝ) : 
  (∀ x : ℤ, ¬((k * x - k^2 - 4) * (x - 4) < 0)) → 
  1 ≤ k ∧ k ≤ 4 :=
sorry

end no_integer_solution_implies_k_range_l2386_238668


namespace ricky_age_solution_l2386_238679

def ricky_age_problem (rickys_age : ℕ) (fathers_age : ℕ) : Prop :=
  fathers_age = 45 ∧
  rickys_age + 5 = (1 / 5 : ℚ) * (fathers_age + 5 : ℚ) + 5

theorem ricky_age_solution :
  ∃ (rickys_age : ℕ), ricky_age_problem rickys_age 45 ∧ rickys_age = 10 :=
sorry

end ricky_age_solution_l2386_238679


namespace min_bricks_needed_l2386_238659

/-- Represents the dimensions of a brick -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of the parallelepiped -/
structure ParallelepipedDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The theorem statement -/
theorem min_bricks_needed
  (brick : BrickDimensions)
  (parallelepiped : ParallelepipedDimensions)
  (h1 : brick.length = 22)
  (h2 : brick.width = 11)
  (h3 : brick.height = 6)
  (h4 : parallelepiped.length = 5 * parallelepiped.height / 4)
  (h5 : parallelepiped.width = 3 * parallelepiped.height / 2)
  (h6 : parallelepiped.length % brick.length = 0)
  (h7 : parallelepiped.width % brick.width = 0)
  (h8 : parallelepiped.height % brick.height = 0) :
  (parallelepiped.length / brick.length) *
  (parallelepiped.width / brick.width) *
  (parallelepiped.height / brick.height) = 13200 := by
  sorry

end min_bricks_needed_l2386_238659


namespace jack_morning_emails_indeterminate_l2386_238648

/-- Represents the number of emails received at different times of the day -/
structure EmailCount where
  morning : ℕ
  afternoon : ℕ
  evening : ℕ

/-- Defines the properties of Jack's email counts -/
def jack_email_properties (e : EmailCount) : Prop :=
  e.afternoon = 5 ∧ 
  e.evening = 8 ∧ 
  e.afternoon + e.evening = 13

/-- Theorem stating that Jack's morning email count cannot be uniquely determined -/
theorem jack_morning_emails_indeterminate :
  ∃ e1 e2 : EmailCount, 
    jack_email_properties e1 ∧ 
    jack_email_properties e2 ∧ 
    e1.morning ≠ e2.morning :=
sorry

end jack_morning_emails_indeterminate_l2386_238648


namespace sam_has_six_balloons_l2386_238621

/-- The number of yellow balloons Fred has -/
def fred_balloons : ℕ := 5

/-- The number of yellow balloons Mary has -/
def mary_balloons : ℕ := 7

/-- The total number of yellow balloons -/
def total_balloons : ℕ := 18

/-- The number of yellow balloons Sam has -/
def sam_balloons : ℕ := total_balloons - fred_balloons - mary_balloons

theorem sam_has_six_balloons : sam_balloons = 6 := by
  sorry

end sam_has_six_balloons_l2386_238621


namespace smallest_three_digit_multiple_of_17_l2386_238636

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by
  sorry

end smallest_three_digit_multiple_of_17_l2386_238636


namespace stellas_dolls_count_l2386_238691

theorem stellas_dolls_count : 
  ∀ (num_dolls : ℕ),
  (num_dolls : ℝ) * 5 + 2 * 15 + 5 * 4 - 40 = 25 →
  num_dolls = 3 := by
sorry

end stellas_dolls_count_l2386_238691


namespace smallest_sequence_sum_l2386_238603

theorem smallest_sequence_sum : ∃ (A B C D : ℕ),
  (A > 0 ∧ B > 0 ∧ C > 0) ∧  -- A, B, C are positive integers
  (∃ (r : ℚ), C - B = B - A ∧ C = B * r ∧ D = C * r) ∧  -- arithmetic and geometric sequences
  (C : ℚ) / B = 7 / 4 ∧  -- C/B = 7/4
  (∀ (A' B' C' D' : ℕ),
    (A' > 0 ∧ B' > 0 ∧ C' > 0) →
    (∃ (r' : ℚ), C' - B' = B' - A' ∧ C' = B' * r' ∧ D' = C' * r') →
    (C' : ℚ) / B' = 7 / 4 →
    A + B + C + D ≤ A' + B' + C' + D') ∧
  A + B + C + D = 97 :=
by sorry

end smallest_sequence_sum_l2386_238603


namespace special_matrix_exists_iff_even_l2386_238616

/-- A matrix with elements from {-1, 0, 1} -/
def SpecialMatrix (n : ℕ) := Matrix (Fin n) (Fin n) (Fin 3)

/-- The sum of elements in a row of a SpecialMatrix -/
def rowSum (A : SpecialMatrix n) (i : Fin n) : ℤ := sorry

/-- The sum of elements in a column of a SpecialMatrix -/
def colSum (A : SpecialMatrix n) (j : Fin n) : ℤ := sorry

/-- All row and column sums are distinct -/
def distinctSums (A : SpecialMatrix n) : Prop :=
  ∀ i j i' j', (i ≠ i' ∨ j ≠ j') → 
    (rowSum A i ≠ rowSum A i' ∧ 
     rowSum A i ≠ colSum A j' ∧ 
     colSum A j ≠ rowSum A i' ∧ 
     colSum A j ≠ colSum A j')

theorem special_matrix_exists_iff_even (n : ℕ) :
  (∃ A : SpecialMatrix n, distinctSums A) ↔ (∃ k : ℕ, n = 2 * k) :=
sorry

end special_matrix_exists_iff_even_l2386_238616


namespace proportion_problem_l2386_238671

theorem proportion_problem (hours_per_day : ℝ) (h : hours_per_day = 24) :
  ∃ x : ℝ, (24 : ℝ) / (6 / hours_per_day) = x / 8 ∧ x = 768 :=
by
  sorry

end proportion_problem_l2386_238671


namespace union_M_complement_N_l2386_238632

universe u

def U : Finset ℕ := {0, 1, 2, 3, 4, 5}
def M : Finset ℕ := {0, 3, 5}
def N : Finset ℕ := {1, 4, 5}

theorem union_M_complement_N : M ∪ (U \ N) = {0, 2, 3, 5} := by sorry

end union_M_complement_N_l2386_238632


namespace intersection_point_l2386_238652

/-- The line equation -/
def line (x y z : ℝ) : Prop :=
  (x - 2) / 4 = (y - 1) / (-3) ∧ (y - 1) / (-3) = (z + 3) / (-2)

/-- The plane equation -/
def plane (x y z : ℝ) : Prop :=
  3 * x - y + 4 * z = 0

/-- The theorem stating that (6, -2, -5) is the unique point of intersection -/
theorem intersection_point : ∃! (x y z : ℝ), line x y z ∧ plane x y z ∧ x = 6 ∧ y = -2 ∧ z = -5 := by
  sorry

end intersection_point_l2386_238652


namespace rainfall_problem_l2386_238627

/-- Rainfall problem statement -/
theorem rainfall_problem (day1 day2 day3 normal_avg this_year_total : ℝ) 
  (h1 : day1 = 26)
  (h2 : day3 = day2 - 12)
  (h3 : normal_avg = 140)
  (h4 : this_year_total = normal_avg - 58)
  (h5 : this_year_total = day1 + day2 + day3) :
  day2 = 34 := by
sorry

end rainfall_problem_l2386_238627


namespace percentage_of_adult_men_l2386_238667

theorem percentage_of_adult_men (total : ℕ) (children : ℕ) 
  (h1 : total = 2000) 
  (h2 : children = 200) 
  (h3 : ∃ (men women : ℕ), men + women + children = total ∧ women = 2 * men) :
  ∃ (men : ℕ), men * 100 / total = 30 := by
sorry

end percentage_of_adult_men_l2386_238667


namespace f_nonnegative_condition_f_two_zeros_condition_l2386_238678

/-- The function f(x) defined as |x^2 - 1| + x^2 + kx -/
def f (k : ℝ) (x : ℝ) : ℝ := |x^2 - 1| + x^2 + k*x

theorem f_nonnegative_condition (k : ℝ) :
  (∀ x > 0, f k x ≥ 0) ↔ k ≥ -1 := by sorry

theorem f_two_zeros_condition (k : ℝ) (x₁ x₂ : ℝ) :
  (0 < x₁ ∧ x₁ < 2 ∧ 0 < x₂ ∧ x₂ < 2 ∧ x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0) →
  (-7/2 < k ∧ k < -1 ∧ 2 < 1/x₁ + 1/x₂ ∧ 1/x₁ + 1/x₂ < 4) := by sorry

end f_nonnegative_condition_f_two_zeros_condition_l2386_238678


namespace labourer_fine_problem_l2386_238626

/-- Calculates the fine per day of absence for a labourer --/
def calculate_fine_per_day (total_days : ℕ) (daily_wage : ℚ) (total_received : ℚ) (days_absent : ℕ) : ℚ :=
  let days_worked := total_days - days_absent
  let total_earned := days_worked * daily_wage
  (total_earned - total_received) / days_absent

/-- Theorem stating the fine per day of absence for the given problem --/
theorem labourer_fine_problem :
  calculate_fine_per_day 25 2 (37 + 1/2) 5 = 1/2 := by
  sorry

end labourer_fine_problem_l2386_238626


namespace marble_sculpture_first_week_cut_l2386_238634

/-- Proves that the percentage of marble cut away in the first week is 30% --/
theorem marble_sculpture_first_week_cut (
  original_weight : ℝ)
  (second_week_cut : ℝ)
  (third_week_cut : ℝ)
  (final_weight : ℝ)
  (h1 : original_weight = 250)
  (h2 : second_week_cut = 20)
  (h3 : third_week_cut = 25)
  (h4 : final_weight = 105)
  : ∃ (first_week_cut : ℝ),
    first_week_cut = 30 ∧
    final_weight = original_weight * 
      (1 - first_week_cut / 100) * 
      (1 - second_week_cut / 100) * 
      (1 - third_week_cut / 100) := by
  sorry


end marble_sculpture_first_week_cut_l2386_238634


namespace necessary_but_not_sufficient_l2386_238663

theorem necessary_but_not_sufficient :
  (∀ a b c d : ℝ, (a > b ∧ c > d) → (a + c > b + d)) ∧
  (∃ a b c d : ℝ, (a + c > b + d) ∧ ¬(a > b ∧ c > d)) :=
by sorry

end necessary_but_not_sufficient_l2386_238663


namespace equation_solution_l2386_238682

theorem equation_solution : ∃ k : ℤ, 2^4 - 6 = 3^3 + k ∧ k = -17 := by
  sorry

end equation_solution_l2386_238682


namespace extreme_value_implies_sum_l2386_238695

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- Theorem: If f(x) has an extreme value of 10 at x = 1, then a + b = -7 -/
theorem extreme_value_implies_sum (a b : ℝ) :
  (∃ (ε : ℝ), ∀ (x : ℝ), |x - 1| < ε → f a b x ≤ f a b 1) ∧
  f a b 1 = 10 →
  a + b = -7 :=
by sorry

end extreme_value_implies_sum_l2386_238695


namespace initial_playtime_l2386_238620

/-- Proof of initial daily playtime in a game scenario -/
theorem initial_playtime (initial_days : ℕ) (initial_completion_percent : ℚ)
  (remaining_days : ℕ) (remaining_hours_per_day : ℕ) :
  initial_days = 14 →
  initial_completion_percent = 2/5 →
  remaining_days = 12 →
  remaining_hours_per_day = 7 →
  ∃ (x : ℚ),
    x * initial_days = initial_completion_percent * (x * initial_days + remaining_days * remaining_hours_per_day) ∧
    x = 4 := by
  sorry

end initial_playtime_l2386_238620


namespace cards_distribution_l2386_238694

theorem cards_distribution (total_cards : ℕ) (total_people : ℕ) 
  (h1 : total_cards = 60) (h2 : total_people = 9) : 
  (total_people - (total_cards % total_people)) = 3 := by
  sorry

end cards_distribution_l2386_238694


namespace inequality_proof_l2386_238697

theorem inequality_proof (x y : ℝ) : x + y + Real.sqrt (x * y) ≤ 3 * (x + y - Real.sqrt (x * y)) := by
  sorry

end inequality_proof_l2386_238697


namespace basket_balls_count_l2386_238699

theorem basket_balls_count (total : ℕ) (red : ℕ) (yellow : ℕ) (prob : ℚ) : 
  red = 8 →
  prob = 2/5 →
  total = red + yellow →
  prob = red / total →
  yellow = 12 := by
sorry

end basket_balls_count_l2386_238699


namespace min_force_to_submerge_cube_l2386_238673

/-- Minimum force required to submerge a cube -/
theorem min_force_to_submerge_cube 
  (cube_volume : Real) 
  (cube_density : Real) 
  (water_density : Real) 
  (gravity : Real) :
  cube_volume = 1e-5 →  -- 10 cm³ = 1e-5 m³
  cube_density = 700 →
  water_density = 1000 →
  gravity = 10 →
  (water_density - cube_density) * cube_volume * gravity = 0.03 := by
  sorry

end min_force_to_submerge_cube_l2386_238673


namespace geometric_sequence_second_term_l2386_238683

/-- A geometric sequence with positive integer terms -/
def GeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℚ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = (a n : ℚ) * r

theorem geometric_sequence_second_term
  (a : ℕ → ℕ)
  (h_geom : GeometricSequence a)
  (h_first : a 0 = 5)
  (h_fourth : a 3 = 480) :
  a 1 = 20 := by
  sorry

end geometric_sequence_second_term_l2386_238683


namespace hotel_revenue_l2386_238654

theorem hotel_revenue
  (total_rooms : ℕ)
  (single_room_cost double_room_cost : ℕ)
  (single_rooms_booked : ℕ)
  (h_total : total_rooms = 260)
  (h_single_cost : single_room_cost = 35)
  (h_double_cost : double_room_cost = 60)
  (h_single_booked : single_rooms_booked = 64) :
  single_room_cost * single_rooms_booked +
  double_room_cost * (total_rooms - single_rooms_booked) = 14000 := by
sorry

end hotel_revenue_l2386_238654


namespace arithmetic_progression_square_l2386_238614

/-- An arithmetic progression containing two natural numbers and the square of the smaller one also contains the square of the larger one. -/
theorem arithmetic_progression_square (a b : ℕ) (d : ℚ) (n m : ℤ) :
  a < b →
  b = a + n * d →
  a^2 = a + m * d →
  ∃ k : ℤ, b^2 = a + k * d :=
by sorry

end arithmetic_progression_square_l2386_238614


namespace cars_already_parked_equals_62_l2386_238690

/-- Represents a multi-story parking lot -/
structure ParkingLot where
  totalCapacity : ℕ
  levels : ℕ
  additionalCapacity : ℕ

/-- The number of cars already parked on one level -/
def carsAlreadyParked (p : ParkingLot) : ℕ :=
  p.totalCapacity / p.levels - p.additionalCapacity

/-- Theorem stating the number of cars already parked on one level -/
theorem cars_already_parked_equals_62 (p : ParkingLot) 
    (h1 : p.totalCapacity = 425)
    (h2 : p.levels = 5)
    (h3 : p.additionalCapacity = 62) :
    carsAlreadyParked p = 62 := by
  sorry

#eval carsAlreadyParked { totalCapacity := 425, levels := 5, additionalCapacity := 62 }

end cars_already_parked_equals_62_l2386_238690


namespace ball_probability_l2386_238601

theorem ball_probability (total : ℕ) (red : ℕ) (purple : ℕ) 
  (h1 : total = 100) 
  (h2 : red = 9) 
  (h3 : purple = 3) : 
  (total - (red + purple)) / total = 88 / 100 :=
by
  sorry

end ball_probability_l2386_238601


namespace reciprocal_problem_l2386_238622

theorem reciprocal_problem (x : ℚ) (h : 8 * x = 3) : 150 * (1 / x) = 400 := by
  sorry

end reciprocal_problem_l2386_238622


namespace total_saltwater_animals_l2386_238670

theorem total_saltwater_animals (num_aquariums : ℕ) (animals_per_aquarium : ℕ) 
  (h1 : num_aquariums = 26)
  (h2 : animals_per_aquarium = 2) :
  num_aquariums * animals_per_aquarium = 52 := by
  sorry

end total_saltwater_animals_l2386_238670


namespace leila_cake_problem_l2386_238600

theorem leila_cake_problem (monday : ℕ) (friday : ℕ) (saturday : ℕ) : 
  friday = 9 →
  saturday = 3 * monday →
  monday + friday + saturday = 33 →
  monday = 6 := by
sorry

end leila_cake_problem_l2386_238600


namespace parallel_transitivity_l2386_238693

-- Define a type for lines in space
structure Line3D where
  -- You might want to add more specific properties here
  -- but for this problem, we just need to distinguish between lines

-- Define parallelism for lines in space
def parallel (l1 l2 : Line3D) : Prop :=
  -- The actual definition of parallelism would go here
  sorry

-- The theorem statement
theorem parallel_transitivity (l m n : Line3D) : 
  parallel l m → parallel l n → parallel m n := by
  sorry

end parallel_transitivity_l2386_238693


namespace parabola_trajectory_l2386_238669

/-- The trajectory of point M given the conditions of the parabola and vector relationship -/
theorem parabola_trajectory (x y t : ℝ) : 
  let F : ℝ × ℝ := (1, 0)  -- Focus of the parabola
  let P : ℝ → ℝ × ℝ := λ t => (t^2/4, t)  -- Point on the parabola
  let M : ℝ × ℝ := (x, y)  -- Point M
  (∀ t, (P t).2^2 = 4 * (P t).1) →  -- P is on the parabola y^2 = 4x
  ((P t).1 - F.1, (P t).2 - F.2) = (2*(x - F.1), 2*(y - F.2)) →  -- FP = 2FM
  y^2 = 2*x - 1  -- Trajectory equation
:= by sorry

end parabola_trajectory_l2386_238669


namespace josh_ribbon_shortage_l2386_238607

/-- Calculates the shortage of ribbon for gift wrapping --/
def ribbon_shortage (total_ribbon : ℝ) (num_gifts : ℕ) 
  (wrap_per_gift : ℝ) (bow_per_gift : ℝ) (tag_per_gift : ℝ) (trim_per_gift : ℝ) : ℝ :=
  let required_ribbon := num_gifts * (wrap_per_gift + bow_per_gift + tag_per_gift + trim_per_gift)
  required_ribbon - total_ribbon

/-- Proves that Josh is short by 7.5 yards of ribbon --/
theorem josh_ribbon_shortage : 
  ribbon_shortage 18 6 2 1.5 0.25 0.5 = 7.5 := by
  sorry

end josh_ribbon_shortage_l2386_238607


namespace expression_evaluation_l2386_238684

theorem expression_evaluation : (3^2 - 3) + (4^2 - 4) - (5^2 - 5) - (6^2 - 6) = -32 := by
  sorry

end expression_evaluation_l2386_238684


namespace max_value_and_sum_l2386_238629

theorem max_value_and_sum (x y z v w : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0 ∧ v > 0 ∧ w > 0) 
  (heq : 4 * x^2 + y^2 + z^2 + v^2 + w^2 = 8080) :
  (∃ (M : ℝ), ∀ (x' y' z' v' w' : ℝ), 
    x' > 0 → y' > 0 → z' > 0 → v' > 0 → w' > 0 →
    4 * x'^2 + y'^2 + z'^2 + v'^2 + w'^2 = 8080 →
    x' * z' + 4 * y' * z' + 6 * z' * v' + 14 * z' * w' ≤ M ∧
    M = 60480 * Real.sqrt 249) ∧
  (∃ (x_M y_M z_M v_M w_M : ℝ),
    x_M > 0 ∧ y_M > 0 ∧ z_M > 0 ∧ v_M > 0 ∧ w_M > 0 ∧
    4 * x_M^2 + y_M^2 + z_M^2 + v_M^2 + w_M^2 = 8080 ∧
    x_M * z_M + 4 * y_M * z_M + 6 * z_M * v_M + 14 * z_M * w_M = 60480 * Real.sqrt 249 ∧
    60480 * Real.sqrt 249 + x_M + y_M + z_M + v_M + w_M = 280 + 60600 * Real.sqrt 249) := by
  sorry

end max_value_and_sum_l2386_238629


namespace sum_multiple_special_property_l2386_238605

def is_sum_multiple (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ m = (n / 100 + (n / 10) % 10 + n % 10) ∧ n % m = 0

def digit_sum (n : ℕ) : ℕ :=
  n / 100 + (n / 10) % 10 + n % 10

def F (n : ℕ) : ℕ :=
  max (n / 100 * 10 + (n / 10) % 10) (max (n / 100 * 10 + n % 10) ((n / 10) % 10 * 10 + n % 10))

def G (n : ℕ) : ℕ :=
  min (n / 100 * 10 + (n / 10) % 10) (min (n / 100 * 10 + n % 10) ((n / 10) % 10 * 10 + n % 10))

theorem sum_multiple_special_property :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧
    is_sum_multiple n ∧
    digit_sum n = 12 ∧
    n / 100 > (n / 10) % 10 ∧ (n / 10) % 10 > n % 10 ∧
    (F n + G n) % 16 = 0} =
  {732, 372, 516, 156} := by
  sorry

end sum_multiple_special_property_l2386_238605


namespace max_perimeter_rectangle_l2386_238646

/-- Represents a rectangular enclosure -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The area of the rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The perimeter of the rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: Maximum perimeter of a rectangle with given constraints -/
theorem max_perimeter_rectangle : 
  ∃ (r : Rectangle), 
    area r = 8000 ∧ 
    r.width ≥ 50 ∧
    ∀ (r' : Rectangle), area r' = 8000 ∧ r'.width ≥ 50 → perimeter r' ≤ perimeter r ∧
    r.length = 100 ∧ 
    r.width = 80 ∧ 
    perimeter r = 360 := by
  sorry

end max_perimeter_rectangle_l2386_238646


namespace road_length_for_given_conditions_l2386_238655

/-- Calculates the length of a road given the number of trees, space between trees, and space occupied by each tree. -/
def road_length (num_trees : ℕ) (space_between : ℕ) (tree_space : ℕ) : ℕ :=
  (num_trees * tree_space) + ((num_trees - 1) * space_between)

/-- Theorem stating that for 11 trees, with 14 feet between each tree, and each tree taking 1 foot of space, the road length is 151 feet. -/
theorem road_length_for_given_conditions :
  road_length 11 14 1 = 151 := by
  sorry

end road_length_for_given_conditions_l2386_238655


namespace binomial_15_4_l2386_238688

theorem binomial_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end binomial_15_4_l2386_238688


namespace quadratic_inequality_solution_l2386_238635

theorem quadratic_inequality_solution (a m : ℝ) : 
  (∀ x : ℝ, (a * x^2 + 6 * x - a^2 < 0) ↔ (x < 1 ∨ x > m)) →
  m = 2 := by
sorry

end quadratic_inequality_solution_l2386_238635


namespace tangent_line_to_circle_l2386_238689

theorem tangent_line_to_circle (m : ℝ) :
  (∀ x y : ℝ, 3 * x - 4 * y - 6 = 0 →
    (x^2 + y^2 - 2*y + m = 0 →
      ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 2*y₀ + m = 0 ∧
        3 * x₀ - 4 * y₀ - 6 = 0 ∧
        ∀ (x' y' : ℝ), x'^2 + y'^2 - 2*y' + m = 0 →
          (x' - x₀)^2 + (y' - y₀)^2 > 0)) →
  m = -3 := by
sorry

end tangent_line_to_circle_l2386_238689


namespace power_mul_l2386_238696

theorem power_mul (a : ℝ) (m n : ℕ) : a^m * a^n = a^(m + n) := by sorry

end power_mul_l2386_238696


namespace sufficient_not_necessary_implication_l2386_238608

theorem sufficient_not_necessary_implication (p q : Prop) :
  (p → q) ∧ ¬(q → p) → (¬q → ¬p) ∧ ¬(¬p → ¬q) := by sorry

end sufficient_not_necessary_implication_l2386_238608


namespace counterexamples_count_l2386_238650

def sumOfDigits (n : ℕ) : ℕ := sorry

def hasNoZeroDigit (n : ℕ) : Prop := sorry

def isPrime (n : ℕ) : Prop := sorry

theorem counterexamples_count :
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, sumOfDigits n = 5 ∧ hasNoZeroDigit n ∧ ¬isPrime n) ∧
    (∀ n ∉ S, ¬(sumOfDigits n = 5 ∧ hasNoZeroDigit n ∧ ¬isPrime n)) ∧
    Finset.card S = 6 := by sorry

end counterexamples_count_l2386_238650


namespace negation_of_absolute_value_inequality_l2386_238653

theorem negation_of_absolute_value_inequality :
  (¬ ∀ x : ℝ, |x - 1| ≥ 2) ↔ (∃ x : ℝ, |x - 1| < 2) := by
  sorry

end negation_of_absolute_value_inequality_l2386_238653


namespace orange_harvest_days_l2386_238609

def sacks_per_day : ℕ := 4
def total_sacks : ℕ := 56

theorem orange_harvest_days : 
  total_sacks / sacks_per_day = 14 := by
  sorry

end orange_harvest_days_l2386_238609


namespace rectangle_measurement_error_l2386_238644

/-- Given a rectangle with sides L and W, where one side is measured 14% in excess
    and the other side is measured x% in deficit, resulting in an 8.3% error in the calculated area,
    prove that x = 5. -/
theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) (h_pos_L : L > 0) (h_pos_W : W > 0) : 
  (1.14 * L) * ((1 - 0.01 * x) * W) = 1.083 * (L * W) → x = 5 := by
sorry

end rectangle_measurement_error_l2386_238644


namespace calculation_proof_l2386_238672

theorem calculation_proof :
  (2 * (Real.sqrt 3 - Real.sqrt 5) + 3 * (Real.sqrt 3 + Real.sqrt 5) = 5 * Real.sqrt 3 + Real.sqrt 5) ∧
  (-1^2 - |1 - Real.sqrt 3| + Real.rpow 8 (1/3) - (-3) * Real.sqrt 9 = 11 - Real.sqrt 3) := by
sorry


end calculation_proof_l2386_238672


namespace janet_lives_lost_l2386_238662

/-- The number of lives Janet lost in the hard part of the game -/
def lives_lost : ℕ := sorry

theorem janet_lives_lost :
  (∃ (initial_lives : ℕ) (gained_lives : ℕ),
    initial_lives = 38 ∧
    gained_lives = 32 ∧
    initial_lives - lives_lost + gained_lives = 54) →
  lives_lost = 16 := by sorry

end janet_lives_lost_l2386_238662


namespace greatest_integer_prime_quadratic_l2386_238610

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem greatest_integer_prime_quadratic :
  ∃ (x : ℤ), (∀ y : ℤ, is_prime (Int.natAbs (5 * y^2 - 42 * y + 8)) → y ≤ x) ∧
             is_prime (Int.natAbs (5 * x^2 - 42 * x + 8)) ∧
             x = 5 :=
sorry

end greatest_integer_prime_quadratic_l2386_238610


namespace largest_common_divisor_of_stamp_books_l2386_238633

theorem largest_common_divisor_of_stamp_books : ∃ (n : ℕ), n > 0 ∧ n ∣ 900 ∧ n ∣ 1200 ∧ n ∣ 1500 ∧ ∀ (m : ℕ), m > n → ¬(m ∣ 900 ∧ m ∣ 1200 ∧ m ∣ 1500) := by
  sorry

end largest_common_divisor_of_stamp_books_l2386_238633


namespace rectangle_area_l2386_238645

/-- Given a rectangular plot where the length is thrice the breadth and the breadth is 30 meters,
    prove that the area is 2700 square meters. -/
theorem rectangle_area (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  breadth = 30 →
  length = 3 * breadth →
  area = length * breadth →
  area = 2700 := by
  sorry

end rectangle_area_l2386_238645


namespace min_value_of_f_l2386_238606

def f (x : ℝ) : ℝ := x^2 - 8*x + 15

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x₀ : ℝ), f x₀ = m) ∧ m = -1 :=
by sorry

end min_value_of_f_l2386_238606


namespace other_number_proof_l2386_238630

/-- Given two positive integers with specific HCF and LCM, prove that if one is 24, the other is 156 -/
theorem other_number_proof (A B : ℕ+) : 
  Nat.gcd A B = 12 →
  Nat.lcm A B = 312 →
  A = 24 →
  B = 156 := by
sorry

end other_number_proof_l2386_238630


namespace expected_sum_of_marbles_l2386_238613

-- Define the set of marbles
def marbles : Finset ℕ := Finset.range 7

-- Define the function to calculate the sum of two marbles
def marbleSum (pair : Finset ℕ) : ℕ := Finset.sum pair id

-- Define the set of all possible pairs of marbles
def marblePairs : Finset (Finset ℕ) := marbles.powerset.filter (fun s => s.card = 2)

-- Statement of the theorem
theorem expected_sum_of_marbles :
  (Finset.sum marblePairs marbleSum) / marblePairs.card = 52 / 7 := by
sorry

end expected_sum_of_marbles_l2386_238613


namespace duck_pond_problem_l2386_238637

theorem duck_pond_problem (initial_ducks : ℕ) (final_ducks : ℕ) 
  (h1 : initial_ducks = 320)
  (h2 : final_ducks = 140) : 
  ∃ (F : ℚ),
    F = 1/6 ∧
    final_ducks = (initial_ducks * 3/4 * (1 - F) * 0.7).floor := by
  sorry

end duck_pond_problem_l2386_238637


namespace sophie_donuts_l2386_238618

/-- The number of donuts left for Sophie after buying boxes and giving some away. -/
def donuts_left (total_boxes : ℕ) (donuts_per_box : ℕ) (boxes_given : ℕ) (donuts_given : ℕ) : ℕ :=
  (total_boxes - boxes_given) * donuts_per_box - donuts_given

/-- Theorem stating that Sophie has 30 donuts left. -/
theorem sophie_donuts :
  donuts_left 4 12 1 6 = 30 := by
  sorry

end sophie_donuts_l2386_238618


namespace carl_lemonade_sales_l2386_238681

/-- 
Given:
- Stanley sells 4 cups of lemonade per hour
- Carl sells some cups of lemonade per hour
- Carl sold 9 more cups than Stanley in 3 hours

Prove that Carl sold 7 cups of lemonade per hour
-/
theorem carl_lemonade_sales (stanley_rate : ℕ) (carl_rate : ℕ) (hours : ℕ) (difference : ℕ) :
  stanley_rate = 4 →
  hours = 3 →
  difference = 9 →
  carl_rate * hours = stanley_rate * hours + difference →
  carl_rate = 7 :=
by
  sorry

#check carl_lemonade_sales

end carl_lemonade_sales_l2386_238681


namespace circle_properties_l2386_238640

-- Define the circle C
def Circle (t s r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - t)^2 + (p.2 - s)^2 = r^2}

-- Define the condition for independence of x₀ and y₀
def IndependentSum (t s r a b : ℝ) : Prop :=
  ∀ (x₀ y₀ : ℝ), (x₀, y₀) ∈ Circle t s r →
    ∃ (k : ℝ), |x₀ - y₀ + a| + |x₀ - y₀ + b| = k

-- Main theorem
theorem circle_properties
  (t s r a b : ℝ)
  (h_r : r > 0)
  (h_ab : a ≠ b)
  (h_ind : IndependentSum t s r a b) :
  (|a - b| = 2 * Real.sqrt 2 * r →
    ∃ (m n : ℝ), ∀ (x y : ℝ), (x, y) ∈ Circle t s r → m * x + n * y = 1) ∧
  (|a - b| = 2 * Real.sqrt 2 →
    r ≤ 1 ∧ ∃ (t₀ s₀ : ℝ), r = 1 ∧ (t₀, s₀) ∈ Circle t s r) :=
sorry

end circle_properties_l2386_238640


namespace graph_quadrants_l2386_238674

def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x - k

theorem graph_quadrants (k : ℝ) (h : k < 0) :
  ∃ (x₁ x₂ x₃ : ℝ), 
    (x₁ > 0 ∧ linear_function k x₁ > 0) ∧  -- Quadrant I
    (x₂ < 0 ∧ linear_function k x₂ > 0) ∧  -- Quadrant II
    (x₃ > 0 ∧ linear_function k x₃ < 0)    -- Quadrant IV
  := by sorry

end graph_quadrants_l2386_238674


namespace expression_evaluation_l2386_238664

theorem expression_evaluation : 
  |(-1/2 : ℝ)| + ((-27 : ℝ) ^ (1/3 : ℝ)) - (1/4 : ℝ).sqrt + (12 : ℝ).sqrt * (3 : ℝ).sqrt = 3 := by
  sorry

end expression_evaluation_l2386_238664


namespace february_production_l2386_238617

/-- Represents the monthly carrot cake production sequence -/
def carrotCakeSequence : ℕ → ℕ
| 0 => 19  -- October (0-indexed)
| n + 1 => carrotCakeSequence n + 2

/-- Theorem stating that the 5th term (February) of the sequence is 27 -/
theorem february_production : carrotCakeSequence 4 = 27 := by
  sorry

end february_production_l2386_238617


namespace rhombus_perimeter_l2386_238628

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 30) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * s = 4 * Real.sqrt 241 := by
  sorry

end rhombus_perimeter_l2386_238628


namespace fourth_term_is_eight_l2386_238665

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- The sum function
  first_term : a 1 = -1
  sum_property : ∀ n, S n = n * (a 1 + a n) / 2
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The theorem stating that a_4 = 8 given the conditions -/
theorem fourth_term_is_eight (seq : ArithmeticSequence) (sum_4 : seq.S 4 = 14) :
  seq.a 4 = 8 := by
  sorry

end fourth_term_is_eight_l2386_238665


namespace stock_percentage_change_l2386_238676

theorem stock_percentage_change 
  (initial_value : ℝ) 
  (day1_decrease_rate : ℝ) 
  (day2_increase_rate : ℝ) 
  (h1 : day1_decrease_rate = 0.3) 
  (h2 : day2_increase_rate = 0.4) : 
  (initial_value - (initial_value * (1 - day1_decrease_rate) * (1 + day2_increase_rate))) / initial_value = 0.02 := by
  sorry

end stock_percentage_change_l2386_238676


namespace max_product_given_sum_l2386_238604

theorem max_product_given_sum (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 40 → ∀ x y : ℝ, x > 0 → y > 0 → x + y = 40 → x * y ≤ a * b → a * b ≤ 400 := by
  sorry

end max_product_given_sum_l2386_238604


namespace real_pair_existence_l2386_238619

theorem real_pair_existence :
  (∃ (u v : ℝ), (∃ (q : ℚ), u + v = q) ∧ 
    (∀ (n : ℕ), n ≥ 2 → ∀ (q : ℚ), u^n + v^n ≠ q)) ∧
  (¬ ∃ (u v : ℝ), (∀ (q : ℚ), u + v ≠ q) ∧ 
    (∀ (n : ℕ), n ≥ 2 → ∃ (q : ℚ), u^n + v^n = q)) :=
by sorry

end real_pair_existence_l2386_238619


namespace min_pieces_is_3n_plus_1_l2386_238631

/-- A rectangular sheet of paper with holes -/
structure PerforatedSheet :=
  (n : ℕ)  -- number of holes
  (noOverlap : Bool)  -- holes do not overlap
  (parallelSides : Bool)  -- holes' sides are parallel to sheet edges

/-- The minimum number of rectangular pieces a perforated sheet can be divided into -/
def minPieces (sheet : PerforatedSheet) : ℕ :=
  3 * sheet.n + 1

/-- Theorem: The minimum number of rectangular pieces is 3n + 1 -/
theorem min_pieces_is_3n_plus_1 (sheet : PerforatedSheet) 
  (h1 : sheet.noOverlap = true) 
  (h2 : sheet.parallelSides = true) : 
  minPieces sheet = 3 * sheet.n + 1 := by
  sorry

end min_pieces_is_3n_plus_1_l2386_238631


namespace divide_by_sqrt_two_l2386_238625

theorem divide_by_sqrt_two : 2 / Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end divide_by_sqrt_two_l2386_238625


namespace age_difference_l2386_238649

theorem age_difference (frank_age john_age : ℕ) : 
  (frank_age + 4 = 16) → 
  (john_age + 3 = 2 * (frank_age + 3)) → 
  (john_age - frank_age = 15) :=
by sorry

end age_difference_l2386_238649


namespace intersection_intercept_sum_l2386_238639

/-- Given two lines that intersect at (3, 6), prove their y-intercepts sum to -9 -/
theorem intersection_intercept_sum (c d : ℝ) : 
  (3 = 2 * 6 + c) →  -- First line passes through (3, 6)
  (6 = 2 * 3 + d) →  -- Second line passes through (3, 6)
  c + d = -9 := by
sorry

end intersection_intercept_sum_l2386_238639


namespace camp_provisions_duration_l2386_238624

/-- Represents the camp provisions problem -/
theorem camp_provisions_duration (initial_men_1 initial_men_2 : ℕ) 
  (initial_days_1 initial_days_2 : ℕ) (additional_men : ℕ) 
  (consumption_rate : ℚ) (days_before_supply : ℕ) 
  (supply_men supply_days : ℕ) : 
  initial_men_1 = 800 →
  initial_men_2 = 200 →
  initial_days_1 = 20 →
  initial_days_2 = 10 →
  additional_men = 200 →
  consumption_rate = 3/2 →
  days_before_supply = 10 →
  supply_men = 300 →
  supply_days = 15 →
  ∃ (remaining_days : ℚ), 
    remaining_days > 7.30 ∧ 
    remaining_days < 7.32 ∧
    remaining_days = 
      (initial_men_1 * initial_days_1 + initial_men_2 * initial_days_2 - 
       (initial_men_1 + initial_men_2 + additional_men * consumption_rate) * days_before_supply +
       supply_men * supply_days) / 
      (initial_men_1 + initial_men_2 + additional_men * consumption_rate) :=
by
  sorry

end camp_provisions_duration_l2386_238624


namespace parallelogram_area_example_l2386_238638

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base : ℝ) (height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 12 cm and height 10 cm is 120 square centimeters -/
theorem parallelogram_area_example : parallelogram_area 12 10 = 120 := by
  sorry

end parallelogram_area_example_l2386_238638


namespace largest_common_term_l2386_238660

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem largest_common_term :
  ∃ (n m : ℕ),
    179 = arithmetic_sequence 3 8 n ∧
    179 = arithmetic_sequence 5 9 m ∧
    179 ≤ 200 ∧
    ∀ (k : ℕ), k > 179 →
      k ≤ 200 →
      (∀ (p q : ℕ), k ≠ arithmetic_sequence 3 8 p ∨ k ≠ arithmetic_sequence 5 9 q) :=
by sorry

end largest_common_term_l2386_238660


namespace eight_solutions_of_g_fourth_composition_l2386_238686

/-- The function g(x) = x^2 - 3x -/
def g (x : ℝ) : ℝ := x^2 - 3*x

/-- The theorem stating that there are exactly 8 distinct real numbers d such that g(g(g(g(d)))) = 2 -/
theorem eight_solutions_of_g_fourth_composition :
  ∃! (s : Finset ℝ), (∀ d ∈ s, g (g (g (g d))) = 2) ∧ s.card = 8 := by
  sorry

end eight_solutions_of_g_fourth_composition_l2386_238686


namespace psychological_survey_selection_l2386_238675

theorem psychological_survey_selection (boys girls selected : ℕ) : 
  boys = 4 → girls = 2 → selected = 4 →
  (Nat.choose (boys + girls) selected) - (Nat.choose boys selected) = 14 :=
by sorry

end psychological_survey_selection_l2386_238675


namespace triangle_piece_count_l2386_238680

/-- Calculate the sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Calculate the sum of the first n natural numbers -/
def triangle_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The number of rows in the rod triangle -/
def rod_rows : ℕ := 10

/-- The number of rows in the connector triangle -/
def connector_rows : ℕ := rod_rows + 1

/-- The first term of the rod arithmetic sequence -/
def first_rod_count : ℕ := 3

/-- The common difference of the rod arithmetic sequence -/
def rod_increment : ℕ := 3

theorem triangle_piece_count : 
  arithmetic_sum first_rod_count rod_increment rod_rows + 
  triangle_number connector_rows = 231 := by
  sorry

end triangle_piece_count_l2386_238680


namespace rebus_solution_l2386_238643

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def are_distinct (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

theorem rebus_solution (a p m i r : ℕ) :
  is_digit a ∧ is_digit p ∧ is_digit m ∧ is_digit i ∧ is_digit r ∧
  are_distinct a p m i r ∧
  (10 * a + p) ^ m = 100 * m + 10 * i + r →
  a = 1 ∧ p = 6 ∧ m = 2 ∧ i = 5 ∧ r = 6 := by
  sorry

end rebus_solution_l2386_238643


namespace frac_repeating_block_length_l2386_238641

/-- The least number of digits in a repeating block of the decimal expansion of 7/13 -/
def repeating_block_length : ℕ := 6

/-- 7/13 is a rational number -/
def frac : ℚ := 7 / 13

theorem frac_repeating_block_length : 
  ∃ (n : ℕ) (k : ℕ+) (a b : ℕ), 
    frac * 10^n = (a : ℚ) + (b : ℚ) / (10^repeating_block_length - 1) ∧
    b < 10^repeating_block_length - 1 ∧
    ∀ m < repeating_block_length, 
      ¬∃ (c d : ℕ), frac * 10^n = (c : ℚ) + (d : ℚ) / (10^m - 1) ∧ d < 10^m - 1 :=
sorry

end frac_repeating_block_length_l2386_238641


namespace angle_between_m_and_n_l2386_238611

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (9, 12)
def c : ℝ × ℝ := (4, -3)
def m : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)
def n : ℝ × ℝ := (a.1 + c.1, a.2 + c.2)

theorem angle_between_m_and_n :
  Real.arccos ((m.1 * n.1 + m.2 * n.2) / (Real.sqrt (m.1^2 + m.2^2) * Real.sqrt (n.1^2 + n.2^2))) = 3 * Real.pi / 4 := by
  sorry

end angle_between_m_and_n_l2386_238611


namespace triangle_area_inequalities_l2386_238656

/-- The area of a triangle ABC with sides a and b is less than or equal to both
    (1/2)(a² - ab + b²) and ((a + b)/(2√2))² -/
theorem triangle_area_inequalities (a b : ℝ) (hpos : 0 < a ∧ 0 < b) :
  let area := (1/2) * a * b * Real.sin C
  ∃ C, 0 ≤ C ∧ C ≤ π ∧
    area ≤ (1/2) * (a^2 - a*b + b^2) ∧
    area ≤ ((a + b)/(2 * Real.sqrt 2))^2 := by
  sorry

end triangle_area_inequalities_l2386_238656


namespace inscribed_sphere_volume_l2386_238623

/-- The volume of a sphere inscribed in a cube with edge length 8 inches -/
theorem inscribed_sphere_volume (π : ℝ) :
  let cube_edge : ℝ := 8
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3
  sphere_volume = 256 * π / 3 := by sorry

end inscribed_sphere_volume_l2386_238623


namespace garden_area_proof_l2386_238647

theorem garden_area_proof (total_posts : ℕ) (post_spacing : ℕ) :
  total_posts = 24 →
  post_spacing = 6 →
  ∃ (short_posts long_posts : ℕ),
    short_posts + long_posts = total_posts / 2 ∧
    long_posts = 3 * short_posts ∧
    (short_posts - 1) * post_spacing * (long_posts - 1) * post_spacing = 576 :=
by sorry

end garden_area_proof_l2386_238647


namespace fixed_point_satisfies_equation_fixed_point_is_unique_l2386_238685

/-- The line equation passing through a fixed point for all real values of a -/
def line_equation (a x y : ℝ) : Prop :=
  (a - 1) * x - y + 2 * a + 1 = 0

/-- The fixed point coordinates -/
def fixed_point : ℝ × ℝ := (-2, 3)

/-- Theorem stating that the fixed point satisfies the line equation for all real a -/
theorem fixed_point_satisfies_equation :
  ∀ (a : ℝ), line_equation a (fixed_point.1) (fixed_point.2) :=
by sorry

/-- Theorem stating that the fixed point is unique -/
theorem fixed_point_is_unique :
  ∀ (x y : ℝ), (∀ (a : ℝ), line_equation a x y) → (x, y) = fixed_point :=
by sorry

end fixed_point_satisfies_equation_fixed_point_is_unique_l2386_238685


namespace largest_negative_integer_solution_l2386_238687

theorem largest_negative_integer_solution :
  ∃ (x : ℝ), x = -1 ∧ 
  x < 0 ∧
  |x - 1| > 1 ∧
  (x - 2) / x > 0 ∧
  (x - 2) / x > |x - 1| ∧
  ∀ (y : ℤ), y < 0 → 
    (y < x ∨ ¬(|y - 1| > 1) ∨ ¬((y - 2) / y > 0) ∨ ¬((y - 2) / y > |y - 1|)) :=
by sorry

end largest_negative_integer_solution_l2386_238687


namespace pond_A_has_more_fish_l2386_238658

-- Define the capture-recapture estimation function
def estimateFishPopulation (totalSecondCatch : ℕ) (totalMarkedReleased : ℕ) (markedInSecondCatch : ℕ) : ℚ :=
  (totalSecondCatch * totalMarkedReleased : ℚ) / markedInSecondCatch

-- Define the parameters for each pond
def pondAMarkedFish : ℕ := 8
def pondBMarkedFish : ℕ := 16
def fishCaught : ℕ := 200
def fishMarked : ℕ := 200

-- Theorem statement
theorem pond_A_has_more_fish :
  estimateFishPopulation fishCaught fishMarked pondAMarkedFish >
  estimateFishPopulation fishCaught fishMarked pondBMarkedFish :=
by
  sorry

end pond_A_has_more_fish_l2386_238658


namespace division_remainder_l2386_238661

theorem division_remainder (dividend quotient divisor remainder : ℕ) : 
  dividend = 507 → 
  quotient = 61 → 
  divisor = 8 → 
  dividend = divisor * quotient + remainder → 
  remainder = 19 := by
  sorry

end division_remainder_l2386_238661


namespace original_denominator_problem_l2386_238612

theorem original_denominator_problem (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 →
  (11 : ℚ) / (d + 8) = 2 / 5 →
  d = 39 / 2 :=
by sorry

end original_denominator_problem_l2386_238612


namespace converse_square_sum_nonzero_l2386_238698

theorem converse_square_sum_nonzero (x y : ℝ) : 
  (x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0 := by
  sorry

end converse_square_sum_nonzero_l2386_238698


namespace paige_homework_pages_l2386_238692

/-- Given the total number of homework problems, the number of finished problems,
    and the number of problems per page, calculate the number of remaining pages. -/
def remaining_pages (total_problems : ℕ) (finished_problems : ℕ) (problems_per_page : ℕ) : ℕ :=
  (total_problems - finished_problems) / problems_per_page

/-- Theorem stating that for Paige's homework scenario, the number of remaining pages is 7. -/
theorem paige_homework_pages : remaining_pages 110 47 9 = 7 := by
  sorry

end paige_homework_pages_l2386_238692


namespace sqrt_product_l2386_238602

theorem sqrt_product (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) : Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) := by
  sorry

end sqrt_product_l2386_238602
