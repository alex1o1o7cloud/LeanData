import Mathlib

namespace journey_speed_problem_l3154_315408

theorem journey_speed_problem (total_distance : ℝ) (total_time : ℝ) 
  (speed1 : ℝ) (speed2 : ℝ) (segment_time : ℝ) :
  total_distance = 150 →
  total_time = 2 →
  speed1 = 50 →
  speed2 = 70 →
  segment_time = 2/3 →
  ∃ (speed3 : ℝ),
    speed3 = 105 ∧
    total_distance = speed1 * segment_time + speed2 * segment_time + speed3 * segment_time :=
by sorry

end journey_speed_problem_l3154_315408


namespace initial_mixture_volume_l3154_315481

/-- Proves that given a mixture with 20% water content, if adding 8.333333333333334 gallons
    of water increases the water percentage to 25%, then the initial volume of the mixture
    is 125 gallons. -/
theorem initial_mixture_volume
  (initial_water_percentage : ℝ)
  (added_water : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_water_percentage = 0.20)
  (h2 : added_water = 8.333333333333334)
  (h3 : final_water_percentage = 0.25)
  (h4 : ∀ v : ℝ, final_water_percentage * (v + added_water) = initial_water_percentage * v + added_water) :
  ∃ v : ℝ, v = 125 := by
  sorry

end initial_mixture_volume_l3154_315481


namespace nut_mixture_weight_l3154_315448

/-- Represents a mixture of nuts -/
structure NutMixture where
  almond_ratio : ℚ
  walnut_ratio : ℚ
  almond_weight : ℚ

/-- Calculates the total weight of a nut mixture -/
def total_weight (mix : NutMixture) : ℚ :=
  (mix.almond_weight / mix.almond_ratio) * (mix.almond_ratio + mix.walnut_ratio)

/-- Theorem: The total weight of the given nut mixture is 210 pounds -/
theorem nut_mixture_weight :
  let mix : NutMixture := {
    almond_ratio := 5,
    walnut_ratio := 2,
    almond_weight := 150
  }
  total_weight mix = 210 := by
  sorry

end nut_mixture_weight_l3154_315448


namespace parallel_vectors_x_value_l3154_315413

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (2*x, -3)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_x_value :
  parallel vector_a (vector_b x) → x = -3/4 :=
by
  sorry

end parallel_vectors_x_value_l3154_315413


namespace binomial_coefficient_sum_l3154_315468

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 2187 := by
sorry

end binomial_coefficient_sum_l3154_315468


namespace cubic_inequality_solution_l3154_315471

theorem cubic_inequality_solution (x : ℝ) : 
  -2 * x^3 + 5 * x^2 + 7 * x - 10 < 0 ↔ x < -1.35 ∨ (1.85 < x ∧ x < 2) :=
by sorry

end cubic_inequality_solution_l3154_315471


namespace square_measurement_unit_l3154_315455

/-- Given a square with sides of length 5 units and an actual area of at least 20.25 square centimeters,
    prove that the length of one unit in this measurement system is 0.9 centimeters. -/
theorem square_measurement_unit (side_length : ℝ) (actual_area : ℝ) :
  side_length = 5 →
  actual_area ≥ 20.25 →
  actual_area = (side_length * 0.9) ^ 2 :=
by sorry

end square_measurement_unit_l3154_315455


namespace second_race_length_is_600_l3154_315428

/-- Represents a race between three runners A, B, and C -/
structure Race where
  length : ℝ
  a_beats_b : ℝ
  a_beats_c : ℝ

/-- Calculates the length of a second race given the first race data -/
def second_race_length (first_race : Race) (b_beats_c : ℝ) : ℝ :=
  600

/-- Theorem stating that given the conditions of the first race and the fact that B beats C by 60m in the second race, the length of the second race is 600m -/
theorem second_race_length_is_600 (first_race : Race) (h1 : first_race.length = 200) 
    (h2 : first_race.a_beats_b = 20) (h3 : first_race.a_beats_c = 38) (h4 : b_beats_c = 60) : 
    second_race_length first_race b_beats_c = 600 := by
  sorry

end second_race_length_is_600_l3154_315428


namespace quadratic_form_equivalence_l3154_315447

theorem quadratic_form_equivalence :
  ∀ x : ℝ, 2 * x^2 + 3 * x - 1 = 2 * (x + 3/4)^2 - 17/8 :=
by
  sorry

end quadratic_form_equivalence_l3154_315447


namespace mens_wages_l3154_315462

theorem mens_wages (total_earnings : ℝ) (num_men : ℕ) (num_boys : ℕ)
  (h_total : total_earnings = 432)
  (h_men : num_men = 15)
  (h_boys : num_boys = 12)
  (h_equal_earnings : ∃ (num_women : ℕ), num_men * (total_earnings / (num_men + num_women + num_boys)) = 
                                         num_women * (total_earnings / (num_men + num_women + num_boys)) ∧
                                         num_women * (total_earnings / (num_men + num_women + num_boys)) = 
                                         num_boys * (total_earnings / (num_men + num_women + num_boys))) :
  num_men * (total_earnings / (num_men + num_men + num_men)) = 144 := by
  sorry

end mens_wages_l3154_315462


namespace min_intersection_at_45_deg_l3154_315498

/-- A square in 2D space -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- Rotation of a square around its center -/
def rotate_square (s : Square) (angle : ℝ) : Square :=
  { s with }  -- The internal structure remains the same after rotation

/-- The area of intersection between two squares -/
def intersection_area (s1 s2 : Square) : ℝ := sorry

/-- Theorem: The area of intersection between a square and its rotated version is minimized at 45 degrees -/
theorem min_intersection_at_45_deg (s : Square) :
  ∀ x : ℝ, 0 ≤ x → x ≤ 2 * π →
    intersection_area s (rotate_square s (π/4)) ≤ intersection_area s (rotate_square s x) := by
  sorry

#check min_intersection_at_45_deg

end min_intersection_at_45_deg_l3154_315498


namespace extreme_value_at_zero_decreasing_on_interval_l3154_315449

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (3 * x^2 + a * x) / Real.exp x

-- Theorem for the first part of the problem
theorem extreme_value_at_zero (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x| ∧ |x| < ε → f a 0 ≥ f a x) ∨
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x| ∧ |x| < ε → f a 0 ≤ f a x) ↔
  a = 0 :=
sorry

-- Theorem for the second part of the problem
theorem decreasing_on_interval (a : ℝ) :
  (∀ (x y : ℝ), 3 ≤ x ∧ x < y → f a x > f a y) ↔
  a ≥ -9/2 :=
sorry

end extreme_value_at_zero_decreasing_on_interval_l3154_315449


namespace target_hitting_probability_l3154_315444

theorem target_hitting_probability : 
  let p_single_hit : ℚ := 1/2
  let total_shots : ℕ := 7
  let total_hits : ℕ := 3
  let consecutive_hits : ℕ := 2

  -- Probability of exactly 3 hits out of 7 shots
  let p_total_hits : ℚ := (Nat.choose total_shots total_hits : ℚ) * p_single_hit ^ total_shots

  -- Number of ways to arrange 2 consecutive hits out of 3 in 7 shots
  let arrangements : ℕ := Nat.descFactorial (total_shots - consecutive_hits) consecutive_hits

  -- Final probability
  (arrangements : ℚ) * p_single_hit ^ total_shots = 5/32 :=
by sorry

end target_hitting_probability_l3154_315444


namespace d_value_for_four_roots_l3154_315489

/-- The polynomial Q(x) -/
def Q (d : ℝ) (x : ℝ) : ℝ := (x^2 - 3*x + 5) * (x^2 - d*x + 7) * (x^2 - 6*x + 18)

/-- The number of distinct roots of Q(x) -/
def distinctRoots (d : ℝ) : ℕ := sorry

/-- Theorem stating that |d| = 9 when Q(x) has exactly 4 distinct roots -/
theorem d_value_for_four_roots :
  ∃ d : ℝ, distinctRoots d = 4 ∧ |d| = 9 := by sorry

end d_value_for_four_roots_l3154_315489


namespace toy_poodle_height_l3154_315420

/-- The height of the toy poodle given the heights of other poodle types -/
theorem toy_poodle_height (h_standard : ℕ) (h_mini : ℕ) (h_toy : ℕ)
  (standard_mini : h_standard = h_mini + 8)
  (mini_toy : h_mini = h_toy + 6)
  (standard_height : h_standard = 28) :
  h_toy = 14 := by
  sorry

end toy_poodle_height_l3154_315420


namespace correct_sample_l3154_315485

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- Represents a sample of student numbers --/
def Sample := List Nat

/-- Checks if a number is a valid student number (between 1 and 50) --/
def isValidStudentNumber (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 50

/-- Selects a sample of distinct student numbers from the random number table --/
def selectSample (table : RandomNumberTable) (startRow : Nat) (startCol : Nat) (sampleSize : Nat) : Sample :=
  sorry

/-- The specific random number table given in the problem --/
def givenTable : RandomNumberTable :=
  [[03, 47, 43, 73, 86, 36, 96, 47, 36, 61, 46, 98, 63, 71, 62, 33, 26, 16, 80],
   [45, 60, 11, 14, 10, 95, 97, 74, 24, 67, 62, 42, 81, 14, 57, 20, 42, 53],
   [32, 37, 32, 27, 07, 36, 07, 51, 24, 51, 79, 89, 73, 16, 76, 62, 27, 66],
   [56, 50, 26, 71, 07, 32, 90, 79, 78, 53, 13, 55, 38, 58, 59, 88, 97, 54],
   [14, 10, 12, 56, 85, 99, 26, 96, 96, 68, 27, 31, 05, 03, 72, 93, 15, 57],
   [12, 10, 14, 21, 88, 26, 49, 81, 76, 55, 59, 56, 35, 64, 38, 54, 82, 46],
   [22, 31, 62, 43, 09, 90, 06, 18, 44, 32, 53, 23, 83, 01, 30, 30]]

theorem correct_sample :
  selectSample givenTable 3 6 5 = [22, 2, 10, 29, 7] :=
sorry

end correct_sample_l3154_315485


namespace sin_plus_sin_alpha_nonperiodic_l3154_315438

/-- A function f is periodic if there exists a non-zero real number T such that f(x + T) = f(x) for all x -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f (x + T) = f x

/-- The main theorem: for any positive irrational α, the function f(x) = sin x + sin(αx) is non-periodic -/
theorem sin_plus_sin_alpha_nonperiodic (α : ℝ) (h_pos : α > 0) (h_irr : Irrational α) :
  ¬IsPeriodic (fun x ↦ Real.sin x + Real.sin (α * x)) := by
  sorry


end sin_plus_sin_alpha_nonperiodic_l3154_315438


namespace readers_overlap_l3154_315461

theorem readers_overlap (total : ℕ) (science_fiction : ℕ) (literary : ℕ) 
  (h1 : total = 650) 
  (h2 : science_fiction = 250) 
  (h3 : literary = 550) : 
  total = science_fiction + literary - 150 := by
  sorry

#check readers_overlap

end readers_overlap_l3154_315461


namespace tangent_line_and_extrema_l3154_315417

noncomputable def f (x : ℝ) : ℝ := (x - 1) / x * Real.log x

theorem tangent_line_and_extrema :
  let a := (1 : ℝ) / 4
  let b := Real.exp 1
  ∃ (tl : ℝ → ℝ) (max_val min_val : ℝ),
    (∀ x, tl x = 2 * x - 2 + Real.log 2) ∧
    (∀ x ∈ Set.Icc a b, f x ≤ max_val) ∧
    (∃ x ∈ Set.Icc a b, f x = max_val) ∧
    (∀ x ∈ Set.Icc a b, min_val ≤ f x) ∧
    (∃ x ∈ Set.Icc a b, f x = min_val) ∧
    max_val = 0 ∧
    min_val = Real.log 4 - 3 ∧
    (HasDerivAt f 2 (1/2) ∧ f (1/2) = -1 + Real.log 2) :=
by sorry

end tangent_line_and_extrema_l3154_315417


namespace gcd_of_B_is_two_l3154_315495

def B : Set ℕ := {n | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2) ∧ x > 0}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 :=
sorry

end gcd_of_B_is_two_l3154_315495


namespace area_of_triangle_DEF_l3154_315487

-- Define the square PQRS
def PQRS_area : ℝ := 36

-- Define the side length of smaller squares
def small_square_side : ℝ := 2

-- Define the triangle DEF
structure Triangle_DEF where
  DE : ℝ
  DF : ℝ
  EF : ℝ

-- Define the folding property
def folds_to_center (t : Triangle_DEF) (s : ℝ) : Prop :=
  t.DE = t.DF ∧ t.DE = s / 2 + 2 * small_square_side

-- Main theorem
theorem area_of_triangle_DEF (t : Triangle_DEF) (s : ℝ) :
  s^2 = PQRS_area →
  folds_to_center t s →
  t.EF = s - 2 * small_square_side →
  (1/2) * t.EF * t.DE = 10 := by sorry

end area_of_triangle_DEF_l3154_315487


namespace five_mondays_in_march_l3154_315431

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a leap year with five Sundays in February -/
structure LeapYearWithFiveSundaysInFebruary :=
  (isLeapYear : Bool)
  (februaryHasFiveSundays : Bool)

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

theorem five_mondays_in_march 
  (year : LeapYearWithFiveSundaysInFebruary) : 
  ∃ (mondayCount : Nat), mondayCount = 5 ∧ 
  (∀ (d : DayOfWeek), d ≠ DayOfWeek.Monday → 
    ∃ (otherCount : Nat), otherCount < 5) :=
by sorry

end five_mondays_in_march_l3154_315431


namespace females_chose_malt_cheerleader_malt_choice_l3154_315435

/-- Represents the group of cheerleaders -/
structure CheerleaderGroup where
  total : Nat
  males : Nat
  females : Nat
  malt_choosers : Nat
  coke_choosers : Nat
  male_malt_choosers : Nat

/-- The theorem to prove -/
theorem females_chose_malt (group : CheerleaderGroup) : Nat :=
  let female_malt_choosers := group.malt_choosers - group.male_malt_choosers
  female_malt_choosers

/-- The main theorem stating the conditions and the result to prove -/
theorem cheerleader_malt_choice : ∃ (group : CheerleaderGroup), 
  group.total = 26 ∧
  group.males = 10 ∧
  group.females = 16 ∧
  group.malt_choosers = 2 * group.coke_choosers ∧
  group.male_malt_choosers = 6 ∧
  females_chose_malt group = 10 := by
  sorry


end females_chose_malt_cheerleader_malt_choice_l3154_315435


namespace sports_club_overlap_l3154_315496

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) :
  total = 30 →
  badminton = 17 →
  tennis = 19 →
  neither = 2 →
  ∃ (both : ℕ), both = 8 ∧
    total = badminton + tennis - both + neither :=
by sorry

end sports_club_overlap_l3154_315496


namespace apple_pear_ratio_l3154_315402

theorem apple_pear_ratio (apples oranges pears : ℕ) 
  (h1 : oranges = 3 * apples) 
  (h2 : pears = 4 * oranges) : 
  apples = (1 : ℚ) / 12 * pears :=
by sorry

end apple_pear_ratio_l3154_315402


namespace clock_time_after_2016_hours_l3154_315425

theorem clock_time_after_2016_hours (current_time : ℕ) (hours_passed : ℕ) : 
  current_time = 7 → hours_passed = 2016 → (current_time + hours_passed) % 12 = 7 := by
  sorry

end clock_time_after_2016_hours_l3154_315425


namespace village_population_percentage_l3154_315456

theorem village_population_percentage (total : ℕ) (part : ℕ) (h1 : total = 9000) (h2 : part = 8100) :
  (part : ℚ) / total * 100 = 90 := by
  sorry

end village_population_percentage_l3154_315456


namespace angle_bisector_length_l3154_315483

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = 5 ∧ BC = 12 ∧ AC = 13

-- Define the angle bisector BE
def angle_bisector (A B C E : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let BE := Real.sqrt ((B.1 - E.1)^2 + (B.2 - E.2)^2)
  let AE := Real.sqrt ((A.1 - E.1)^2 + (A.2 - E.2)^2)
  let CE := Real.sqrt ((C.1 - E.1)^2 + (C.2 - E.2)^2)
  AE / AB = CE / BC

-- Theorem statement
theorem angle_bisector_length 
  (A B C E : ℝ × ℝ) 
  (h1 : triangle_ABC A B C) 
  (h2 : angle_bisector A B C E) :
  let BE := Real.sqrt ((B.1 - E.1)^2 + (B.2 - E.2)^2)
  ∃ m : ℝ, BE = m * Real.sqrt 2 ∧ m = Real.sqrt 138 / 4 := by
  sorry

end angle_bisector_length_l3154_315483


namespace simplify_part1_simplify_part2_l3154_315459

-- Part 1
theorem simplify_part1 (x : ℝ) (h : x ≠ -2) :
  x^2 / (x + 2) + (4*x + 4) / (x + 2) = x + 2 := by sorry

-- Part 2
theorem simplify_part2 (x : ℝ) (h : x ≠ 1) :
  x^2 / (x^2 - 2*x + 1) / ((1 - 2*x) / (x - 1) - x + 1) = -1 / (x - 1) := by sorry

end simplify_part1_simplify_part2_l3154_315459


namespace no_reassignment_possible_l3154_315464

/-- Represents a classroom with rows and columns of chairs -/
structure Classroom where
  rows : Nat
  columns : Nat

/-- Represents the total number of chairs in the classroom -/
def Classroom.totalChairs (c : Classroom) : Nat :=
  c.rows * c.columns

/-- Represents the number of occupied chairs -/
def Classroom.occupiedChairs (c : Classroom) (students : Nat) : Nat :=
  students

/-- Represents whether a reassignment is possible -/
def isReassignmentPossible (c : Classroom) (students : Nat) : Prop :=
  ∃ (redChairs blackChairs : Nat),
    redChairs + blackChairs = c.totalChairs - 1 ∧
    redChairs = students ∧
    blackChairs > redChairs

theorem no_reassignment_possible (c : Classroom) (students : Nat) :
  c.rows = 5 →
  c.columns = 7 →
  students = 34 →
  ¬ isReassignmentPossible c students :=
sorry

end no_reassignment_possible_l3154_315464


namespace dairy_farmer_june_income_l3154_315440

/-- Calculates the total income for a dairy farmer in June -/
theorem dairy_farmer_june_income 
  (daily_production : ℕ) 
  (price_per_gallon : ℚ) 
  (days_in_june : ℕ) 
  (h1 : daily_production = 200)
  (h2 : price_per_gallon = 355/100)
  (h3 : days_in_june = 30) :
  daily_production * days_in_june * price_per_gallon = 21300 := by
sorry

end dairy_farmer_june_income_l3154_315440


namespace obtuse_angle_range_l3154_315445

/-- Two vectors form an obtuse angle if their dot product is negative -/
def obtuse_angle (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 < 0

/-- The theorem stating the range of m for which vectors a and b form an obtuse angle -/
theorem obtuse_angle_range :
  ∀ m : ℝ, obtuse_angle (-2, 3) (1, m) ↔ m < 2/3 ∧ m ≠ -3/2 := by
  sorry

end obtuse_angle_range_l3154_315445


namespace non_empty_proper_subsets_of_A_l3154_315454

def A : Set ℕ := {2, 3}

theorem non_empty_proper_subsets_of_A :
  {s : Set ℕ | s ⊆ A ∧ s ≠ ∅ ∧ s ≠ A} = {{2}, {3}} := by sorry

end non_empty_proper_subsets_of_A_l3154_315454


namespace trig_fraction_value_l3154_315407

theorem trig_fraction_value (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (2 * Real.cos α + Real.sin α) = 3/4 := by
  sorry

end trig_fraction_value_l3154_315407


namespace joe_paint_usage_l3154_315469

/-- The amount of paint Joe used in total -/
def total_paint_used (initial_paint : ℚ) (first_week_fraction : ℚ) (second_week_fraction : ℚ) : ℚ :=
  let first_week_usage := first_week_fraction * initial_paint
  let remaining_paint := initial_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  first_week_usage + second_week_usage

/-- Theorem stating that Joe used 264 gallons of paint -/
theorem joe_paint_usage :
  total_paint_used 360 (2/3) (1/5) = 264 := by
  sorry

end joe_paint_usage_l3154_315469


namespace max_sides_of_special_polygon_existence_of_five_sided_polygon_l3154_315478

-- Define a convex polygon
def ConvexPolygon (n : ℕ) := Unit

-- Define a property that a polygon has at least one side of length 1
def HasSideOfLengthOne (p : ConvexPolygon n) : Prop := sorry

-- Define a property that all diagonals of a polygon have integer lengths
def AllDiagonalsInteger (p : ConvexPolygon n) : Prop := sorry

-- State the theorem
theorem max_sides_of_special_polygon :
  ∀ n : ℕ, n > 5 →
  ¬∃ (p : ConvexPolygon n), HasSideOfLengthOne p ∧ AllDiagonalsInteger p :=
sorry

theorem existence_of_five_sided_polygon :
  ∃ (p : ConvexPolygon 5), HasSideOfLengthOne p ∧ AllDiagonalsInteger p :=
sorry

end max_sides_of_special_polygon_existence_of_five_sided_polygon_l3154_315478


namespace possible_values_of_a_l3154_315422

theorem possible_values_of_a (x y a : ℝ) 
  (h1 : x + y = a) 
  (h2 : x^3 + y^3 = a) 
  (h3 : x^5 + y^5 = a) : 
  a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2 := by
  sorry

end possible_values_of_a_l3154_315422


namespace pens_before_discount_is_75_l3154_315465

/-- The number of pens that can be bought before the discount -/
def pens_before_discount : ℕ := 75

/-- The discount rate -/
def discount_rate : ℚ := 1/4

/-- The number of additional pens that can be bought after the discount -/
def additional_pens : ℕ := 25

theorem pens_before_discount_is_75 :
  pens_before_discount = 75 ∧
  discount_rate = 1/4 ∧
  additional_pens = 25 ∧
  (pens_before_discount : ℚ) = (pens_before_discount + additional_pens) * (1 - discount_rate) :=
by sorry

end pens_before_discount_is_75_l3154_315465


namespace circumscribable_with_special_area_is_inscribable_l3154_315400

-- Define a quadrilateral
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  area : ℝ

-- Define the properties of being circumscribable and inscribable
def is_circumscribable (q : Quadrilateral) : Prop := sorry
def is_inscribable (q : Quadrilateral) : Prop := sorry

-- State the theorem
theorem circumscribable_with_special_area_is_inscribable (q : Quadrilateral) :
  is_circumscribable q →
  q.area = Real.sqrt (q.a * q.b * q.c * q.d) →
  is_inscribable q := by sorry

end circumscribable_with_special_area_is_inscribable_l3154_315400


namespace min_value_problem1_min_value_problem2_l3154_315458

theorem min_value_problem1 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + y = 1) :
  2*x + 1/(3*y) ≥ (13 + 4*Real.sqrt 3) / 3 :=
sorry

theorem min_value_problem2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 1) :
  1/(2*x) + x/(y+1) ≥ 5/4 :=
sorry

end min_value_problem1_min_value_problem2_l3154_315458


namespace ratio_nine_to_five_percent_l3154_315480

/-- The ratio 9 : 5 expressed as a percentage -/
def ratio_to_percent : ℚ := 9 / 5 * 100

/-- Theorem: The ratio 9 : 5 expressed as a percentage is equal to 180% -/
theorem ratio_nine_to_five_percent : ratio_to_percent = 180 := by
  sorry

end ratio_nine_to_five_percent_l3154_315480


namespace last_two_digits_of_7_pow_2017_l3154_315491

-- Define the pattern of last two digits
def lastTwoDigitsPattern : Fin 4 → Nat
  | 0 => 49
  | 1 => 43
  | 2 => 01
  | 3 => 07

-- Define the function to get the last two digits of 7^n
def lastTwoDigits (n : Nat) : Nat :=
  lastTwoDigitsPattern ((n - 2) % 4)

-- Theorem statement
theorem last_two_digits_of_7_pow_2017 :
  lastTwoDigits 2017 = 07 := by
  sorry

end last_two_digits_of_7_pow_2017_l3154_315491


namespace initial_fund_is_740_l3154_315460

/-- Represents the company fund problem --/
structure CompanyFund where
  intended_bonus : ℕ
  actual_bonus : ℕ
  remaining_amount : ℕ
  fixed_expense : ℕ
  shortage : ℕ

/-- Calculates the initial fund amount before bonuses and expenses --/
def initial_fund_amount (cf : CompanyFund) : ℕ :=
  sorry

/-- Theorem stating the initial fund amount is 740 given the problem conditions --/
theorem initial_fund_is_740 (cf : CompanyFund) 
  (h1 : cf.intended_bonus = 60)
  (h2 : cf.actual_bonus = 50)
  (h3 : cf.remaining_amount = 110)
  (h4 : cf.fixed_expense = 30)
  (h5 : cf.shortage = 10) :
  initial_fund_amount cf = 740 :=
sorry

end initial_fund_is_740_l3154_315460


namespace probability_of_valid_roll_l3154_315474

/-- A standard six-sided die -/
def Die : Type := Fin 6

/-- The set of possible outcomes when rolling two dice -/
def TwoDiceRoll : Type := Die × Die

/-- The set of valid two-digit numbers between 40 and 50 (inclusive) -/
def ValidNumbers : Set ℕ := {n : ℕ | 40 ≤ n ∧ n ≤ 50}

/-- Function to convert a dice roll to a two-digit number -/
def rollToNumber (roll : TwoDiceRoll) : ℕ :=
  10 * (roll.1.val + 1) + (roll.2.val + 1)

/-- The set of favorable outcomes -/
def FavorableOutcomes : Set TwoDiceRoll :=
  {roll : TwoDiceRoll | rollToNumber roll ∈ ValidNumbers}

/-- Total number of possible outcomes when rolling two dice -/
def TotalOutcomes : ℕ := 36

/-- Number of favorable outcomes -/
def FavorableOutcomesCount : ℕ := 12

/-- Probability of rolling a number between 40 and 50 (inclusive) -/
theorem probability_of_valid_roll :
  (FavorableOutcomesCount : ℚ) / TotalOutcomes = 1 / 3 := by
  sorry

end probability_of_valid_roll_l3154_315474


namespace quadratic_inequality_solution_l3154_315446

theorem quadratic_inequality_solution : 
  {x : ℝ | x^2 + 2*x ≤ -1} = {-1} := by sorry

end quadratic_inequality_solution_l3154_315446


namespace camdens_dogs_legs_l3154_315427

theorem camdens_dogs_legs : 
  ∀ (justin_dogs rico_dogs camden_dogs : ℕ) (legs_per_dog : ℕ),
  justin_dogs = 14 →
  rico_dogs = justin_dogs + 10 →
  camden_dogs = rico_dogs * 3 / 4 →
  legs_per_dog = 4 →
  camden_dogs * legs_per_dog = 72 :=
by sorry

end camdens_dogs_legs_l3154_315427


namespace remainder_4536_div_32_l3154_315453

theorem remainder_4536_div_32 : 4536 % 32 = 24 := by
  sorry

end remainder_4536_div_32_l3154_315453


namespace u_2008_eq_225_l3154_315430

/-- Defines the sequence u_n as described in the problem -/
def u : ℕ → ℕ := sorry

/-- The 2008th term of the sequence is 225 -/
theorem u_2008_eq_225 : u 2008 = 225 := by sorry

end u_2008_eq_225_l3154_315430


namespace substance_volume_weight_relation_l3154_315424

/-- Given a substance where volume is directly proportional to weight,
    prove that if 48 cubic inches weigh 112 ounces,
    then 63 ounces will have a volume of 27 cubic inches. -/
theorem substance_volume_weight_relation 
  (k : ℚ) -- Constant of proportionality
  (h1 : 48 = k * 112) -- 48 cubic inches weigh 112 ounces
  : k * 63 = 27 := by
  sorry

end substance_volume_weight_relation_l3154_315424


namespace negation_of_universal_proposition_l3154_315442

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 5*x = 4) ↔ (∃ x : ℝ, x^2 + 5*x ≠ 4) := by sorry

end negation_of_universal_proposition_l3154_315442


namespace expression_simplification_l3154_315415

theorem expression_simplification (x y : ℝ) 
  (hx : x = Real.tan (60 * π / 180)^2 + 1)
  (hy : y = Real.tan (45 * π / 180) - 2 * Real.cos (30 * π / 180)) :
  (x - (2*x*y - y^2) / x) / ((x^2 - y^2) / (x^2 + x*y)) = 3 + Real.sqrt 3 := by
  sorry

end expression_simplification_l3154_315415


namespace last_digit_of_one_over_three_to_fifteen_l3154_315403

theorem last_digit_of_one_over_three_to_fifteen (n : ℕ) : 
  n = 15 → (1 : ℚ) / 3^n % 10 = 7 := by
  sorry

end last_digit_of_one_over_three_to_fifteen_l3154_315403


namespace euler_function_gcd_l3154_315486

open Nat

theorem euler_function_gcd (m n : ℕ) (h : φ (5^m - 1) = 5^n - 1) : (m.gcd n) > 1 := by
  sorry

end euler_function_gcd_l3154_315486


namespace geometric_sum_n_terms_l3154_315421

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_n_terms (a r : ℚ) (n : ℕ) (h1 : a = 1/3) (h2 : r = 1/3) :
  geometric_sum a r n = 80/243 ↔ n = 5 := by
  sorry

end geometric_sum_n_terms_l3154_315421


namespace min_area_rectangle_l3154_315406

/-- A rectangle with even integer dimensions and perimeter 120 has a minimum area of 116 -/
theorem min_area_rectangle (l w : ℕ) : 
  Even l → Even w → 
  2 * (l + w) = 120 → 
  ∀ a : ℕ, (Even a.sqrt ∧ Even (60 - a.sqrt) ∧ a = a.sqrt * (60 - a.sqrt)) → 
  116 ≤ a := by
sorry

end min_area_rectangle_l3154_315406


namespace decagon_diagonal_intersection_probability_l3154_315484

/-- A regular decagon -/
structure RegularDecagon where
  -- Add necessary fields here

/-- The probability that two randomly chosen diagonals of a regular decagon intersect inside the decagon -/
def intersection_probability (d : RegularDecagon) : ℚ :=
  42 / 119

/-- Theorem stating that the probability of two randomly chosen diagonals 
    of a regular decagon intersecting inside the decagon is 42/119 -/
theorem decagon_diagonal_intersection_probability (d : RegularDecagon) :
  intersection_probability d = 42 / 119 := by
  sorry

#check decagon_diagonal_intersection_probability

end decagon_diagonal_intersection_probability_l3154_315484


namespace gcf_of_104_and_156_l3154_315451

theorem gcf_of_104_and_156 : Nat.gcd 104 156 = 52 := by
  sorry

end gcf_of_104_and_156_l3154_315451


namespace loot_box_solution_l3154_315434

/-- Represents the loot box problem -/
def LootBoxProblem (cost_per_box : ℝ) (avg_value_per_box : ℝ) (total_spent : ℝ) : Prop :=
  let num_boxes : ℝ := total_spent / cost_per_box
  let total_avg_value : ℝ := avg_value_per_box * num_boxes
  let total_lost : ℝ := total_spent - total_avg_value
  let avg_lost_per_box : ℝ := total_lost / num_boxes
  avg_lost_per_box = 1.5

/-- Theorem stating the solution to the loot box problem -/
theorem loot_box_solution :
  LootBoxProblem 5 3.5 40 := by
  sorry

#check loot_box_solution

end loot_box_solution_l3154_315434


namespace cyclic_inequality_l3154_315416

theorem cyclic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 / (a^2 + a*b + b^2)) + (b^3 / (b^2 + b*c + c^2)) + (c^3 / (c^2 + a*c + a^2)) ≥ (a + b + c) / 3 := by
  sorry

end cyclic_inequality_l3154_315416


namespace least_positive_integer_congruence_l3154_315488

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 3490) % 15 = 2801 % 15 ∧
  ∀ (y : ℕ), y > 0 ∧ (y + 3490) % 15 = 2801 % 15 → x ≤ y :=
by sorry

end least_positive_integer_congruence_l3154_315488


namespace polynomial_division_remainder_l3154_315450

theorem polynomial_division_remainder (t : ℚ) :
  (∀ x, (6 * x^2 - 7 * x + 8) = (5 * x^2 + t * x + 12) * (4 * x^2 - 9 * x + 12)) →
  t = -7/12 := by
sorry

end polynomial_division_remainder_l3154_315450


namespace halfway_point_between_fractions_l3154_315457

theorem halfway_point_between_fractions :
  (1 / 12 + 1 / 15) / 2 = 3 / 40 := by
  sorry

end halfway_point_between_fractions_l3154_315457


namespace two_sqrt_six_lt_five_l3154_315452

theorem two_sqrt_six_lt_five : 2 * Real.sqrt 6 < 5 := by
  sorry

end two_sqrt_six_lt_five_l3154_315452


namespace rachel_saturday_water_consumption_l3154_315437

def glassesToOunces (glasses : ℕ) : ℕ := glasses * 10

def waterConsumed (sun mon tue wed thu fri : ℕ) : ℕ :=
  glassesToOunces (sun + mon + tue + wed + thu + fri)

theorem rachel_saturday_water_consumption
  (h1 : waterConsumed 2 4 3 3 3 3 + glassesToOunces x = 220) :
  x = 4 := by
  sorry

end rachel_saturday_water_consumption_l3154_315437


namespace prob_one_sunny_day_l3154_315499

/-- The probability of exactly one sunny day in a three-day festival --/
theorem prob_one_sunny_day (p_sunny : ℝ) (p_not_sunny : ℝ) :
  p_sunny = 0.1 →
  p_not_sunny = 0.9 →
  3 * (p_sunny * p_not_sunny * p_not_sunny) = 0.243 :=
by sorry

end prob_one_sunny_day_l3154_315499


namespace differential_equation_solution_l3154_315475

/-- The differential equation dy/dx = y^2 has a general solution y = a₀ / (1 - a₀x) -/
theorem differential_equation_solution (x : ℝ) (a₀ : ℝ) :
  let y : ℝ → ℝ := λ x => a₀ / (1 - a₀ * x)
  ∀ x, (deriv y) x = (y x)^2 :=
by sorry

end differential_equation_solution_l3154_315475


namespace coefficient_x_squared_in_expansion_l3154_315429

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (Finset.range 7).sum (fun k => (Nat.choose 6 k) * ((-1)^k) * ((Real.sqrt 2)^(6-k)) * (x^k)) = 
  60 * x^2 + (Finset.range 7).sum (fun k => if k ≠ 2 then (Nat.choose 6 k) * ((-1)^k) * ((Real.sqrt 2)^(6-k)) * (x^k) else 0) :=
by sorry

end coefficient_x_squared_in_expansion_l3154_315429


namespace gcd_840_1764_l3154_315412

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l3154_315412


namespace stability_comparison_A_more_stable_than_B_l3154_315423

-- Define a structure for a student's test scores
structure StudentScores where
  average : ℝ
  variance : ℝ

-- Define the stability comparison function
def more_stable (a b : StudentScores) : Prop :=
  a.average = b.average ∧ a.variance < b.variance

-- Theorem statement
theorem stability_comparison (a b : StudentScores) 
  (h_avg : a.average = b.average) 
  (h_var : a.variance < b.variance) : 
  more_stable a b := by
  sorry

-- Define students A and B
def student_A : StudentScores := { average := 88, variance := 0.61 }
def student_B : StudentScores := { average := 88, variance := 0.72 }

-- Theorem application to students A and B
theorem A_more_stable_than_B : more_stable student_A student_B := by
  sorry

end stability_comparison_A_more_stable_than_B_l3154_315423


namespace roxy_garden_plants_l3154_315467

/-- Calculates the total number of plants in Roxy's garden after buying and giving away plants -/
def total_plants_remaining (initial_flowering : ℕ) (bought_flowering : ℕ) (bought_fruiting : ℕ) 
  (given_away_flowering : ℕ) (given_away_fruiting : ℕ) : ℕ :=
  let initial_fruiting := 2 * initial_flowering
  let final_flowering := initial_flowering + bought_flowering - given_away_flowering
  let final_fruiting := initial_fruiting + bought_fruiting - given_away_fruiting
  final_flowering + final_fruiting

/-- Theorem stating that the total number of plants remaining in Roxy's garden is 21 -/
theorem roxy_garden_plants : 
  total_plants_remaining 7 3 2 1 4 = 21 := by
  sorry

end roxy_garden_plants_l3154_315467


namespace power_sum_equality_l3154_315414

theorem power_sum_equality (a b c d : ℝ) 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  (a^5 + b^5 = c^5 + d^5) ∧ 
  (∃ a b c d : ℝ, (a + b = c + d) ∧ (a^3 + b^3 = c^3 + d^3) ∧ (a^4 + b^4 ≠ c^4 + d^4)) :=
by sorry

end power_sum_equality_l3154_315414


namespace regular_polygon_sides_l3154_315409

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (((n : ℝ) - 2) * 180 = n * 160) → n = 18 := by
  sorry

end regular_polygon_sides_l3154_315409


namespace quadratic_solution_unique_positive_l3154_315418

theorem quadratic_solution_unique_positive (x : ℝ) :
  x > 0 ∧ 3 * x^2 + 8 * x - 35 = 0 ↔ x = 7/3 := by
  sorry

end quadratic_solution_unique_positive_l3154_315418


namespace race_time_difference_l3154_315497

/-- Race parameters and result -/
theorem race_time_difference
  (malcolm_speed : ℝ) -- Malcolm's speed in minutes per mile
  (joshua_speed : ℝ)  -- Joshua's speed in minutes per mile
  (race_distance : ℝ) -- Race distance in miles
  (h1 : malcolm_speed = 7)
  (h2 : joshua_speed = 8)
  (h3 : race_distance = 12) :
  joshua_speed * race_distance - malcolm_speed * race_distance = 12 := by
  sorry

end race_time_difference_l3154_315497


namespace sum_of_evens_l3154_315436

theorem sum_of_evens (n : ℕ) (sum_first_n : ℕ) (first_term : ℕ) (last_term : ℕ) : 
  n = 50 → 
  sum_first_n = 2550 → 
  first_term = 102 → 
  last_term = 200 → 
  (n : ℕ) * (first_term + last_term) / 2 = 7550 := by
  sorry

end sum_of_evens_l3154_315436


namespace stock_price_example_l3154_315492

/-- Given a stock with income, dividend rate, and investment amount, calculate its price. -/
def stock_price (income : ℚ) (dividend_rate : ℚ) (investment : ℚ) : ℚ :=
  let face_value := (income * 100) / dividend_rate
  (investment / face_value) * 100

/-- Theorem: The price of a stock with income Rs. 650, 10% dividend rate, and Rs. 6240 investment is Rs. 96. -/
theorem stock_price_example : stock_price 650 10 6240 = 96 := by
  sorry

end stock_price_example_l3154_315492


namespace line_parallel_to_plane_theorem_l3154_315404

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (inside_plane : Line → Plane → Prop)
variable (skew_lines : Line → Line → Prop)

-- State the theorem
theorem line_parallel_to_plane_theorem 
  (a b : Line) (α : Plane) 
  (h1 : parallel_line_plane a α) 
  (h2 : inside_plane b α) :
  parallel_lines a b ∨ skew_lines a b :=
sorry

end line_parallel_to_plane_theorem_l3154_315404


namespace hoseok_persimmons_l3154_315494

theorem hoseok_persimmons (jungkook_persimmons hoseok_persimmons : ℕ) : 
  jungkook_persimmons = 25 → 
  3 * hoseok_persimmons = jungkook_persimmons - 4 →
  hoseok_persimmons = 7 := by
sorry

end hoseok_persimmons_l3154_315494


namespace smiths_bakery_pies_l3154_315477

/-- The number of pies sold by Mcgee's Bakery -/
def mcgees_pies : ℕ := 16

/-- The number of pies sold by Smith's Bakery -/
def smiths_pies : ℕ := 4 * mcgees_pies + 6

/-- Theorem stating that Smith's Bakery sold 70 pies -/
theorem smiths_bakery_pies : smiths_pies = 70 := by
  sorry

end smiths_bakery_pies_l3154_315477


namespace hyperbola_asymptote_slope_l3154_315493

/-- The value of m for a hyperbola with equation (y^2/16) - (x^2/9) = 1 and asymptotes y = ±mx -/
theorem hyperbola_asymptote_slope (m : ℝ) : m > 0 →
  (∀ x y : ℝ, y^2/16 - x^2/9 = 1 → (y = m*x ∨ y = -m*x)) → m = 4/3 := by
  sorry

end hyperbola_asymptote_slope_l3154_315493


namespace linda_notebooks_count_l3154_315443

/-- The number of notebooks Linda bought -/
def num_notebooks : ℕ := 3

/-- The cost of each notebook in dollars -/
def notebook_cost : ℚ := 6/5

/-- The cost of a box of pencils in dollars -/
def pencil_box_cost : ℚ := 3/2

/-- The cost of a box of pens in dollars -/
def pen_box_cost : ℚ := 17/10

/-- The total amount spent in dollars -/
def total_spent : ℚ := 68/10

theorem linda_notebooks_count :
  (num_notebooks : ℚ) * notebook_cost + pencil_box_cost + pen_box_cost = total_spent :=
sorry

end linda_notebooks_count_l3154_315443


namespace three_cones_problem_l3154_315432

/-- A cone with vertex A -/
structure Cone (A : Point) where
  vertex_angle : ℝ

/-- A plane passing through a point -/
structure Plane (A : Point)

/-- Three cones touching each other externally -/
def touching_cones (A : Point) (c1 c2 c3 : Cone A) : Prop :=
  sorry

/-- Two cones are identical -/
def identical_cones (c1 c2 : Cone A) : Prop :=
  c1.vertex_angle = c2.vertex_angle

/-- A cone touches a plane -/
def cone_touches_plane (c : Cone A) (p : Plane A) : Prop :=
  sorry

/-- A cone lies on one side of a plane -/
def cone_on_one_side (c : Cone A) (p : Plane A) : Prop :=
  sorry

theorem three_cones_problem (A : Point) (c1 c2 c3 : Cone A) (p : Plane A) :
  touching_cones A c1 c2 c3 →
  identical_cones c1 c2 →
  c3.vertex_angle = π / 2 →
  cone_touches_plane c1 p →
  cone_touches_plane c2 p →
  cone_touches_plane c3 p →
  cone_on_one_side c1 p →
  cone_on_one_side c2 p →
  cone_on_one_side c3 p →
  c1.vertex_angle = 2 * Real.arctan (4 / 5) :=
sorry

end three_cones_problem_l3154_315432


namespace compound_interest_rate_l3154_315405

/-- Given a principal amount, time period, and final amount, 
    calculate the annual interest rate for compound interest. -/
theorem compound_interest_rate 
  (principal : ℝ) 
  (time : ℝ) 
  (final_amount : ℝ) 
  (h1 : principal = 8000) 
  (h2 : time = 2) 
  (h3 : final_amount = 8820) : 
  ∃ (rate : ℝ), 
    final_amount = principal * (1 + rate) ^ time ∧ 
    rate = 0.05 := by
  sorry

end compound_interest_rate_l3154_315405


namespace find_a_l3154_315439

-- Define the points A and B
def A (a : ℝ) : ℝ × ℝ := (-1, a)
def B (a : ℝ) : ℝ × ℝ := (a, 8)

-- Define the slope of the line 2x - y + 1 = 0
def slope_given_line : ℝ := 2

-- Define the theorem
theorem find_a : ∃ a : ℝ, 
  (B a).2 - (A a).2 = slope_given_line * ((B a).1 - (A a).1) :=
sorry

-- Note: (p.1) and (p.2) represent the x and y coordinates of a point p respectively

end find_a_l3154_315439


namespace perfect_square_condition_l3154_315411

theorem perfect_square_condition (a b c d : ℕ+) : 
  (↑a + Real.rpow 2 (1/3 : ℝ) * ↑b + Real.rpow 2 (2/3 : ℝ) * ↑c)^2 = ↑d → 
  ∃ (n : ℕ), d = n^2 := by
  sorry

end perfect_square_condition_l3154_315411


namespace element_type_determined_by_protons_nuclide_type_determined_by_protons_and_neutrons_chemical_properties_determined_by_outermost_electrons_highest_valence_equals_main_group_number_l3154_315410

-- Define basic types
structure Element where
  protons : ℕ

structure Nuclide where
  protons : ℕ
  neutrons : ℕ

structure MainGroupElement where
  protons : ℕ
  outermostElectrons : ℕ

-- Define properties
def elementType (e : Element) : ℕ := e.protons

def nuclideType (n : Nuclide) : ℕ × ℕ := (n.protons, n.neutrons)

def mainChemicalProperties (e : MainGroupElement) : ℕ := e.outermostElectrons

def highestPositiveValence (e : MainGroupElement) : ℕ := e.outermostElectrons

-- Theorem statements
theorem element_type_determined_by_protons (e1 e2 : Element) :
  elementType e1 = elementType e2 ↔ e1.protons = e2.protons :=
sorry

theorem nuclide_type_determined_by_protons_and_neutrons (n1 n2 : Nuclide) :
  nuclideType n1 = nuclideType n2 ↔ n1.protons = n2.protons ∧ n1.neutrons = n2.neutrons :=
sorry

theorem chemical_properties_determined_by_outermost_electrons (e : MainGroupElement) :
  mainChemicalProperties e = e.outermostElectrons :=
sorry

theorem highest_valence_equals_main_group_number (e : MainGroupElement) :
  highestPositiveValence e = e.outermostElectrons :=
sorry

end element_type_determined_by_protons_nuclide_type_determined_by_protons_and_neutrons_chemical_properties_determined_by_outermost_electrons_highest_valence_equals_main_group_number_l3154_315410


namespace supremum_inequality_l3154_315490

theorem supremum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  - (1 / (2 * a)) - (2 / b) ≤ - (9 / 2) ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ - (1 / (2 * a₀)) - (2 / b₀) = - (9 / 2) :=
sorry

end supremum_inequality_l3154_315490


namespace three_number_product_l3154_315419

theorem three_number_product (a b c : ℝ) 
  (sum_eq : a + b + c = 30)
  (first_eq : a = 2 * (b + c))
  (second_eq : b = 5 * c) :
  a * b * c = 2500 / 9 := by
sorry

end three_number_product_l3154_315419


namespace range_of_a_l3154_315401

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}
def B : Set ℝ := {x : ℝ | 3 * x - 1 < x + 5}
def C (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}

-- Define the complement of A with respect to ℝ
def complementA : Set ℝ := {x : ℝ | x ≤ 1 ∨ x ≥ 4}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (complementA ∩ C a = C a) → a ≤ 1 := by
  sorry

end range_of_a_l3154_315401


namespace technician_count_l3154_315482

theorem technician_count (total_workers : ℕ) (avg_salary : ℝ) (avg_tech_salary : ℝ) (avg_rest_salary : ℝ) :
  total_workers = 12 ∧ 
  avg_salary = 9000 ∧ 
  avg_tech_salary = 12000 ∧ 
  avg_rest_salary = 6000 →
  ∃ (tech_count : ℕ),
    tech_count = 6 ∧
    tech_count + (total_workers - tech_count) = total_workers ∧
    (avg_tech_salary * tech_count + avg_rest_salary * (total_workers - tech_count)) / total_workers = avg_salary :=
by
  sorry

end technician_count_l3154_315482


namespace victor_sugar_usage_l3154_315472

theorem victor_sugar_usage (brown_sugar : ℝ) (difference : ℝ) (white_sugar : ℝ)
  (h1 : brown_sugar = 0.62)
  (h2 : brown_sugar = white_sugar + difference)
  (h3 : difference = 0.38) :
  white_sugar = 0.24 := by
  sorry

end victor_sugar_usage_l3154_315472


namespace min_students_with_all_characteristics_l3154_315466

theorem min_students_with_all_characteristics
  (total : ℕ)
  (brown_eyes : ℕ)
  (lunch_boxes : ℕ)
  (glasses : ℕ)
  (h_total : total = 35)
  (h_brown_eyes : brown_eyes = 15)
  (h_lunch_boxes : lunch_boxes = 25)
  (h_glasses : glasses = 10) :
  ∃ (n : ℕ), n ≥ 5 ∧
    n = (brown_eyes + lunch_boxes + glasses - total).max 0 :=
by sorry

end min_students_with_all_characteristics_l3154_315466


namespace one_fifths_in_one_tenth_l3154_315470

theorem one_fifths_in_one_tenth : (1 / 10 : ℚ) / (1 / 5 : ℚ) = 1 / 2 := by
  sorry

end one_fifths_in_one_tenth_l3154_315470


namespace special_sequence_sum_property_l3154_315476

/-- A sequence of pairwise distinct nonnegative integers satisfying the given conditions -/
def SpecialSequence (b : ℕ → ℕ) : Prop :=
  (∀ i j, i ≠ j → b i ≠ b j) ∧ 
  (b 0 = 0) ∧ 
  (∀ n > 0, b n < 2 * n)

/-- The main theorem -/
theorem special_sequence_sum_property (b : ℕ → ℕ) (h : SpecialSequence b) :
  ∀ m : ℕ, ∃ k ℓ : ℕ, b k + b ℓ = m := by
  sorry

end special_sequence_sum_property_l3154_315476


namespace odd_implies_derivative_even_exists_increasing_not_increasing_derivative_l3154_315426

-- Define a real-valued function on R
variable (f : ℝ → ℝ)

-- Define the property of being an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the property of being strictly increasing
def StrictlyIncreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

-- Proposition 1: If f is odd, then f' is even
theorem odd_implies_derivative_even (hf : IsOdd f) : IsEven (deriv f) := by sorry

-- Proposition 2: There exists a strictly increasing function whose derivative is not strictly increasing
theorem exists_increasing_not_increasing_derivative : 
  ∃ f : ℝ → ℝ, StrictlyIncreasing f ∧ ¬StrictlyIncreasing (deriv f) := by sorry

end odd_implies_derivative_even_exists_increasing_not_increasing_derivative_l3154_315426


namespace sin_negative_nineteen_sixths_pi_l3154_315433

theorem sin_negative_nineteen_sixths_pi : 
  Real.sin (-19/6 * Real.pi) = 1/2 := by sorry

end sin_negative_nineteen_sixths_pi_l3154_315433


namespace diophantine_equation_solutions_l3154_315441

theorem diophantine_equation_solutions :
  (∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    4 * p.1 + 7 * p.2 = 1003 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 1004) (Finset.range 1004))).card ∧ n = 36) :=
by sorry

end diophantine_equation_solutions_l3154_315441


namespace jerry_shelf_difference_l3154_315473

/-- Calculates the difference between books and action figures on Jerry's shelf -/
def shelf_difference (initial_figures : ℕ) (initial_books : ℕ) (added_figures : ℕ) : ℕ :=
  initial_books - (initial_figures + added_figures)

/-- Proves that the difference between books and action figures on Jerry's shelf is 4 -/
theorem jerry_shelf_difference :
  shelf_difference 2 10 4 = 4 := by
  sorry

end jerry_shelf_difference_l3154_315473


namespace goldbach_for_given_numbers_l3154_315463

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def goldbach_for_number (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p + q

theorem goldbach_for_given_numbers :
  goldbach_for_number 102 ∧
  goldbach_for_number 144 ∧
  goldbach_for_number 178 ∧
  goldbach_for_number 200 :=
sorry

end goldbach_for_given_numbers_l3154_315463


namespace dog_paws_on_ground_l3154_315479

theorem dog_paws_on_ground (total_dogs : ℕ) (dogs_on_back_legs : ℕ) (dogs_on_all_legs : ℕ) : 
  total_dogs = 12 →
  dogs_on_back_legs = total_dogs / 2 →
  dogs_on_all_legs = total_dogs / 2 →
  dogs_on_back_legs * 2 + dogs_on_all_legs * 4 = 36 := by
  sorry

end dog_paws_on_ground_l3154_315479
