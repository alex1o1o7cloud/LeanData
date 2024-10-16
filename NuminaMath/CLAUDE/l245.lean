import Mathlib

namespace NUMINAMATH_CALUDE_calculate_expression_l245_24570

theorem calculate_expression : (8^5 / 8^2) * 3^6 = 373248 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l245_24570


namespace NUMINAMATH_CALUDE_initial_milk_percentage_l245_24576

/-- Given a mixture of milk and water, prove the initial percentage of milk. -/
theorem initial_milk_percentage
  (total_initial_volume : ℝ)
  (added_water : ℝ)
  (final_milk_percentage : ℝ)
  (h1 : total_initial_volume = 60)
  (h2 : added_water = 40.8)
  (h3 : final_milk_percentage = 50) :
  (total_initial_volume * 84 / 100) / total_initial_volume = 
  (total_initial_volume * final_milk_percentage / 100) / (total_initial_volume + added_water) :=
by sorry

end NUMINAMATH_CALUDE_initial_milk_percentage_l245_24576


namespace NUMINAMATH_CALUDE_pyramid_volume_l245_24592

theorem pyramid_volume (total_surface_area : ℝ) (triangular_face_ratio : ℝ) :
  total_surface_area = 600 →
  triangular_face_ratio = 2 →
  ∃ (volume : ℝ),
    volume = (1/3) * (total_surface_area / (4 * triangular_face_ratio + 1)) * 
             (Real.sqrt ((4 * triangular_face_ratio + 1) * 
             (4 * triangular_face_ratio - 1) / (triangular_face_ratio^2))) *
             Real.sqrt (total_surface_area / (4 * triangular_face_ratio + 1)) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l245_24592


namespace NUMINAMATH_CALUDE_min_value_of_d_l245_24562

/-- The function d(x, y) to be minimized -/
def d (x y : ℝ) : ℝ := x^2 + y^2 - 2*x - 4*y + 6

/-- Theorem stating that the minimum value of d(x, y) is 1 -/
theorem min_value_of_d :
  ∃ (min : ℝ), min = 1 ∧ ∀ (x y : ℝ), d x y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_d_l245_24562


namespace NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l245_24526

-- Part 1
theorem simplify_expression (x y : ℝ) : x - (2*x - y) + (3*x - 2*y) = 2*x - y := by
  sorry

-- Part 2
theorem evaluate_expression : 
  let x : ℚ := -2/3
  let y : ℚ := 3/2
  2*x*y + (-3*x^3 + 5*x*y + 2) - 3*(2*x*y - x^3 + 1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l245_24526


namespace NUMINAMATH_CALUDE_student_minimum_earnings_l245_24585

/-- Represents the student's work situation -/
structure WorkSituation where
  library_rate : ℝ
  construction_rate : ℝ
  total_hours : ℝ
  library_hours : ℝ

/-- Calculates the minimum weekly earnings for the student -/
def minimum_weekly_earnings (w : WorkSituation) : ℝ :=
  w.library_rate * w.library_hours + 
  w.construction_rate * (w.total_hours - w.library_hours)

/-- Theorem stating the minimum weekly earnings for the given work situation -/
theorem student_minimum_earnings : 
  let w : WorkSituation := {
    library_rate := 8,
    construction_rate := 15,
    total_hours := 25,
    library_hours := 10
  }
  minimum_weekly_earnings w = 305 := by sorry

end NUMINAMATH_CALUDE_student_minimum_earnings_l245_24585


namespace NUMINAMATH_CALUDE_kickball_players_l245_24563

/-- The number of students who played kickball on Wednesday -/
def wednesday_players : ℕ := sorry

/-- The number of students who played kickball on Thursday -/
def thursday_players : ℕ := sorry

/-- The total number of students who played kickball on both days -/
def total_players : ℕ := 65

theorem kickball_players :
  wednesday_players = 37 ∧
  thursday_players = wednesday_players - 9 ∧
  wednesday_players + thursday_players = total_players :=
by sorry

end NUMINAMATH_CALUDE_kickball_players_l245_24563


namespace NUMINAMATH_CALUDE_man_speed_calculation_man_speed_approximately_5_004_l245_24501

/-- Calculates the speed of a man walking opposite to a train, given the train's length, speed, and time to cross the man. -/
theorem man_speed_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / crossing_time
  let man_speed_ms := relative_speed - train_speed_ms
  let man_speed_kmh := man_speed_ms * 3600 / 1000
  man_speed_kmh

/-- The speed of the man is approximately 5.004 km/h -/
theorem man_speed_approximately_5_004 :
  ∃ ε > 0, |man_speed_calculation 200 114.99 6 - 5.004| < ε :=
sorry

end NUMINAMATH_CALUDE_man_speed_calculation_man_speed_approximately_5_004_l245_24501


namespace NUMINAMATH_CALUDE_math_books_count_l245_24528

theorem math_books_count (total_books : ℕ) (math_cost history_cost total_price : ℕ) :
  total_books = 90 →
  math_cost = 4 →
  history_cost = 5 →
  total_price = 396 →
  ∃ (math_books : ℕ), 
    math_books * math_cost + (total_books - math_books) * history_cost = total_price ∧ 
    math_books = 54 := by
  sorry

end NUMINAMATH_CALUDE_math_books_count_l245_24528


namespace NUMINAMATH_CALUDE_brian_video_time_l245_24579

/-- The duration of Brian's animal video watching session -/
def total_video_time (cat_video_duration : ℕ) : ℕ :=
  let dog_video_duration := 2 * cat_video_duration
  let first_two_videos_duration := cat_video_duration + dog_video_duration
  let gorilla_video_duration := 2 * first_two_videos_duration
  cat_video_duration + dog_video_duration + gorilla_video_duration

/-- Theorem stating that Brian spends 36 minutes watching animal videos -/
theorem brian_video_time : total_video_time 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_brian_video_time_l245_24579


namespace NUMINAMATH_CALUDE_R_duration_approx_l245_24590

/-- Represents the investment and profit information for three partners -/
structure PartnershipData where
  inv_ratio_P : ℚ
  inv_ratio_Q : ℚ
  inv_ratio_R : ℚ
  profit_ratio_P : ℚ
  profit_ratio_Q : ℚ
  profit_ratio_R : ℚ
  duration_P : ℚ
  duration_Q : ℚ

/-- Calculates the investment duration for partner R given the partnership data -/
def calculate_R_duration (data : PartnershipData) : ℚ :=
  (data.profit_ratio_R * data.inv_ratio_Q * data.duration_Q) /
  (data.profit_ratio_Q * data.inv_ratio_R)

/-- Theorem stating that R's investment duration is approximately 5.185 months -/
theorem R_duration_approx (data : PartnershipData)
  (h1 : data.inv_ratio_P = 7)
  (h2 : data.inv_ratio_Q = 5)
  (h3 : data.inv_ratio_R = 3)
  (h4 : data.profit_ratio_P = 7)
  (h5 : data.profit_ratio_Q = 9)
  (h6 : data.profit_ratio_R = 4)
  (h7 : data.duration_P = 5)
  (h8 : data.duration_Q = 7) :
  abs (calculate_R_duration data - 5.185) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_R_duration_approx_l245_24590


namespace NUMINAMATH_CALUDE_ages_solution_l245_24566

/-- Represents the ages of three individuals -/
structure Ages where
  shekhar : ℚ
  shobha : ℚ
  kapil : ℚ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  -- The ratio of ages is 4:3:2
  ages.shekhar / ages.shobha = 4 / 3 ∧
  ages.shekhar / ages.kapil = 2 ∧
  -- In 10 years, Kapil's age will equal Shekhar's present age
  ages.kapil + 10 = ages.shekhar ∧
  -- Shekhar's age will be 30 in 8 years
  ages.shekhar + 8 = 30

/-- The theorem to prove -/
theorem ages_solution :
  ∃ (ages : Ages), satisfies_conditions ages ∧
    ages.shekhar = 22 ∧ ages.shobha = 33/2 ∧ ages.kapil = 10 := by
  sorry


end NUMINAMATH_CALUDE_ages_solution_l245_24566


namespace NUMINAMATH_CALUDE_complex_equation_real_part_l245_24503

-- Define complex number z as a + bi
def z (a b : ℝ) : ℂ := Complex.mk a b

-- State the theorem
theorem complex_equation_real_part 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : z a b ^ 3 + 2 * z a b ^ 2 * Complex.I - 2 * z a b * Complex.I - 8 = 1624 * Complex.I) : 
  a ^ 3 - 3 * a * b ^ 2 - 8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_real_part_l245_24503


namespace NUMINAMATH_CALUDE_min_distance_point_to_line_l245_24549

/-- The minimum distance between a point (1,0) and the line x - y + 5 = 0 is 3√2 -/
theorem min_distance_point_to_line : 
  let F : ℝ × ℝ := (1, 0)
  let line (x y : ℝ) : Prop := x - y + 5 = 0
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 ∧ 
    ∀ (P : ℝ × ℝ), line P.1 P.2 → Real.sqrt ((F.1 - P.1)^2 + (F.2 - P.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_point_to_line_l245_24549


namespace NUMINAMATH_CALUDE_transformation_D_not_always_valid_transformation_A_valid_transformation_B_valid_transformation_C_valid_l245_24583

-- Define the transformations
def transformation_A (x y : ℝ) : Prop := x = y → x + 3 = y + 3
def transformation_B (x y : ℝ) : Prop := -2 * x = -2 * y → x = y
def transformation_C (x y m : ℝ) : Prop := x / m = y / m → x = y
def transformation_D (x y m : ℝ) : Prop := x = y → x / m = y / m

-- Define a property that checks if a transformation satisfies equation properties
def satisfies_equation_properties (t : (ℝ → ℝ → Prop)) : Prop :=
  ∀ x y : ℝ, t x y ↔ x = y

-- Theorem stating that transformation D does not always satisfy equation properties
theorem transformation_D_not_always_valid :
  ¬(∀ m : ℝ, satisfies_equation_properties (transformation_D · · m)) :=
sorry

-- Theorems stating that transformations A, B, and C satisfy equation properties
theorem transformation_A_valid :
  satisfies_equation_properties transformation_A :=
sorry

theorem transformation_B_valid :
  satisfies_equation_properties transformation_B :=
sorry

theorem transformation_C_valid :
  ∀ m : ℝ, m ≠ 0 → satisfies_equation_properties (transformation_C · · m) :=
sorry

end NUMINAMATH_CALUDE_transformation_D_not_always_valid_transformation_A_valid_transformation_B_valid_transformation_C_valid_l245_24583


namespace NUMINAMATH_CALUDE_amusement_park_visitors_l245_24518

/-- Represents the amusement park ticket sales problem -/
theorem amusement_park_visitors 
  (ticket_price : ℕ) 
  (saturday_visitors : ℕ) 
  (sunday_visitors : ℕ) 
  (total_revenue : ℕ) 
  (h1 : ticket_price = 3)
  (h2 : saturday_visitors = 200)
  (h3 : sunday_visitors = 300)
  (h4 : total_revenue = 3000) :
  ∃ (daily_visitors : ℕ), 
    daily_visitors * 5 * ticket_price + (saturday_visitors + sunday_visitors) * ticket_price = total_revenue ∧ 
    daily_visitors = 100 := by
  sorry


end NUMINAMATH_CALUDE_amusement_park_visitors_l245_24518


namespace NUMINAMATH_CALUDE_omega_even_implies_periodic_l245_24556

/-- Definition of an Ω function -/
def is_omega_function (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f x = T * f (x + T)

/-- Definition of an even function -/
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- Definition of a periodic function -/
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- Theorem: If f is an Ω function and even, then it is periodic -/
theorem omega_even_implies_periodic
  (f : ℝ → ℝ) (h_omega : is_omega_function f) (h_even : is_even_function f) :
  ∃ T : ℝ, T ≠ 0 ∧ is_periodic f (2 * T) :=
by sorry


end NUMINAMATH_CALUDE_omega_even_implies_periodic_l245_24556


namespace NUMINAMATH_CALUDE_ellipse_max_distance_sum_l245_24520

/-- Given an ellipse with equation x^2/4 + y^2/3 = 1 and foci F₁ and F₂,
    where a line l passing through F₁ intersects the ellipse at points A and B,
    the maximum value of |BF₂| + |AF₂| is 5. -/
theorem ellipse_max_distance_sum (F₁ F₂ A B : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  (∀ x y, x^2/4 + y^2/3 = 1 → (x, y) ∈ l → (x, y) = A ∨ (x, y) = B) →
  F₁ ∈ l →
  F₁.1 < F₂.1 →
  (∀ x y, x^2/4 + y^2/3 = 1 → dist (x, y) F₁ + dist (x, y) F₂ = 4) →
  dist B F₂ + dist A F₂ ≤ 5 :=
sorry


end NUMINAMATH_CALUDE_ellipse_max_distance_sum_l245_24520


namespace NUMINAMATH_CALUDE_cuboid_lateral_surface_area_l245_24564

/-- The lateral surface area of a cuboid with given dimensions -/
def lateralSurfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * height + width * height)

/-- Theorem: The lateral surface area of a cuboid with length 10 m, width 14 m, and height 18 m is 864 m² -/
theorem cuboid_lateral_surface_area :
  lateralSurfaceArea 10 14 18 = 864 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_lateral_surface_area_l245_24564


namespace NUMINAMATH_CALUDE_jane_sequins_count_l245_24567

/-- The number of rows of blue sequins -/
def blue_rows : ℕ := 6

/-- The number of blue sequins in each row -/
def blue_per_row : ℕ := 8

/-- The number of rows of purple sequins -/
def purple_rows : ℕ := 5

/-- The number of purple sequins in each row -/
def purple_per_row : ℕ := 12

/-- The number of rows of green sequins -/
def green_rows : ℕ := 9

/-- The number of green sequins in each row -/
def green_per_row : ℕ := 6

/-- The total number of sequins Jane adds to her costume -/
def total_sequins : ℕ := blue_rows * blue_per_row + purple_rows * purple_per_row + green_rows * green_per_row

theorem jane_sequins_count : total_sequins = 162 := by
  sorry

end NUMINAMATH_CALUDE_jane_sequins_count_l245_24567


namespace NUMINAMATH_CALUDE_first_saline_concentration_l245_24551

theorem first_saline_concentration 
  (desired_concentration : ℝ)
  (total_volume : ℝ)
  (first_volume : ℝ)
  (second_volume : ℝ)
  (second_concentration : ℝ)
  (h1 : desired_concentration = 3.24)
  (h2 : total_volume = 5)
  (h3 : first_volume = 3.6)
  (h4 : second_volume = 1.4)
  (h5 : second_concentration = 9)
  (h6 : total_volume = first_volume + second_volume)
  : ∃ (first_concentration : ℝ),
    first_concentration = 1 ∧
    desired_concentration * total_volume = 
      first_concentration * first_volume + second_concentration * second_volume :=
by sorry

end NUMINAMATH_CALUDE_first_saline_concentration_l245_24551


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l245_24597

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- The angles are complementary
  a / b = 3 / 2 →  -- The ratio of the angles is 3:2
  b = 36 :=  -- The smaller angle is 36°
by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l245_24597


namespace NUMINAMATH_CALUDE_percentage_fraction_difference_l245_24547

theorem percentage_fraction_difference : (75 / 100 * 40) - (4 / 5 * 25) = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_fraction_difference_l245_24547


namespace NUMINAMATH_CALUDE_competition_probabilities_l245_24560

/-- Represents the type of question in the competition -/
inductive QuestionType
| MultipleChoice
| TrueFalse

/-- Represents a question in the competition -/
structure Question where
  id : Nat
  type : QuestionType

/-- Represents the competition setup -/
structure Competition where
  questions : Finset Question
  numMultipleChoice : Nat
  numTrueFalse : Nat

/-- Represents a draw outcome for two participants -/
structure DrawOutcome where
  questionA : Question
  questionB : Question

/-- The probability of A drawing a multiple-choice question and B drawing a true/false question -/
def probAMultipleBTrue (c : Competition) : ℚ :=
  sorry

/-- The probability of at least one of A and B drawing a multiple-choice question -/
def probAtLeastOneMultiple (c : Competition) : ℚ :=
  sorry

/-- The main theorem stating the probabilities for the given competition setup -/
theorem competition_probabilities (c : Competition) 
  (h1 : c.questions.card = 4)
  (h2 : c.numMultipleChoice = 2)
  (h3 : c.numTrueFalse = 2) :
  probAMultipleBTrue c = 1/3 ∧ probAtLeastOneMultiple c = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_competition_probabilities_l245_24560


namespace NUMINAMATH_CALUDE_food_consumption_reduction_l245_24544

/-- Calculates the required reduction in food consumption per student to maintain the same total cost, given a decrease in the number of students and an increase in food price. -/
theorem food_consumption_reduction 
  (student_decrease_rate : ℝ) 
  (food_price_increase_rate : ℝ) 
  (ε : ℝ) -- tolerance for approximation
  (h1 : student_decrease_rate = 0.05)
  (h2 : food_price_increase_rate = 0.20)
  (h3 : ε > 0)
  : ∃ (reduction_rate : ℝ), 
    abs (reduction_rate - (1 - 1 / ((1 - student_decrease_rate) * (1 + food_price_increase_rate)))) < ε ∧ 
    abs (reduction_rate - 0.1228) < ε := by
  sorry

end NUMINAMATH_CALUDE_food_consumption_reduction_l245_24544


namespace NUMINAMATH_CALUDE_triangle_theorem_l245_24521

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_A : 0 < A
  pos_B : 0 < B
  pos_C : 0 < C
  sum_angles : A + B + C = Real.pi
  sine_law_ab : a / (Real.sin A) = b / (Real.sin B)
  sine_law_bc : b / (Real.sin B) = c / (Real.sin C)

/-- The main theorem -/
theorem triangle_theorem (t : Triangle) (h : 4 * t.b * Real.sin t.A = Real.sqrt 7 * t.a) :
  (Real.sin t.B = Real.sqrt 7 / 4) ∧
  (t.a < t.b ∧ t.b < t.c → t.c - t.b = t.b - t.a → Real.cos t.A - Real.cos t.C = Real.sqrt 7 / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l245_24521


namespace NUMINAMATH_CALUDE_exhibition_arrangements_l245_24532

def total_arrangements (n : ℕ) : ℕ := n.factorial

def adjacent_arrangements (n : ℕ) : ℕ := (n - 1).factorial * 2

theorem exhibition_arrangements :
  let n := 4
  let total := total_arrangements n
  let adjacent := adjacent_arrangements n
  total - adjacent = 12 := by sorry

end NUMINAMATH_CALUDE_exhibition_arrangements_l245_24532


namespace NUMINAMATH_CALUDE_number_of_divisors_36_l245_24552

/-- The number of positive divisors of 36 is 9. -/
theorem number_of_divisors_36 : Finset.card (Nat.divisors 36) = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_36_l245_24552


namespace NUMINAMATH_CALUDE_other_solution_of_quadratic_equation_l245_24542

theorem other_solution_of_quadratic_equation :
  let equation := fun (x : ℚ) => 72 * x^2 + 43 = 113 * x - 12
  equation (3/8) → ∃ x : ℚ, x ≠ 3/8 ∧ equation x ∧ x = 43/36 := by
  sorry

end NUMINAMATH_CALUDE_other_solution_of_quadratic_equation_l245_24542


namespace NUMINAMATH_CALUDE_cat_sale_theorem_l245_24508

/-- Represents the count of cats for each breed -/
structure CatCounts where
  siamese : Nat
  persian : Nat
  house : Nat
  maineCoon : Nat

/-- Represents the number of pairs sold for each breed -/
structure SoldPairs where
  siamese : Nat
  persian : Nat
  maineCoon : Nat

/-- Calculates the remaining cats after the sale -/
def remainingCats (initial : CatCounts) (sold : SoldPairs) : CatCounts :=
  { siamese := initial.siamese - sold.siamese,
    persian := initial.persian - sold.persian,
    house := initial.house,
    maineCoon := initial.maineCoon - sold.maineCoon }

theorem cat_sale_theorem (initial : CatCounts) (sold : SoldPairs) :
  initial.siamese = 25 →
  initial.persian = 18 →
  initial.house = 12 →
  initial.maineCoon = 10 →
  sold.siamese = 6 →
  sold.persian = 4 →
  sold.maineCoon = 3 →
  let remaining := remainingCats initial sold
  remaining.siamese = 19 ∧
  remaining.persian = 14 ∧
  remaining.house = 12 ∧
  remaining.maineCoon = 7 :=
by sorry

end NUMINAMATH_CALUDE_cat_sale_theorem_l245_24508


namespace NUMINAMATH_CALUDE_mikes_books_l245_24543

/-- Mike's book counting problem -/
theorem mikes_books (initial_books bought_books : ℕ) :
  initial_books = 35 →
  bought_books = 56 →
  initial_books + bought_books = 91 := by
  sorry

end NUMINAMATH_CALUDE_mikes_books_l245_24543


namespace NUMINAMATH_CALUDE_congruent_integers_count_l245_24586

theorem congruent_integers_count : 
  (Finset.filter (fun n => n > 0 ∧ n < 2000 ∧ n % 13 = 6) (Finset.range 2000)).card = 154 :=
by sorry

end NUMINAMATH_CALUDE_congruent_integers_count_l245_24586


namespace NUMINAMATH_CALUDE_hilary_jar_regular_toenails_l245_24588

/-- Represents the capacity and contents of a toenail jar. -/
structure ToenailJar where
  capacity : ℕ  -- Total capacity in terms of regular toenails
  bigToenailSize : ℕ  -- Size of a big toenail relative to a regular toenail
  bigToenailsInJar : ℕ  -- Number of big toenails already in the jar
  remainingSpace : ℕ  -- Remaining space in terms of regular toenails

/-- Calculates the number of regular toenails already in the jar. -/
def regularToenailsInJar (jar : ToenailJar) : ℕ :=
  jar.capacity - (jar.bigToenailsInJar * jar.bigToenailSize) - jar.remainingSpace

/-- Theorem stating the number of regular toenails already in the jar. -/
theorem hilary_jar_regular_toenails :
  let jar : ToenailJar := {
    capacity := 100,
    bigToenailSize := 2,
    bigToenailsInJar := 20,
    remainingSpace := 20
  }
  regularToenailsInJar jar = 40 := by
  sorry

end NUMINAMATH_CALUDE_hilary_jar_regular_toenails_l245_24588


namespace NUMINAMATH_CALUDE_tinks_are_falars_and_gymes_l245_24557

-- Define the types for our entities
variable (U : Type) -- Universe type
variable (Falar Gyme Halp Tink Isoy : Set U)

-- State the given conditions
variable (h1 : Falar ⊆ Gyme)
variable (h2 : Halp ⊆ Tink)
variable (h3 : Isoy ⊆ Falar)
variable (h4 : Tink ⊆ Isoy)

-- State the theorem to be proved
theorem tinks_are_falars_and_gymes : Tink ⊆ Falar ∧ Tink ⊆ Gyme := by
  sorry

end NUMINAMATH_CALUDE_tinks_are_falars_and_gymes_l245_24557


namespace NUMINAMATH_CALUDE_circle_line_distance_range_l245_24571

theorem circle_line_distance_range (b : ℝ) : 
  (∃! (p q : ℝ × ℝ), 
    p.1^2 + p.2^2 = 4 ∧ 
    q.1^2 + q.2^2 = 4 ∧ 
    (p ≠ q) ∧
    (|p.2 - p.1 - b| / Real.sqrt 2 = 1) ∧
    (|q.2 - q.1 - b| / Real.sqrt 2 = 1)) →
  (b < -Real.sqrt 2 ∧ b > -3 * Real.sqrt 2) ∨ 
  (b > Real.sqrt 2 ∧ b < 3 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_circle_line_distance_range_l245_24571


namespace NUMINAMATH_CALUDE_brazilian_coffee_price_l245_24595

/-- Proves that the price of Brazilian coffee is $3.75 per pound given the conditions of the coffee mix problem. -/
theorem brazilian_coffee_price
  (total_mix : ℝ)
  (columbian_price : ℝ)
  (final_mix_price : ℝ)
  (columbian_amount : ℝ)
  (h_total_mix : total_mix = 100)
  (h_columbian_price : columbian_price = 8.75)
  (h_final_mix_price : final_mix_price = 6.35)
  (h_columbian_amount : columbian_amount = 52) :
  let brazilian_amount : ℝ := total_mix - columbian_amount
  let brazilian_price : ℝ := (total_mix * final_mix_price - columbian_amount * columbian_price) / brazilian_amount
  brazilian_price = 3.75 := by
sorry


end NUMINAMATH_CALUDE_brazilian_coffee_price_l245_24595


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l245_24524

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then
    ⌈(2 : ℝ) / (x + 3)⌉
  else if x < -3 then
    ⌊(2 : ℝ) / (x + 3)⌋
  else
    0  -- Arbitrary value for x = -3, as g is not defined there

theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -3 → g x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l245_24524


namespace NUMINAMATH_CALUDE_yan_journey_ratio_l245_24582

/-- Represents a point on a line --/
structure Point :=
  (position : ℝ)

/-- Represents the scenario of Yan's journey --/
structure Journey :=
  (home : Point)
  (stadium : Point)
  (yan : Point)
  (walking_speed : ℝ)
  (cycling_speed : ℝ)

/-- The conditions of the journey --/
def journey_conditions (j : Journey) : Prop :=
  j.home.position < j.yan.position ∧
  j.yan.position < j.stadium.position ∧
  j.cycling_speed = 5 * j.walking_speed ∧
  (j.stadium.position - j.yan.position) / j.walking_speed =
    (j.yan.position - j.home.position) / j.walking_speed +
    (j.stadium.position - j.home.position) / j.cycling_speed

/-- The theorem to be proved --/
theorem yan_journey_ratio (j : Journey) (h : journey_conditions j) :
  (j.yan.position - j.home.position) / (j.stadium.position - j.yan.position) = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_yan_journey_ratio_l245_24582


namespace NUMINAMATH_CALUDE_pencil_profit_l245_24509

def pencil_problem (pencils : ℕ) (buy_price : ℚ) (sell_price : ℚ) : Prop :=
  let cost := (pencils : ℚ) * buy_price / 4
  let revenue := (pencils : ℚ) * sell_price / 5
  let profit := revenue - cost
  profit = 60

theorem pencil_profit : 
  pencil_problem 1200 3 4 :=
sorry

end NUMINAMATH_CALUDE_pencil_profit_l245_24509


namespace NUMINAMATH_CALUDE_easter_egg_ratio_l245_24559

def total_eggs : ℕ := 63
def hannah_eggs : ℕ := 42

theorem easter_egg_ratio :
  let helen_eggs := total_eggs - hannah_eggs
  (hannah_eggs : ℚ) / helen_eggs = 2 / 1 := by sorry

end NUMINAMATH_CALUDE_easter_egg_ratio_l245_24559


namespace NUMINAMATH_CALUDE_ellipse_max_ratio_l245_24541

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    where a > b > 0, prove that the maximum value of |FA|/|OH| is 1/4, 
    where F is the right focus, A is the right vertex, O is the center, 
    and H is the intersection of the right directrix with the x-axis. -/
theorem ellipse_max_ratio (a b : ℝ) (h : a > b ∧ b > 0) : 
  let e := Real.sqrt (1 - b^2 / a^2)  -- eccentricity
  ∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    (∀ (x' y' : ℝ), x'^2 / a^2 + y'^2 / b^2 = 1 → 
      (a - a * e) / (a^2 / (a * e)) ≤ 1/4) ∧
    (a - a * e) / (a^2 / (a * e)) = 1/4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_max_ratio_l245_24541


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l245_24581

-- Define the function f(x) = x^2 + 3x - 1
def f (x : ℝ) : ℝ := x^2 + 3*x - 1

-- State the theorem
theorem root_exists_in_interval :
  Continuous f →
  f 0 < 0 →
  f 0.5 > 0 →
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo 0 0.5 ∧ f x₀ = 0 :=
by
  sorry

#check root_exists_in_interval

end NUMINAMATH_CALUDE_root_exists_in_interval_l245_24581


namespace NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l245_24575

theorem smallest_x_satisfying_equation : 
  ∃ (x : ℝ), x > 0 ∧ 
  (⌊x^2⌋ : ℝ) - x * (⌊x⌋ : ℝ) = 8 ∧ 
  (∀ y : ℝ, y > 0 ∧ (⌊y^2⌋ : ℝ) - y * (⌊y⌋ : ℝ) = 8 → x ≤ y) ∧
  x = 89 / 9 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l245_24575


namespace NUMINAMATH_CALUDE_square_area_from_smaller_squares_l245_24599

/-- The area of a square composed of smaller squares -/
theorem square_area_from_smaller_squares
  (n : ℕ) -- number of smaller squares
  (side_length : ℝ) -- side length of each smaller square
  (h_n : n = 8) -- there are 8 smaller squares
  (h_side : side_length = 2) -- side length of each smaller square is 2 cm
  : n * side_length^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_smaller_squares_l245_24599


namespace NUMINAMATH_CALUDE_complex_equation_solution_count_l245_24574

theorem complex_equation_solution_count : 
  ∃! (c : ℝ), Complex.abs (2/3 - c * Complex.I) = 5/6 ∧ Complex.im (3 + c * Complex.I) > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_count_l245_24574


namespace NUMINAMATH_CALUDE_number_operations_l245_24540

theorem number_operations (x : ℚ) : (x - 5) / 7 = 7 → (x - 2) / 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_operations_l245_24540


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l245_24578

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 = 198 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l245_24578


namespace NUMINAMATH_CALUDE_factorial_less_than_power_l245_24550

theorem factorial_less_than_power (n : ℕ) (h : n > 1) : 
  Nat.factorial n < ((n + 1) / 2 : ℚ) ^ n := by
  sorry

end NUMINAMATH_CALUDE_factorial_less_than_power_l245_24550


namespace NUMINAMATH_CALUDE_banana_slices_per_yogurt_l245_24553

/-- Given that one banana yields 10 slices, 5 yogurts need to be made, and 4 bananas are bought,
    prove that 8 banana slices are needed for each yogurt. -/
theorem banana_slices_per_yogurt :
  let slices_per_banana : ℕ := 10
  let yogurts_to_make : ℕ := 5
  let bananas_bought : ℕ := 4
  let total_slices : ℕ := slices_per_banana * bananas_bought
  let slices_per_yogurt : ℕ := total_slices / yogurts_to_make
  slices_per_yogurt = 8 := by sorry

end NUMINAMATH_CALUDE_banana_slices_per_yogurt_l245_24553


namespace NUMINAMATH_CALUDE_salary_calculation_l245_24598

theorem salary_calculation (salary : ℚ) : 
  (salary * (1 - 0.2) * (1 - 0.1) * (1 - 0.1) = 1377) → 
  salary = 2125 := by
sorry

end NUMINAMATH_CALUDE_salary_calculation_l245_24598


namespace NUMINAMATH_CALUDE_hockey_puck_price_comparison_l245_24527

theorem hockey_puck_price_comparison (P : ℝ) (h : P > 0) : P > 0.99 * P := by
  sorry

end NUMINAMATH_CALUDE_hockey_puck_price_comparison_l245_24527


namespace NUMINAMATH_CALUDE_total_collection_is_32_49_l245_24523

/-- Represents the number of members in the group -/
def group_size : ℕ := 57

/-- Represents the contribution of each member in paise -/
def contribution_per_member : ℕ := group_size

/-- Converts paise to rupees -/
def paise_to_rupees (paise : ℕ) : ℚ :=
  (paise : ℚ) / 100

/-- Calculates the total collection amount in rupees -/
def total_collection : ℚ :=
  paise_to_rupees (group_size * contribution_per_member)

/-- Theorem stating that the total collection amount is 32.49 rupees -/
theorem total_collection_is_32_49 :
  total_collection = 32.49 := by sorry

end NUMINAMATH_CALUDE_total_collection_is_32_49_l245_24523


namespace NUMINAMATH_CALUDE_eulers_theorem_parallelepiped_l245_24529

/-- Represents a parallelepiped with edges a, b, c meeting at a vertex,
    face diagonals d, e, f, and space diagonal g. -/
structure Parallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  g : ℝ

/-- Euler's theorem for parallelepipeds:
    The sum of the squares of the edges and the space diagonal at one vertex
    is equal to the sum of the squares of the face diagonals. -/
theorem eulers_theorem_parallelepiped (p : Parallelepiped) :
  p.a^2 + p.b^2 + p.c^2 + p.g^2 = p.d^2 + p.e^2 + p.f^2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_theorem_parallelepiped_l245_24529


namespace NUMINAMATH_CALUDE_dressing_p_vinegar_percent_l245_24572

/-- Represents a salad dressing with a specific percentage of vinegar -/
structure SaladDressing where
  vinegar_percent : ℝ
  oil_percent : ℝ
  vinegar_oil_sum : vinegar_percent + oil_percent = 100

/-- The percentage of dressing P in the new mixture -/
def p_mixture_percent : ℝ := 10

/-- The percentage of dressing Q in the new mixture -/
def q_mixture_percent : ℝ := 100 - p_mixture_percent

/-- Dressing Q contains 10% vinegar -/
def dressing_q : SaladDressing := ⟨10, 90, by norm_num⟩

/-- The percentage of vinegar in the new mixture -/
def new_mixture_vinegar_percent : ℝ := 12

/-- Theorem stating that dressing P contains 30% vinegar -/
theorem dressing_p_vinegar_percent :
  ∃ (dressing_p : SaladDressing),
    dressing_p.vinegar_percent = 30 ∧
    (p_mixture_percent / 100 * dressing_p.vinegar_percent +
     q_mixture_percent / 100 * dressing_q.vinegar_percent = new_mixture_vinegar_percent) :=
by sorry

end NUMINAMATH_CALUDE_dressing_p_vinegar_percent_l245_24572


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_eight_pi_thirds_l245_24502

noncomputable def f (x : Real) : Real := Real.sqrt 3 * Real.sin x + Real.cos x

theorem sum_of_roots_equals_eight_pi_thirds (a : Real) :
  0 < a → a < 1 → ∃ x₁ x₂ : Real, 
    x₁ ∈ Set.Icc 0 (2 * Real.pi) ∧ 
    x₂ ∈ Set.Icc 0 (2 * Real.pi) ∧ 
    f x₁ = a ∧ 
    f x₂ = a ∧ 
    x₁ + x₂ = 8 * Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_eight_pi_thirds_l245_24502


namespace NUMINAMATH_CALUDE_min_sum_of_product_2550_l245_24512

theorem min_sum_of_product_2550 (a b c : ℕ+) (h : a * b * c = 2550) :
  ∃ (x y z : ℕ+), x * y * z = 2550 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 48 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_2550_l245_24512


namespace NUMINAMATH_CALUDE_chocolate_division_l245_24513

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_to_shaina : ℕ) :
  total_chocolate = 64/7 →
  num_piles = 6 →
  piles_to_shaina = 2 →
  piles_to_shaina * (total_chocolate / num_piles) = 64/21 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_division_l245_24513


namespace NUMINAMATH_CALUDE_largest_centrally_symmetric_polygon_area_l245_24537

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A polygon in a 2D plane --/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Checks if a polygon is centrally symmetric --/
def isCentrallySymmetric (p : Polygon) : Prop := sorry

/-- Checks if a polygon is inside a triangle --/
def isInsideTriangle (p : Polygon) (t : Triangle) : Prop := sorry

/-- Calculates the area of a polygon --/
def area (p : Polygon) : ℝ := sorry

/-- Calculates the area of a triangle --/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The theorem to be proved --/
theorem largest_centrally_symmetric_polygon_area (t : Triangle) : 
  ∃ (p : Polygon), 
    isCentrallySymmetric p ∧ 
    isInsideTriangle p t ∧ 
    (∀ (q : Polygon), isCentrallySymmetric q → isInsideTriangle q t → area q ≤ area p) ∧
    area p = (2/3) * triangleArea t := by
  sorry

end NUMINAMATH_CALUDE_largest_centrally_symmetric_polygon_area_l245_24537


namespace NUMINAMATH_CALUDE_repeating_decimal_equality_l245_24584

theorem repeating_decimal_equality (a : ℕ) : 
  1 ≤ a ∧ a ≤ 9 → (0.1 * a : ℚ) = 1 / a → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equality_l245_24584


namespace NUMINAMATH_CALUDE_coordinate_uniqueness_l245_24554

/-- A type representing a location description -/
inductive LocationDescription
| Coordinates (longitude : Real) (latitude : Real)
| CityLandmark (city : String) (landmark : String)
| Direction (angle : Real)
| VenueSeat (venue : String) (seat : String)

/-- Function to check if a location description uniquely determines a location -/
def uniquelyDeterminesLocation (desc : LocationDescription) : Prop :=
  match desc with
  | LocationDescription.Coordinates _ _ => True
  | _ => False

/-- Theorem stating that only coordinate-based descriptions uniquely determine locations -/
theorem coordinate_uniqueness 
  (descriptions : List LocationDescription) 
  (h_contains_coordinates : ∃ (long lat : Real), LocationDescription.Coordinates long lat ∈ descriptions) :
  ∃! (desc : LocationDescription), desc ∈ descriptions ∧ uniquelyDeterminesLocation desc :=
sorry

end NUMINAMATH_CALUDE_coordinate_uniqueness_l245_24554


namespace NUMINAMATH_CALUDE_polynomial_functional_equation_l245_24565

/-- Given a polynomial P : ℝ × ℝ → ℝ satisfying P(x - 1, y - 2x + 1) = P(x, y) for all x and y,
    there exists a polynomial Φ : ℝ → ℝ such that P(x, y) = Φ(y - x^2) for all x and y. -/
theorem polynomial_functional_equation
  (P : ℝ → ℝ → ℝ)
  (h : ∀ x y : ℝ, P (x - 1) (y - 2*x + 1) = P x y)
  : ∃ Φ : ℝ → ℝ, ∀ x y : ℝ, P x y = Φ (y - x^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_functional_equation_l245_24565


namespace NUMINAMATH_CALUDE_germination_expectation_l245_24510

/-- The germination rate of seeds -/
def germination_rate : ℝ := 0.8

/-- The number of seeds sown -/
def seeds_sown : ℕ := 100

/-- The expected number of germinated seeds -/
def expected_germinated_seeds : ℝ := germination_rate * seeds_sown

theorem germination_expectation :
  expected_germinated_seeds = 80 := by sorry

end NUMINAMATH_CALUDE_germination_expectation_l245_24510


namespace NUMINAMATH_CALUDE_minimize_theta_l245_24538

def angle : ℝ := -495

theorem minimize_theta : 
  ∃ (K : ℤ) (θ : ℝ), 
    angle = K * 360 + θ ∧ 
    ∀ (K' : ℤ) (θ' : ℝ), angle = K' * 360 + θ' → |θ| ≤ |θ'| ∧
    θ = -135 := by
  sorry

end NUMINAMATH_CALUDE_minimize_theta_l245_24538


namespace NUMINAMATH_CALUDE_store_opening_cost_l245_24515

/-- The cost to open Kim's store -/
def openingCost (monthlyRevenue : ℕ) (monthlyExpenses : ℕ) (monthsToPayback : ℕ) : ℕ :=
  (monthlyRevenue - monthlyExpenses) * monthsToPayback

/-- Theorem stating the cost to open Kim's store -/
theorem store_opening_cost : openingCost 4000 1500 10 = 25000 := by
  sorry

end NUMINAMATH_CALUDE_store_opening_cost_l245_24515


namespace NUMINAMATH_CALUDE_probability_two_teachers_in_A_l245_24500

def num_teachers : ℕ := 3
def num_places : ℕ := 2

def total_assignments : ℕ := num_places ^ num_teachers

def assignments_with_two_in_A : ℕ := (Nat.choose num_teachers 2)

theorem probability_two_teachers_in_A :
  (assignments_with_two_in_A : ℚ) / total_assignments = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_teachers_in_A_l245_24500


namespace NUMINAMATH_CALUDE_bags_needed_is_17_l245_24569

/-- Calculates the number of bags of special dog food needed for a puppy's first year --/
def bags_needed : ℕ :=
  let days_in_year : ℕ := 365
  let ounces_per_pound : ℕ := 16
  let bag_size : ℕ := 5 -- in pounds
  let initial_period : ℕ := 60 -- in days
  let initial_daily_amount : ℕ := 2 -- in ounces
  let later_daily_amount : ℕ := 4 -- in ounces
  
  let initial_total : ℕ := initial_period * initial_daily_amount
  let later_period : ℕ := days_in_year - initial_period
  let later_total : ℕ := later_period * later_daily_amount
  
  let total_ounces : ℕ := initial_total + later_total
  let total_pounds : ℕ := (total_ounces + ounces_per_pound - 1) / ounces_per_pound
  (total_pounds + bag_size - 1) / bag_size

theorem bags_needed_is_17 : bags_needed = 17 := by
  sorry

end NUMINAMATH_CALUDE_bags_needed_is_17_l245_24569


namespace NUMINAMATH_CALUDE_number_problem_l245_24535

theorem number_problem (x : ℝ) (h : 0.5 * x = (3/5) * x - 10) : x = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l245_24535


namespace NUMINAMATH_CALUDE_pulsar_rotation_scientific_notation_l245_24522

/-- The rotation period of the millisecond pulsar in seconds -/
def rotation_period : ℝ := 0.00519

/-- The coefficient in the scientific notation representation -/
def coefficient : ℝ := 5.19

/-- The exponent in the scientific notation representation -/
def exponent : ℤ := -3

theorem pulsar_rotation_scientific_notation :
  rotation_period = coefficient * (10 : ℝ) ^ exponent := by
  sorry

end NUMINAMATH_CALUDE_pulsar_rotation_scientific_notation_l245_24522


namespace NUMINAMATH_CALUDE_rectangular_prism_prime_edges_l245_24545

theorem rectangular_prism_prime_edges (a b c : ℕ) (k : ℕ) : 
  Prime a → Prime b → Prime c →
  ∃ p n : ℕ, Prime p ∧ 2 * (a * b + b * c + c * a) = p^n →
  (a = 2^k - 1 ∧ Prime (2^k - 1) ∧ b = 2 ∧ c = 2) ∨
  (b = 2^k - 1 ∧ Prime (2^k - 1) ∧ a = 2 ∧ c = 2) ∨
  (c = 2^k - 1 ∧ Prime (2^k - 1) ∧ a = 2 ∧ b = 2) :=
sorry

end NUMINAMATH_CALUDE_rectangular_prism_prime_edges_l245_24545


namespace NUMINAMATH_CALUDE_duck_cow_problem_l245_24507

theorem duck_cow_problem (D C : ℕ) : 
  2 * D + 4 * C = 2 * (D + C) + 28 → C = 14 := by
sorry

end NUMINAMATH_CALUDE_duck_cow_problem_l245_24507


namespace NUMINAMATH_CALUDE_pizza_slices_theorem_l245_24589

/-- Given a number of pizzas and slices per pizza, calculate the total number of slices -/
def total_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) : ℕ :=
  num_pizzas * slices_per_pizza

/-- Theorem: With 14 pizzas and 2 slices per pizza, the total number of slices is 28 -/
theorem pizza_slices_theorem : total_slices 14 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_theorem_l245_24589


namespace NUMINAMATH_CALUDE_min_value_expression_l245_24539

theorem min_value_expression (a b c : ℝ) (h1 : c > 0) (h2 : a ≠ 0) (h3 : b ≠ 0)
  (h4 : 4 * a^2 - 2 * a * b + 4 * b^2 - c = 0)
  (h5 : ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → 4 * x^2 - 2 * x * y + 4 * y^2 - c = 0 →
    |2 * a + b| ≥ |2 * x + y|) :
  ∃ (m : ℝ), m = -2 ∧ ∀ (x y z : ℝ), x ≠ 0 → y ≠ 0 → z > 0 →
    4 * x^2 - 2 * x * y + 4 * y^2 - z = 0 →
    3 / x - 4 / y + 5 / z ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l245_24539


namespace NUMINAMATH_CALUDE_parallel_perpendicular_implication_l245_24525

/-- Two lines are parallel -/
def parallel (m n : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular (l : Line) (α : Plane) : Prop := sorry

/-- Theorem: If two lines are parallel and one is perpendicular to a plane, 
    then the other is also perpendicular to that plane -/
theorem parallel_perpendicular_implication 
  (m n : Line) (α : Plane) : 
  parallel m n → perpendicular m α → perpendicular n α := by
  sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_implication_l245_24525


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l245_24534

theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∃ r : ℝ, ∀ n, a (n + 1) = r * a n) →
  a 3 * a 7 = 64 →
  a 5 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l245_24534


namespace NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l245_24514

/-- Represents a ratio of three ingredients -/
structure Ratio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- The original ratio of ingredients -/
def original_ratio : Ratio :=
  { flour := 10, water := 6, sugar := 3 }

/-- The new ratio after adjusting flour to water and flour to sugar -/
def new_ratio : Ratio :=
  { flour := 20, water := 6, sugar := 12 }

/-- The amount of water in the new recipe -/
def new_water_amount : ℚ := 2

theorem sugar_amount_in_new_recipe :
  let sugar_amount := (new_ratio.sugar / new_ratio.water) * new_water_amount
  sugar_amount = 4 := by sorry

end NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l245_24514


namespace NUMINAMATH_CALUDE_vector_magnitude_l245_24506

theorem vector_magnitude (a b : ℝ × ℝ) (m : ℝ) :
  a = (2, 1) →
  b = (3, m) →
  (∃ k : ℝ, (2 • a - b) = k • b) →
  ‖b‖ = (3 * Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_l245_24506


namespace NUMINAMATH_CALUDE_line_equation_l245_24587

theorem line_equation (slope_angle : Real) (y_intercept : Real) :
  slope_angle = Real.pi / 4 →
  y_intercept = 2 →
  ∀ x y : Real, y = x + y_intercept ↔ y = x + 2 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l245_24587


namespace NUMINAMATH_CALUDE_percentage_problem_l245_24519

theorem percentage_problem (n : ℝ) (p : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * n = 10 →
  (p / 100) * n = 120 →
  p = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l245_24519


namespace NUMINAMATH_CALUDE_geometric_sequence_extreme_points_l245_24555

/-- Given a geometric sequence {a_n} where a_3 and a_7 are extreme points of f(x) = (1/3)x^3 + 4x^2 + 9x - 1, prove a_5 = -3 -/
theorem geometric_sequence_extreme_points (a : ℕ → ℝ) (h_geometric : ∀ n, a (n+1) / a n = a (n+2) / a (n+1)) :
  (∀ x, (x^2 + 8*x + 9) * (x - a 3) * (x - a 7) ≥ 0) →
  a 3 * a 7 = 9 →
  a 3 + a 7 = -8 →
  a 5 = -3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_extreme_points_l245_24555


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l245_24546

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {a b c d : ℝ} : 
  (∀ x y : ℝ, a * x + b * y = 0 ↔ y = c * x + d) → b ≠ 0 → a / b = -c

/-- The value of m for which the lines 2x + my = 0 and y = 3x - 1 are parallel -/
theorem parallel_lines_m_value : 
  ∃ m : ℝ, (∀ x y : ℝ, 2 * x + m * y = 0 ↔ y = 3 * x - 1) ∧ m = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l245_24546


namespace NUMINAMATH_CALUDE_factor_x_sixth_plus_64_l245_24568

theorem factor_x_sixth_plus_64 (x : ℝ) : x^6 + 64 = (x^2 + 4) * (x^4 - 4*x^2 + 16) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_sixth_plus_64_l245_24568


namespace NUMINAMATH_CALUDE_pythagorean_theorem_l245_24577

-- Define a right-angled triangle
def RightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

-- Theorem statement
theorem pythagorean_theorem (a b c : ℝ) :
  RightTriangle a b c → a^2 + b^2 = c^2 :=
by
  sorry

end NUMINAMATH_CALUDE_pythagorean_theorem_l245_24577


namespace NUMINAMATH_CALUDE_inequality_equivalence_l245_24594

def solution_set (x y : ℝ) : Prop :=
  (x ≤ -1 ∧ y ≤ x + 2 ∧ y ≥ -x - 2) ∨
  (-1 < x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1) ∨
  (x > 1 ∧ y ≤ 2 - x ∧ y ≥ x - 2)

theorem inequality_equivalence (x y : ℝ) :
  |x - 1| + |x + 1| + |2 * y| ≤ 4 ↔ solution_set x y := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l245_24594


namespace NUMINAMATH_CALUDE_largest_k_for_inequality_l245_24533

theorem largest_k_for_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_3 : a + b + c = 3) :
  (∀ k : ℝ, 0 < k → k ≤ 5 → a^3 + b^3 + c^3 - 3 ≥ k * (3 - a*b - b*c - c*a)) ∧
  (∃ a₀ b₀ c₀ : ℝ, 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ + b₀ + c₀ = 3 ∧
    a₀^3 + b₀^3 + c₀^3 - 3 = 5 * (3 - a₀*b₀ - b₀*c₀ - c₀*a₀)) :=
by sorry

end NUMINAMATH_CALUDE_largest_k_for_inequality_l245_24533


namespace NUMINAMATH_CALUDE_max_points_is_700_l245_24548

/-- A board game with the following properties:
  - The board is 7 × 8 (7 rows and 8 columns)
  - Two players take turns placing pieces
  - The second player (Gretel) earns 4 points for each piece already in the same row
  - Gretel earns 3 points for each piece already in the same column
  - The game ends when all cells are filled -/
structure BoardGame where
  rows : Nat
  cols : Nat
  row_points : Nat
  col_points : Nat

/-- The maximum number of points Gretel can earn in the game -/
def max_points (game : BoardGame) : Nat :=
  700

/-- Theorem stating that the maximum number of points Gretel can earn is 700 -/
theorem max_points_is_700 (game : BoardGame) 
  (h1 : game.rows = 7) 
  (h2 : game.cols = 8) 
  (h3 : game.row_points = 4) 
  (h4 : game.col_points = 3) : 
  max_points game = 700 := by
  sorry

#check max_points_is_700

end NUMINAMATH_CALUDE_max_points_is_700_l245_24548


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l245_24531

def arithmetic_sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ i j : ℕ, i < n → j < n → a (i + 1) - a i = a (j + 1) - a j

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (n : ℕ) :
  arithmetic_sequence a n →
  a 0 = 3 →
  a 1 = 8 →
  a 2 = 13 →
  a (n - 1) = 38 →
  a (n - 2) + a (n - 3) = 61 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l245_24531


namespace NUMINAMATH_CALUDE_system_solution_l245_24593

theorem system_solution : 
  ∃! (s : Set (ℝ × ℝ)), s = {(2, 4), (4, 2)} ∧
  ∀ (x y : ℝ), (x, y) ∈ s ↔ 
    ((x / y + y / x) * (x + y) = 15 ∧
     (x^2 / y^2 + y^2 / x^2) * (x^2 + y^2) = 85) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l245_24593


namespace NUMINAMATH_CALUDE_robins_gum_pieces_l245_24504

/-- 
Given that Robin had an initial number of gum pieces, her brother gave her 26 more,
and now she has 44 pieces in total, prove that she initially had 18 pieces.
-/
theorem robins_gum_pieces (x : ℕ) : x + 26 = 44 → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_robins_gum_pieces_l245_24504


namespace NUMINAMATH_CALUDE_traffic_light_statement_correct_l245_24558

/-- A traffic light state can be either red or green -/
inductive TrafficLightState
  | Red
  | Green

/-- A traffic light intersection scenario -/
structure TrafficLightIntersection where
  state : TrafficLightState

/-- The statement about traffic light outcomes is correct -/
theorem traffic_light_statement_correct :
  ∀ (intersection : TrafficLightIntersection),
    (intersection.state = TrafficLightState.Red) ∨
    (intersection.state = TrafficLightState.Green) :=
by sorry

end NUMINAMATH_CALUDE_traffic_light_statement_correct_l245_24558


namespace NUMINAMATH_CALUDE_premium_rate_calculation_l245_24505

/-- Given a tempo insured to 4/5 of its original value of $87,500, with a premium of $910,
    the rate of the premium is 1.3%. -/
theorem premium_rate_calculation (original_value : ℝ) (insurance_ratio : ℝ) (premium : ℝ) :
  original_value = 87500 →
  insurance_ratio = 4 / 5 →
  premium = 910 →
  (premium / (insurance_ratio * original_value)) * 100 = 1.3 := by
  sorry

end NUMINAMATH_CALUDE_premium_rate_calculation_l245_24505


namespace NUMINAMATH_CALUDE_line_property_l245_24580

/-- Given a line passing through points (1, -1) and (3, 7), 
    prove that 3m - b = 17, where m is the slope and b is the y-intercept. -/
theorem line_property (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b) →  -- Line equation
  (1 : ℝ) * m + b = -1 →        -- Point (1, -1) satisfies the equation
  (3 : ℝ) * m + b = 7 →         -- Point (3, 7) satisfies the equation
  3 * m - b = 17 := by
sorry

end NUMINAMATH_CALUDE_line_property_l245_24580


namespace NUMINAMATH_CALUDE_vessel_weight_percentage_l245_24591

theorem vessel_weight_percentage (E P : ℝ) 
  (h1 : (1/2) * (E + P) = E + 0.42857142857142855 * P) : 
  (E / (E + P)) * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_vessel_weight_percentage_l245_24591


namespace NUMINAMATH_CALUDE_almond_walnut_ratio_l245_24536

/-- Given a mixture of almonds and walnuts, prove the ratio of almonds to walnuts -/
theorem almond_walnut_ratio 
  (total_weight : ℝ) 
  (almond_weight : ℝ) 
  (almond_parts : ℕ) 
  (h1 : total_weight = 280) 
  (h2 : almond_weight = 200) 
  (h3 : almond_parts = 5) :
  ∃ (walnut_parts : ℕ), 
    (almond_parts : ℝ) / walnut_parts = 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_almond_walnut_ratio_l245_24536


namespace NUMINAMATH_CALUDE_irrational_among_options_l245_24517

theorem irrational_among_options : 
  (¬ (∃ (a b : ℤ), -Real.sqrt 3 = (a : ℚ) / (b : ℚ) ∧ b ≠ 0)) ∧
  (∃ (a b : ℤ), (-2 : ℚ) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0) ∧
  (∃ (a b : ℤ), (0.1010 : ℚ) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0) ∧
  (∃ (a b : ℤ), (1/3 : ℚ) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_irrational_among_options_l245_24517


namespace NUMINAMATH_CALUDE_attendance_theorem_l245_24530

/-- Represents the admission prices and attendance for a play -/
structure PlayAttendance where
  adult_price : ℕ
  child_price : ℕ
  total_receipts : ℕ
  num_children : ℕ

/-- Calculates the total number of attendees given the play attendance data -/
def total_attendees (p : PlayAttendance) : ℕ :=
  p.num_children + (p.total_receipts - p.num_children * p.child_price) / p.adult_price

/-- Theorem stating that given the specific conditions, the total number of attendees is 610 -/
theorem attendance_theorem (p : PlayAttendance) 
    (h1 : p.adult_price = 2)
    (h2 : p.child_price = 1)
    (h3 : p.total_receipts = 960)
    (h4 : p.num_children = 260) : 
  total_attendees p = 610 := by
  sorry

#eval total_attendees ⟨2, 1, 960, 260⟩

end NUMINAMATH_CALUDE_attendance_theorem_l245_24530


namespace NUMINAMATH_CALUDE_power_calculation_l245_24573

theorem power_calculation : (9^4 * 3^10) / 27^7 = 1 / 27 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l245_24573


namespace NUMINAMATH_CALUDE_no_real_solution_for_equation_l245_24516

theorem no_real_solution_for_equation :
  ¬ ∃ x : ℝ, (Real.sqrt (4 * x + 2) + 1) / Real.sqrt (8 * x + 10) = 2 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_for_equation_l245_24516


namespace NUMINAMATH_CALUDE_min_n_for_cuboid_sum_l245_24596

theorem min_n_for_cuboid_sum (n : ℕ) : (∀ m : ℕ, m > 0 ∧ 128 * m > 2011 → n ≤ m) ∧ n > 0 ∧ 128 * n > 2011 ↔ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_n_for_cuboid_sum_l245_24596


namespace NUMINAMATH_CALUDE_perfect_square_sum_in_pile_l245_24511

theorem perfect_square_sum_in_pile (n : ℕ) (h : n ≥ 100) :
  ∀ (S₁ S₂ : Set ℕ), 
    (∀ k, n ≤ k ∧ k ≤ 2*n → k ∈ S₁ ∨ k ∈ S₂) →
    (S₁ ∩ S₂ = ∅) →
    (∃ (a b : ℕ), (a ∈ S₁ ∧ b ∈ S₁ ∧ a ≠ b ∧ ∃ (m : ℕ), a + b = m^2) ∨
                   (a ∈ S₂ ∧ b ∈ S₂ ∧ a ≠ b ∧ ∃ (m : ℕ), a + b = m^2)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_sum_in_pile_l245_24511


namespace NUMINAMATH_CALUDE_distance_after_two_hours_l245_24561

/-- Alice's walking speed in miles per minute -/
def alice_speed : ℚ := 1 / 20

/-- Bob's jogging speed in miles per minute -/
def bob_speed : ℚ := 3 / 40

/-- Time elapsed in minutes -/
def time_elapsed : ℚ := 120

/-- The distance between Alice and Bob after 2 hours -/
def distance_between : ℚ := alice_speed * time_elapsed + bob_speed * time_elapsed

theorem distance_after_two_hours :
  distance_between = 15 := by sorry

end NUMINAMATH_CALUDE_distance_after_two_hours_l245_24561
