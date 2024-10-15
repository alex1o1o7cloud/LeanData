import Mathlib

namespace NUMINAMATH_CALUDE_product_of_integers_l3304_330442

theorem product_of_integers (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0)
  (sum_eq : p + q + r = 24)
  (frac_eq : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 240 / (p * q * r) = 1) :
  p * q * r = 384 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l3304_330442


namespace NUMINAMATH_CALUDE_f_is_odd_f_def_nonneg_f_neg_one_eq_neg_two_l3304_330452

/-- An odd function f defined on ℝ with f(x) = x(1+x) for x ≥ 0 -/
def f : ℝ → ℝ :=
  sorry

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
  sorry

theorem f_def_nonneg : ∀ x : ℝ, x ≥ 0 → f x = x * (1 + x) :=
  sorry

theorem f_neg_one_eq_neg_two : f (-1) = -2 :=
  sorry

end NUMINAMATH_CALUDE_f_is_odd_f_def_nonneg_f_neg_one_eq_neg_two_l3304_330452


namespace NUMINAMATH_CALUDE_anya_lost_games_l3304_330448

/-- Represents a girl in the table tennis game -/
inductive Girl
| Anya
| Bella
| Valya
| Galya
| Dasha

/-- Represents the number of games played by each girl -/
def games_played (g : Girl) : ℕ :=
  match g with
  | .Anya => 4
  | .Bella => 6
  | .Valya => 7
  | .Galya => 10
  | .Dasha => 11

/-- The total number of games played -/
def total_games : ℕ := 19

/-- Theorem stating that Anya lost in specific games -/
theorem anya_lost_games :
  ∃ (lost_games : List ℕ),
    lost_games = [4, 8, 12, 16] ∧
    (∀ g ∈ lost_games, g ≤ total_games) ∧
    (∀ g ∈ lost_games, ∃ i, g = 4 * i) ∧
    lost_games.length = games_played Girl.Anya := by
  sorry

end NUMINAMATH_CALUDE_anya_lost_games_l3304_330448


namespace NUMINAMATH_CALUDE_divisibility_by_six_l3304_330469

theorem divisibility_by_six (a b c : ℤ) (h : 18 ∣ (a^3 + b^3 + c^3)) : 6 ∣ (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_six_l3304_330469


namespace NUMINAMATH_CALUDE_inverse_of_composed_linear_functions_l3304_330422

/-- Given two functions p and q, we define r as their composition and prove its inverse -/
theorem inverse_of_composed_linear_functions 
  (p q r : ℝ → ℝ)
  (hp : ∀ x, p x = 4 * x - 7)
  (hq : ∀ x, q x = 3 * x + 2)
  (hr : ∀ x, r x = p (q x))
  : (∀ x, r x = 12 * x + 1) ∧ 
    (∀ x, Function.invFun r x = (x - 1) / 12) := by
  sorry


end NUMINAMATH_CALUDE_inverse_of_composed_linear_functions_l3304_330422


namespace NUMINAMATH_CALUDE_remaining_student_l3304_330498

theorem remaining_student (n : ℕ) (hn : n ≤ 2002) : n % 1331 = 0 ↔ n = 1331 :=
by sorry

#check remaining_student

end NUMINAMATH_CALUDE_remaining_student_l3304_330498


namespace NUMINAMATH_CALUDE_harkamal_mangoes_l3304_330416

/-- Calculates the amount of mangoes purchased given the total cost, grape quantity, grape price, and mango price -/
def mangoes_purchased (total_cost : ℕ) (grape_quantity : ℕ) (grape_price : ℕ) (mango_price : ℕ) : ℕ :=
  (total_cost - grape_quantity * grape_price) / mango_price

theorem harkamal_mangoes :
  mangoes_purchased 1145 8 70 65 = 9 := by
sorry

end NUMINAMATH_CALUDE_harkamal_mangoes_l3304_330416


namespace NUMINAMATH_CALUDE_min_detectors_for_ship_detection_l3304_330435

/-- Represents a cell on the board -/
structure Cell :=
  (x : Fin 7)
  (y : Fin 7)

/-- Represents a 2x2 ship placement on the board -/
structure Ship :=
  (topLeft : Cell)

/-- Represents a detector placement on the board -/
structure Detector :=
  (position : Cell)

/-- A function that determines if a ship occupies a given cell -/
def shipOccupies (s : Ship) (c : Cell) : Prop :=
  s.topLeft.x ≤ c.x ∧ c.x < s.topLeft.x + 2 ∧
  s.topLeft.y ≤ c.y ∧ c.y < s.topLeft.y + 2

/-- A function that determines if a detector can detect a ship -/
def detectorDetects (d : Detector) (s : Ship) : Prop :=
  shipOccupies s d.position

/-- The main theorem stating that 16 detectors are sufficient and necessary -/
theorem min_detectors_for_ship_detection :
  ∃ (detectors : Finset Detector),
    (detectors.card = 16) ∧
    (∀ (s : Ship), ∃ (d : Detector), d ∈ detectors ∧ detectorDetects d s) ∧
    (∀ (detectors' : Finset Detector),
      detectors'.card < 16 →
      ∃ (s : Ship), ∀ (d : Detector), d ∈ detectors' → ¬detectorDetects d s) :=
by sorry

end NUMINAMATH_CALUDE_min_detectors_for_ship_detection_l3304_330435


namespace NUMINAMATH_CALUDE_mean_of_remaining_numbers_l3304_330441

def numbers : List ℕ := [1877, 1999, 2039, 2045, 2119, 2131]

theorem mean_of_remaining_numbers :
  ∀ (subset : List ℕ),
    subset.length = 4 ∧
    subset ⊆ numbers ∧
    (subset.sum : ℚ) / 4 = 2015 →
    let remaining := numbers.filter (λ x => x ∉ subset)
    (remaining.sum : ℚ) / 2 = 2075 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_remaining_numbers_l3304_330441


namespace NUMINAMATH_CALUDE_band_formation_proof_l3304_330405

/-- Represents the number of columns in the rectangular formation -/
def n : ℕ := 14

/-- The total number of band members -/
def total_members : ℕ := n * (n + 7)

/-- The side length of the square formation -/
def square_side : ℕ := 17

theorem band_formation_proof :
  -- Square formation condition
  total_members = square_side ^ 2 + 5 ∧
  -- Rectangular formation condition
  total_members = n * (n + 7) ∧
  -- Maximum number of members
  total_members = 294 ∧
  -- No larger n satisfies the conditions
  ∀ m : ℕ, m > n → ¬(∃ k : ℕ, m * (m + 7) = k ^ 2 + 5) :=
by sorry

end NUMINAMATH_CALUDE_band_formation_proof_l3304_330405


namespace NUMINAMATH_CALUDE_special_triangle_AB_length_l3304_330421

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Point K on BC -/
  K : ℝ × ℝ
  /-- Point M on AB -/
  M : ℝ × ℝ
  /-- Point N on AC -/
  N : ℝ × ℝ
  /-- AC length is 18 -/
  h_AC : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 18
  /-- BC length is 21 -/
  h_BC : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 21
  /-- K is midpoint of BC -/
  h_K_midpoint : K = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  /-- M is midpoint of AB -/
  h_M_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  /-- AN length is 6 -/
  h_AN : Real.sqrt ((N.1 - A.1)^2 + (N.2 - A.2)^2) = 6
  /-- MN = KN -/
  h_MN_eq_KN : Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) = Real.sqrt ((N.1 - K.1)^2 + (N.2 - K.2)^2)

/-- The length of AB in the special triangle is 15 -/
theorem special_triangle_AB_length (t : SpecialTriangle) : 
  Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_AB_length_l3304_330421


namespace NUMINAMATH_CALUDE_kim_total_points_l3304_330415

/-- Represents the points structure of a math contest --/
structure ContestPoints where
  easy : Nat
  average : Nat
  hard : Nat
  expert : Nat
  bonusPerComplex : Nat

/-- Represents a contestant's performance in the math contest --/
structure ContestPerformance where
  points : ContestPoints
  easyCorrect : Nat
  averageCorrect : Nat
  hardCorrect : Nat
  expertCorrect : Nat
  complexSolved : Nat

/-- Calculates the total points for a contestant --/
def calculateTotalPoints (performance : ContestPerformance) : Nat :=
  performance.easyCorrect * performance.points.easy +
  performance.averageCorrect * performance.points.average +
  performance.hardCorrect * performance.points.hard +
  performance.expertCorrect * performance.points.expert +
  performance.complexSolved * performance.points.bonusPerComplex

/-- Theorem stating that Kim's total points in the contest equal 61 --/
theorem kim_total_points :
  let contestPoints : ContestPoints := {
    easy := 2,
    average := 3,
    hard := 5,
    expert := 7,
    bonusPerComplex := 1
  }
  let kimPerformance : ContestPerformance := {
    points := contestPoints,
    easyCorrect := 6,
    averageCorrect := 2,
    hardCorrect := 4,
    expertCorrect := 3,
    complexSolved := 2
  }
  calculateTotalPoints kimPerformance = 61 := by
  sorry


end NUMINAMATH_CALUDE_kim_total_points_l3304_330415


namespace NUMINAMATH_CALUDE_fraction_equality_implies_values_l3304_330400

theorem fraction_equality_implies_values (A B : ℚ) :
  (∀ x : ℚ, x ≠ 2 ∧ x ≠ 5 ∧ x^2 - 7*x + 10 ≠ 0 →
    (B*x - 7) / (x^2 - 7*x + 10) = A / (x - 2) + 5 / (x - 5)) →
  A = -3/5 ∧ B = 22/5 ∧ A + B = 19/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_values_l3304_330400


namespace NUMINAMATH_CALUDE_a_work_days_l3304_330413

/-- The number of days B takes to finish the work alone -/
def b_days : ℝ := 15

/-- The number of days A and B work together -/
def together_days : ℝ := 2

/-- The number of days B works alone after A leaves -/
def b_alone_days : ℝ := 7

/-- The total amount of work to be done -/
def total_work : ℝ := 1

-- The theorem to prove
theorem a_work_days : 
  ∃ (x : ℝ), 
    x > 0 ∧ 
    together_days * (1/x + 1/b_days) + b_alone_days * (1/b_days) = total_work ∧ 
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_a_work_days_l3304_330413


namespace NUMINAMATH_CALUDE_andrew_age_is_five_l3304_330446

/-- Andrew's age in years -/
def andrew_age : ℕ := 5

/-- Andrew's grandfather's age in years -/
def grandfather_age : ℕ := 10 * andrew_age

/-- The age difference between Andrew's grandfather and Andrew when Andrew was born -/
def age_difference_at_birth : ℕ := 45

theorem andrew_age_is_five :
  andrew_age = 5 ∧
  grandfather_age = 10 * andrew_age ∧
  grandfather_age - andrew_age = age_difference_at_birth :=
by sorry

end NUMINAMATH_CALUDE_andrew_age_is_five_l3304_330446


namespace NUMINAMATH_CALUDE_sum_of_sixth_powers_mod_seven_l3304_330431

theorem sum_of_sixth_powers_mod_seven :
  (1^6 + 2^6 + 3^6 + 4^6 + 5^6 + 6^6) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sixth_powers_mod_seven_l3304_330431


namespace NUMINAMATH_CALUDE_smallest_lychee_count_correct_l3304_330462

/-- The smallest number of lychees satisfying the distribution condition -/
def smallest_lychee_count : ℕ := 839

/-- Checks if a number satisfies the lychee distribution condition -/
def satisfies_condition (x : ℕ) : Prop :=
  ∀ n : ℕ, 3 ≤ n → n ≤ 8 → x % n = n - 1

theorem smallest_lychee_count_correct :
  satisfies_condition smallest_lychee_count ∧
  ∀ y : ℕ, y < smallest_lychee_count → ¬(satisfies_condition y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_lychee_count_correct_l3304_330462


namespace NUMINAMATH_CALUDE_f_ratio_is_integer_l3304_330430

/-- Sequence a_n defined recursively -/
def a (r s : ℕ+) : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => r * a r s (n + 1) + s * a r s n

/-- Product f_n of the first n terms of a_n -/
def f (r s : ℕ+) : ℕ → ℕ
  | 0 => 1
  | (n + 1) => f r s n * a r s (n + 1)

/-- Main theorem: f_n / (f_k * f_(n-k)) is an integer for 0 < k < n -/
theorem f_ratio_is_integer (r s : ℕ+) (n k : ℕ) (h1 : 0 < k) (h2 : k < n) :
  ∃ m : ℕ, f r s n = m * (f r s k * f r s (n - k)) := by
  sorry

end NUMINAMATH_CALUDE_f_ratio_is_integer_l3304_330430


namespace NUMINAMATH_CALUDE_school_band_seats_l3304_330458

/-- Calculates the total number of seats needed for a school band given the number of players for each instrument. -/
def total_seats (flute trumpet trombone drummer clarinet french_horn : ℕ) : ℕ :=
  flute + trumpet + trombone + drummer + clarinet + french_horn

/-- Proves that the total number of seats needed for the school band is 65. -/
theorem school_band_seats : ∃ (flute trumpet trombone drummer clarinet french_horn : ℕ),
  flute = 5 ∧
  trumpet = 3 * flute ∧
  trombone = trumpet - 8 ∧
  drummer = trombone + 11 ∧
  clarinet = 2 * flute ∧
  french_horn = trombone + 3 ∧
  total_seats flute trumpet trombone drummer clarinet french_horn = 65 := by
  sorry

#eval total_seats 5 15 7 18 10 10

end NUMINAMATH_CALUDE_school_band_seats_l3304_330458


namespace NUMINAMATH_CALUDE_arman_work_hours_l3304_330485

/-- Proves that Arman worked 35 hours last week given the conditions of his work schedule and pay. -/
theorem arman_work_hours : ∀ (last_week_hours : ℝ),
  (last_week_hours * 10 + 40 * 10.5 = 770) →
  last_week_hours = 35 := by
  sorry

end NUMINAMATH_CALUDE_arman_work_hours_l3304_330485


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3304_330434

/-- The function representing y = 2x^2 --/
def f (x : ℝ) : ℝ := 2 * x^2

/-- The function representing y = 4x + c --/
def g (c : ℝ) (x : ℝ) : ℝ := 4 * x + c

/-- The condition for two identical solutions --/
def has_two_identical_solutions (c : ℝ) : Prop :=
  ∃! x : ℝ, f x = g c x ∧ ∀ y : ℝ, f y = g c y → y = x

theorem unique_solution_condition (c : ℝ) :
  has_two_identical_solutions c ↔ c = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3304_330434


namespace NUMINAMATH_CALUDE_equation_solutions_l3304_330477

theorem equation_solutions : ∃ (x₁ x₂ : ℚ), 
  (x₁ = -1/2 ∧ x₂ = 3/4) ∧ 
  (∀ x : ℚ, 4*x*(2*x+1) = 3*(2*x+1) ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3304_330477


namespace NUMINAMATH_CALUDE_girls_fraction_is_37_75_l3304_330479

/-- Represents a school with a given number of students and boy-to-girl ratio -/
structure School where
  total_students : ℕ
  boys_ratio : ℕ
  girls_ratio : ℕ

/-- Calculates the number of girls in a school -/
def girls_count (s : School) : ℚ :=
  (s.total_students : ℚ) * s.girls_ratio / (s.boys_ratio + s.girls_ratio)

/-- Calculates the fraction of girls in a gathering of two schools -/
def girls_fraction (s1 s2 : School) : ℚ :=
  (girls_count s1 + girls_count s2) / (s1.total_students + s2.total_students)

theorem girls_fraction_is_37_75 (school_a school_b : School)
  (ha : school_a.total_students = 240 ∧ school_a.boys_ratio = 3 ∧ school_a.girls_ratio = 2)
  (hb : school_b.total_students = 210 ∧ school_b.boys_ratio = 2 ∧ school_b.girls_ratio = 3) :
  girls_fraction school_a school_b = 37 / 75 := by
  sorry

end NUMINAMATH_CALUDE_girls_fraction_is_37_75_l3304_330479


namespace NUMINAMATH_CALUDE_largest_side_is_sixty_l3304_330472

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  perimeter_eq : length * 2 + width * 2 = 180
  area_eq : length * width = 10 * 180

/-- The largest side of a SpecialRectangle is 60 feet -/
theorem largest_side_is_sixty (r : SpecialRectangle) : 
  max r.length r.width = 60 := by
  sorry

#check largest_side_is_sixty

end NUMINAMATH_CALUDE_largest_side_is_sixty_l3304_330472


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l3304_330425

theorem decimal_sum_to_fraction :
  (0.2 + 0.04 + 0.006 + 0.0008 + 0.00010 : ℚ) = 12345 / 160000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l3304_330425


namespace NUMINAMATH_CALUDE_vertical_line_equation_l3304_330437

/-- A line passing through the point (-2,1) with an undefined slope (vertical line) has the equation x + 2 = 0 -/
theorem vertical_line_equation : 
  ∀ (l : Set (ℝ × ℝ)), 
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x = -2) → 
  (-2, 1) ∈ l → 
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_vertical_line_equation_l3304_330437


namespace NUMINAMATH_CALUDE_same_solution_equations_l3304_330471

theorem same_solution_equations (x c : ℝ) : 
  (3 * x + 9 = 0) ∧ (c * x - 5 = -11) → c = 2 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_equations_l3304_330471


namespace NUMINAMATH_CALUDE_nancy_spent_95_40_l3304_330404

/-- The total amount Nancy spends on beads -/
def total_spent (crystal_price metal_price : ℚ) (crystal_sets metal_sets : ℕ) 
  (crystal_discount metal_tax : ℚ) : ℚ :=
  let crystal_cost := crystal_price * crystal_sets
  let metal_cost := metal_price * metal_sets
  let discounted_crystal := crystal_cost * (1 - crystal_discount)
  let taxed_metal := metal_cost * (1 + metal_tax)
  discounted_crystal + taxed_metal

/-- Theorem: Nancy spends $95.40 on beads -/
theorem nancy_spent_95_40 : 
  total_spent 12 15 3 4 (1/10) (1/20) = 95.4 := by
  sorry

end NUMINAMATH_CALUDE_nancy_spent_95_40_l3304_330404


namespace NUMINAMATH_CALUDE_fresh_to_dried_grapes_l3304_330487

/-- Given fresh grapes with 60% water content and dried grapes with 20% water content,
    prove that 15 kg of dried grapes comes from 30 kg of fresh grapes. -/
theorem fresh_to_dried_grapes (fresh_water_content : ℝ) (dried_water_content : ℝ) 
  (dried_weight : ℝ) (fresh_weight : ℝ) : 
  fresh_water_content = 0.6 →
  dried_water_content = 0.2 →
  dried_weight = 15 →
  (1 - fresh_water_content) * fresh_weight = (1 - dried_water_content) * dried_weight →
  fresh_weight = 30 := by
sorry

end NUMINAMATH_CALUDE_fresh_to_dried_grapes_l3304_330487


namespace NUMINAMATH_CALUDE_square_to_rectangle_area_increase_l3304_330453

theorem square_to_rectangle_area_increase (a : ℝ) (h : a > 0) :
  let square_area := a ^ 2
  let rectangle_length := a * 1.4
  let rectangle_breadth := a * 1.3
  let rectangle_area := rectangle_length * rectangle_breadth
  rectangle_area - square_area = 0.82 * square_area := by sorry

end NUMINAMATH_CALUDE_square_to_rectangle_area_increase_l3304_330453


namespace NUMINAMATH_CALUDE_percentage_increase_l3304_330470

theorem percentage_increase (x : ℝ) (h : x = 89.6) :
  ((x - 80) / 80) * 100 = 12 := by sorry

end NUMINAMATH_CALUDE_percentage_increase_l3304_330470


namespace NUMINAMATH_CALUDE_prob_one_pilot_hits_l3304_330461

/-- The probability that exactly one of two independent events occurs,
    given their individual probabilities of occurrence. -/
def prob_exactly_one (p_a p_b : ℝ) : ℝ := p_a * (1 - p_b) + (1 - p_a) * p_b

/-- The probability of pilot A hitting the target -/
def p_a : ℝ := 0.4

/-- The probability of pilot B hitting the target -/
def p_b : ℝ := 0.5

/-- Theorem: The probability that exactly one pilot hits the target is 0.5 -/
theorem prob_one_pilot_hits : prob_exactly_one p_a p_b = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_pilot_hits_l3304_330461


namespace NUMINAMATH_CALUDE_unique_coefficients_sum_l3304_330427

theorem unique_coefficients_sum : 
  let y : ℝ := Real.sqrt ((Real.sqrt 75 / 3) - 5/2)
  ∃! (a b c : ℕ+), 
    y^100 = 3*y^98 + 15*y^96 + 12*y^94 - 2*y^50 + (a : ℝ)*y^46 + (b : ℝ)*y^44 + (c : ℝ)*y^40 ∧
    a + b + c = 66 := by sorry

end NUMINAMATH_CALUDE_unique_coefficients_sum_l3304_330427


namespace NUMINAMATH_CALUDE_movie_length_after_cuts_l3304_330466

def original_length : ℝ := 97
def cut_scene1 : ℝ := 4.5
def cut_scene2 : ℝ := 2.75
def cut_scene3 : ℝ := 6.25

theorem movie_length_after_cuts :
  original_length - (cut_scene1 + cut_scene2 + cut_scene3) = 83.5 := by
  sorry

end NUMINAMATH_CALUDE_movie_length_after_cuts_l3304_330466


namespace NUMINAMATH_CALUDE_cafeteria_red_apples_l3304_330468

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := 42

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 7

/-- The number of students who wanted fruit -/
def students_wanting_fruit : ℕ := 9

/-- The number of extra apples -/
def extra_apples : ℕ := 40

/-- Theorem: The cafeteria ordered 42 red apples -/
theorem cafeteria_red_apples :
  red_apples = 42 ∧
  red_apples + green_apples = students_wanting_fruit + extra_apples :=
sorry

end NUMINAMATH_CALUDE_cafeteria_red_apples_l3304_330468


namespace NUMINAMATH_CALUDE_two_books_from_different_genres_l3304_330473

/-- The number of ways to choose two books from different genres -/
def choose_two_books (mystery : ℕ) (fantasy : ℕ) (biography : ℕ) : ℕ :=
  mystery * fantasy + mystery * biography + fantasy * biography

/-- Theorem: Given 5 mystery novels, 3 fantasy novels, and 2 biographies,
    the number of ways to choose 2 books from different genres is 31 -/
theorem two_books_from_different_genres :
  choose_two_books 5 3 2 = 31 := by
  sorry

end NUMINAMATH_CALUDE_two_books_from_different_genres_l3304_330473


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3304_330443

/-- Converts a repeating decimal with a single repeating digit to a rational number -/
def repeating_decimal_to_rational (n : ℕ) : ℚ := n / 9

theorem sum_of_repeating_decimals :
  repeating_decimal_to_rational 6 + 
  repeating_decimal_to_rational 2 - 
  repeating_decimal_to_rational 4 + 
  repeating_decimal_to_rational 9 = 13 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3304_330443


namespace NUMINAMATH_CALUDE_not_even_and_composite_two_l3304_330499

/-- Definition of an even number -/
def IsEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- Definition of a composite number -/
def IsComposite (n : ℕ) : Prop := ∃ a b : ℕ, a ≠ 1 ∧ b ≠ 1 ∧ n = a * b

/-- Theorem: It is false that 2 is both an even number and a composite number -/
theorem not_even_and_composite_two : ¬(IsEven 2 ∧ IsComposite 2) := by
  sorry

end NUMINAMATH_CALUDE_not_even_and_composite_two_l3304_330499


namespace NUMINAMATH_CALUDE_equation_general_form_l3304_330419

theorem equation_general_form :
  ∀ x : ℝ, (x - 1) * (2 * x + 1) = 2 ↔ 2 * x^2 - x - 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_general_form_l3304_330419


namespace NUMINAMATH_CALUDE_jason_omelet_eggs_l3304_330439

/-- The number of eggs Jason consumes in two weeks -/
def total_eggs : ℕ := 42

/-- The number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- The number of eggs Jason uses for his omelet each morning -/
def eggs_per_day : ℚ := total_eggs / days_in_two_weeks

theorem jason_omelet_eggs : eggs_per_day = 3 := by
  sorry

end NUMINAMATH_CALUDE_jason_omelet_eggs_l3304_330439


namespace NUMINAMATH_CALUDE_evies_age_l3304_330402

theorem evies_age (x : ℕ) : x + 4 = 3 * (x - 2) → x + 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_evies_age_l3304_330402


namespace NUMINAMATH_CALUDE_farm_animals_count_l3304_330423

/-- Represents the number of animals on a farm --/
structure FarmAnimals where
  cows : ℕ
  dogs : ℕ
  cats : ℕ
  sheep : ℕ
  chickens : ℕ

/-- Calculates the total number of animals on the farm --/
def totalAnimals (farm : FarmAnimals) : ℕ :=
  farm.cows + farm.dogs + farm.cats + farm.sheep + farm.chickens

/-- Represents the initial state of the farm --/
def initialFarm : FarmAnimals where
  cows := 120
  dogs := 18
  cats := 6
  sheep := 0
  chickens := 288

/-- Applies the changes to the farm as described in the problem --/
def applyChanges (farm : FarmAnimals) : FarmAnimals :=
  let soldCows := farm.cows / 4
  let soldDogs := farm.dogs * 3 / 5
  let remainingDogs := farm.dogs - soldDogs + soldDogs  -- Sell and add back equal number
  { cows := farm.cows - soldCows,
    dogs := remainingDogs,
    cats := farm.cats,
    sheep := remainingDogs / 2,
    chickens := farm.chickens * 3 / 2 }  -- 50% increase

theorem farm_animals_count :
  totalAnimals (applyChanges initialFarm) = 555 :=
by sorry

end NUMINAMATH_CALUDE_farm_animals_count_l3304_330423


namespace NUMINAMATH_CALUDE_mountain_paths_l3304_330484

/-- Given a mountain with paths from east and west sides, calculate the total number of ways to ascend and descend -/
theorem mountain_paths (east_paths west_paths : ℕ) : 
  east_paths = 3 → west_paths = 2 → (east_paths + west_paths) * (east_paths + west_paths) = 25 := by
  sorry

#check mountain_paths

end NUMINAMATH_CALUDE_mountain_paths_l3304_330484


namespace NUMINAMATH_CALUDE_book_price_reduction_l3304_330410

theorem book_price_reduction (original_price : ℝ) : 
  original_price = 20 → 
  (original_price * (1 - 0.25) * (1 - 0.40) = 9) := by
  sorry

end NUMINAMATH_CALUDE_book_price_reduction_l3304_330410


namespace NUMINAMATH_CALUDE_power_inequality_l3304_330429

theorem power_inequality (x y a : ℝ) (hx : 0 < x) (hy : x < y) (hy1 : y < 1) (ha : 0 < a) (ha1 : a < 1) :
  x^a < y^a := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l3304_330429


namespace NUMINAMATH_CALUDE_correct_systematic_sampling_l3304_330464

def total_missiles : ℕ := 50
def selected_missiles : ℕ := 5

def systematic_sampling (total : ℕ) (selected : ℕ) : ℕ := total / selected

def generate_sequence (start : ℕ) (interval : ℕ) (count : ℕ) : List ℕ :=
  List.range count |>.map (fun i => start + i * interval)

theorem correct_systematic_sampling :
  let interval := systematic_sampling total_missiles selected_missiles
  let sequence := generate_sequence 3 interval selected_missiles
  interval = 10 ∧ sequence = [3, 13, 23, 33, 43] := by sorry

end NUMINAMATH_CALUDE_correct_systematic_sampling_l3304_330464


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l3304_330424

theorem weight_loss_challenge (W : ℝ) (W_pos : W > 0) : 
  let weight_after_loss := W * 0.9
  let weight_with_clothes := weight_after_loss * 1.02
  let measured_loss_percentage := (W - weight_with_clothes) / W * 100
  measured_loss_percentage = 8.2 := by
sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l3304_330424


namespace NUMINAMATH_CALUDE_max_value_of_e_l3304_330465

theorem max_value_of_e (a b c d e : ℝ) 
  (sum_condition : a + b + c + d + e = 10)
  (product_condition : a*b + a*c + a*d + a*e + b*c + b*d + b*e + c*d + c*e + d*e = 20) :
  e ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_e_l3304_330465


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l3304_330447

theorem polynomial_multiplication (x : ℝ) : 
  (x^4 + 50*x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l3304_330447


namespace NUMINAMATH_CALUDE_right_triangle_area_l3304_330411

theorem right_triangle_area (a b : ℝ) (h1 : a + b = 4) (h2 : a^2 + b^2 = 14) :
  (1/2) * a * b = (1/2) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3304_330411


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3304_330449

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem intersection_complement_equality : A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3304_330449


namespace NUMINAMATH_CALUDE_percent_less_than_l3304_330432

theorem percent_less_than (x y z : ℝ) 
  (h1 : x = 1.3 * y) 
  (h2 : x = 0.78 * z) : 
  y = 0.6 * z := by
sorry

end NUMINAMATH_CALUDE_percent_less_than_l3304_330432


namespace NUMINAMATH_CALUDE_third_digit_is_one_l3304_330495

/-- A self-descriptive 7-digit number -/
structure SelfDescriptiveNumber where
  digits : Fin 7 → Fin 7
  sum_is_seven : (Finset.sum Finset.univ (λ i => digits i)) = 7
  first_digit : digits 0 = 3
  second_digit : digits 1 = 2
  fourth_digit : digits 3 = 1
  fifth_digit : digits 4 = 0

/-- The third digit of a self-descriptive number is 1 -/
theorem third_digit_is_one (n : SelfDescriptiveNumber) : n.digits 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_third_digit_is_one_l3304_330495


namespace NUMINAMATH_CALUDE_division_simplification_l3304_330440

theorem division_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  -6 * x^2 * y^3 / (2 * x^2 * y^2) = -3 * y := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l3304_330440


namespace NUMINAMATH_CALUDE_complement_of_A_l3304_330460

def U : Set Int := {-1, 0, 1, 2}

def A : Set Int := {x : Int | x^2 < 2}

theorem complement_of_A : (U \ A) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3304_330460


namespace NUMINAMATH_CALUDE_trapezoid_area_l3304_330497

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a trapezoid PQRS with diagonals intersecting at T -/
structure Trapezoid :=
  (P Q R S T : Point)

/-- Calculates the area of a triangle given three points -/
def triangleArea (A B C : Point) : ℝ := sorry

/-- Checks if two line segments are parallel -/
def isParallel (A B C D : Point) : Prop := sorry

theorem trapezoid_area (PQRS : Trapezoid) : 
  isParallel PQRS.P PQRS.Q PQRS.R PQRS.S →
  triangleArea PQRS.P PQRS.Q PQRS.T = 40 →
  triangleArea PQRS.P PQRS.R PQRS.T = 25 →
  triangleArea PQRS.P PQRS.Q PQRS.R + 
  triangleArea PQRS.P PQRS.R PQRS.S + 
  triangleArea PQRS.P PQRS.S PQRS.Q = 105.625 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3304_330497


namespace NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l3304_330414

theorem alcohol_percentage_after_dilution :
  let initial_volume : ℝ := 15
  let initial_alcohol_percentage : ℝ := 20
  let added_water : ℝ := 2
  let initial_alcohol_volume : ℝ := initial_volume * (initial_alcohol_percentage / 100)
  let new_total_volume : ℝ := initial_volume + added_water
  let new_alcohol_percentage : ℝ := (initial_alcohol_volume / new_total_volume) * 100
  ∀ ε > 0, |new_alcohol_percentage - 17.65| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l3304_330414


namespace NUMINAMATH_CALUDE_circle_area_problem_l3304_330403

/-- The area of the region outside a circle of radius 1.5 and inside two circles of radius 2
    that are internally tangent to the smaller circle at opposite ends of its diameter -/
theorem circle_area_problem : ∃ (area : ℝ),
  let r₁ : ℝ := 1.5 -- radius of smaller circle
  let r₂ : ℝ := 2   -- radius of larger circles
  area = (13/4 : ℝ) * Real.pi - 3 * Real.sqrt 1.75 ∧
  area = 2 * (
    -- Area of sector in larger circle
    (1/3 : ℝ) * Real.pi * r₂^2 -
    -- Area of triangle
    (1/2 : ℝ) * r₁ * Real.sqrt (r₂^2 - r₁^2) -
    -- Area of quarter of smaller circle
    (1/4 : ℝ) * Real.pi * r₁^2
  ) := by sorry


end NUMINAMATH_CALUDE_circle_area_problem_l3304_330403


namespace NUMINAMATH_CALUDE_wilson_family_seating_arrangements_l3304_330481

/-- The number of ways to seat a family with the given constraints -/
def seatingArrangements (numBoys numGirls : ℕ) : ℕ :=
  let numAdjacentBoys := 3
  let totalSeats := numBoys + numGirls
  let numRemainingBoys := numBoys - numAdjacentBoys
  let numEntities := numRemainingBoys + numGirls + 1  -- +1 for the block of 3 boys
  (numBoys.choose numAdjacentBoys) * (Nat.factorial numAdjacentBoys) *
  (Nat.factorial numEntities) * (Nat.factorial numRemainingBoys) *
  (Nat.factorial numGirls)

/-- Theorem stating that the number of seating arrangements for the Wilson family is 5760 -/
theorem wilson_family_seating_arrangements :
  seatingArrangements 5 2 = 5760 := by
  sorry

#eval seatingArrangements 5 2

end NUMINAMATH_CALUDE_wilson_family_seating_arrangements_l3304_330481


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_105_l3304_330444

theorem last_three_digits_of_7_to_105 : 7^105 ≡ 783 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_105_l3304_330444


namespace NUMINAMATH_CALUDE_math_olympiad_reform_l3304_330420

-- Define the probability of achieving a top-20 ranking in a single competition
def top20_prob : ℚ := 1/4

-- Define the maximum number of competitions
def max_competitions : ℕ := 5

-- Define the number of top-20 rankings needed to join the provincial team
def required_top20 : ℕ := 2

-- Define the function to calculate the probability of joining the provincial team
def prob_join_team : ℚ := sorry

-- Define the random variable ξ representing the number of competitions participated
def ξ : ℕ → ℚ
| 2 => 1/16
| 3 => 3/32
| 4 => 27/64
| 5 => 27/64
| _ => 0

-- Define the expected value of ξ
def expected_ξ : ℚ := sorry

-- Theorem statement
theorem math_olympiad_reform :
  (prob_join_team = 67/256) ∧ (expected_ξ = 356/256) := by sorry

end NUMINAMATH_CALUDE_math_olympiad_reform_l3304_330420


namespace NUMINAMATH_CALUDE_equation_holds_l3304_330445

theorem equation_holds (x : ℝ) (h : x = 12) : ((17.28 / x) / (3.6 * 0.2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l3304_330445


namespace NUMINAMATH_CALUDE_chord_length_from_arc_and_angle_l3304_330488

theorem chord_length_from_arc_and_angle (m : ℝ) (h : m > 0) :
  let arc_length := m
  let central_angle : ℝ := 120 * π / 180
  let radius := arc_length / central_angle
  let chord_length := 2 * radius * Real.sin (central_angle / 2)
  chord_length = (3 * Real.sqrt 3 / (4 * π)) * m :=
by sorry

end NUMINAMATH_CALUDE_chord_length_from_arc_and_angle_l3304_330488


namespace NUMINAMATH_CALUDE_sum_of_squares_with_means_l3304_330412

/-- Given three positive real numbers with specific arithmetic, geometric, and harmonic means, 
    prove that the sum of their squares equals 385.5 -/
theorem sum_of_squares_with_means (x y z : ℝ) 
    (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
    (h_arithmetic : (x + y + z) / 3 = 10)
    (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 7)
    (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 385.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_with_means_l3304_330412


namespace NUMINAMATH_CALUDE_cube_volume_problem_l3304_330457

theorem cube_volume_problem (a : ℝ) : 
  (a + 3) * (a - 2) * a - a^3 = 6 → a = 3 + Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l3304_330457


namespace NUMINAMATH_CALUDE_max_single_game_schedules_max_n_value_l3304_330463

/-- Represents a chess tournament between two teams -/
structure ChessTournament where
  team_size : ℕ
  total_games : ℕ
  games_played : ℕ

/-- Creates a chess tournament with the given parameters -/
def create_tournament (size : ℕ) : ChessTournament :=
  { team_size := size
  , total_games := size * size
  , games_played := 0 }

/-- Theorem stating the maximum number of ways to schedule a single game -/
theorem max_single_game_schedules (t : ChessTournament) (h1 : t.team_size = 15) :
  (t.total_games - t.games_played) ≤ 120 := by
  sorry

/-- Main theorem proving the maximum value of N -/
theorem max_n_value :
  ∃ (t : ChessTournament), t.team_size = 15 ∧ (t.total_games - t.games_played) = 120 := by
  sorry

end NUMINAMATH_CALUDE_max_single_game_schedules_max_n_value_l3304_330463


namespace NUMINAMATH_CALUDE_gem_bonus_percentage_l3304_330428

theorem gem_bonus_percentage (purchase : ℝ) (rate : ℝ) (final_gems : ℝ) : 
  purchase = 250 → 
  rate = 100 → 
  final_gems = 30000 → 
  (final_gems - purchase * rate) / (purchase * rate) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_gem_bonus_percentage_l3304_330428


namespace NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l3304_330496

theorem min_sum_with_reciprocal_constraint (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : 1 / a + 2 / b = 2) : 
  a + b ≥ (3 + 2 * Real.sqrt 2) / 2 ∧ 
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 1 / a₀ + 2 / b₀ = 2 ∧ a₀ + b₀ = (3 + 2 * Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l3304_330496


namespace NUMINAMATH_CALUDE_sin_product_equals_one_eighth_l3304_330454

theorem sin_product_equals_one_eighth :
  Real.sin (π / 14) * Real.sin (3 * π / 14) * Real.sin (5 * π / 14) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_eighth_l3304_330454


namespace NUMINAMATH_CALUDE_max_m_inequality_l3304_330451

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ m : ℝ, (2/a + 1/b ≥ m/(2*a + b)) → m ≤ 9) ∧ 
  (∃ m : ℝ, m = 9 ∧ 2/a + 1/b ≥ m/(2*a + b)) :=
sorry

end NUMINAMATH_CALUDE_max_m_inequality_l3304_330451


namespace NUMINAMATH_CALUDE_inverse_image_of_three_l3304_330418

-- Define the mapping f: A → B
def f (x : ℝ) : ℝ := 2 * x + 1

-- Theorem statement
theorem inverse_image_of_three (h : f 1 = 3) : ∃ x, f x = 3 ∧ x = 1 := by
  sorry


end NUMINAMATH_CALUDE_inverse_image_of_three_l3304_330418


namespace NUMINAMATH_CALUDE_simplify_fraction_l3304_330475

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : 
  (x^2 + 1) / (x - 1) - 2*x / (x - 1) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3304_330475


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3304_330436

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, x > 0 → x ≥ 1) ↔ (∃ x : ℝ, x > 0 ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3304_330436


namespace NUMINAMATH_CALUDE_dan_helmet_craters_l3304_330489

/-- The number of craters in helmets owned by Dan, Daniel, and Rin. -/
structure HelmetsWithCraters where
  dan : ℕ
  daniel : ℕ
  rin : ℕ

/-- The conditions of the helmet crater problem. -/
def helmet_crater_conditions (h : HelmetsWithCraters) : Prop :=
  h.dan = h.daniel + 10 ∧
  h.rin = h.dan + h.daniel + 15 ∧
  h.rin = 75

/-- The theorem stating that Dan's helmet has 35 craters given the conditions. -/
theorem dan_helmet_craters (h : HelmetsWithCraters) 
  (hc : helmet_crater_conditions h) : h.dan = 35 := by
  sorry

end NUMINAMATH_CALUDE_dan_helmet_craters_l3304_330489


namespace NUMINAMATH_CALUDE_quadratic_real_equal_roots_l3304_330482

/-- 
For a quadratic equation of the form 3x^2 + 6kx + 9 = 0, 
the roots are real and equal if and only if k = ± √3.
-/
theorem quadratic_real_equal_roots (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + 6 * k * x + 9 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 + 6 * k * y + 9 = 0 → y = x) ↔ 
  k = Real.sqrt 3 ∨ k = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_equal_roots_l3304_330482


namespace NUMINAMATH_CALUDE_round_trip_speed_l3304_330476

/-- Proves that given a round trip where the return journey takes twice as long as the outward journey,
    and the average speed of the entire trip is 32 miles per hour, the speed of the outward journey is 21⅓ miles per hour. -/
theorem round_trip_speed (d : ℝ) (v : ℝ) (h1 : v > 0) (h2 : d > 0) : 
  (2 * d) / (d / v + 2 * d / v) = 32 → v = 64 / 3 := by
  sorry

#eval (64 : ℚ) / 3  -- To show that 64/3 is indeed equal to 21⅓

end NUMINAMATH_CALUDE_round_trip_speed_l3304_330476


namespace NUMINAMATH_CALUDE_least_bamboo_sticks_l3304_330490

/-- Represents the number of bamboo sticks each panda takes initially -/
structure BambooDistribution where
  s1 : ℕ
  s2 : ℕ
  s3 : ℕ
  s4 : ℕ

/-- Represents the final number of bamboo sticks each panda has -/
structure FinalDistribution where
  p1 : ℕ
  p2 : ℕ
  p3 : ℕ
  p4 : ℕ

/-- Calculates the final distribution based on the initial distribution -/
def calculateFinalDistribution (initial : BambooDistribution) : FinalDistribution :=
  { p1 := (2 * initial.s1) / 3 + initial.s2 / 2 + initial.s3 / 6 + (8 * initial.s4) / 9
  , p2 := (2 * initial.s1) / 3 + initial.s2 / 2 + initial.s3 / 6 + initial.s4 / 9
  , p3 := (2 * initial.s1) / 3 + initial.s2 / 2 + initial.s3 / 6 + initial.s4 / 9
  , p4 := (2 * initial.s1) / 3 + initial.s2 / 2 + initial.s3 / 6 + initial.s4 / 9
  }

/-- Checks if the final distribution satisfies the 4:3:2:1 ratio -/
def isValidRatio (final : FinalDistribution) : Prop :=
  4 * final.p4 = final.p1 ∧
  3 * final.p4 = final.p2 ∧
  2 * final.p4 = final.p3

/-- The main theorem stating the least possible total number of bamboo sticks -/
theorem least_bamboo_sticks :
  ∃ (initial : BambooDistribution),
    let final := calculateFinalDistribution initial
    isValidRatio final ∧
    initial.s1 + initial.s2 + initial.s3 + initial.s4 = 93 ∧
    ∀ (other : BambooDistribution),
      let otherFinal := calculateFinalDistribution other
      isValidRatio otherFinal →
      other.s1 + other.s2 + other.s3 + other.s4 ≥ 93 :=
by sorry


end NUMINAMATH_CALUDE_least_bamboo_sticks_l3304_330490


namespace NUMINAMATH_CALUDE_problem_solution_l3304_330459

theorem problem_solution (a b : ℝ) (h : (a - 1)^2 + |b + 2| = 0) :
  2 * (5 * a^2 - 7 * a * b + 9 * b^2) - 3 * (14 * a^2 - 2 * a * b + 3 * b^2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3304_330459


namespace NUMINAMATH_CALUDE_ceiling_of_3_7_l3304_330438

theorem ceiling_of_3_7 : ⌈(3.7 : ℝ)⌉ = 4 := by sorry

end NUMINAMATH_CALUDE_ceiling_of_3_7_l3304_330438


namespace NUMINAMATH_CALUDE_at_most_12_moves_for_9_l3304_330456

/-- A move is defined as reversing the order of any block of consecutive increasing or decreasing numbers -/
def is_valid_move (perm : List Nat) (start finish : Nat) : Prop :=
  start < finish ∧ finish ≤ perm.length ∧
  (∀ i, start < i ∧ i < finish → perm[i-1]! < perm[i]! ∨ perm[i-1]! > perm[i]!)

/-- The function that counts the minimum number of moves needed to sort a permutation -/
def min_moves_to_sort (perm : List Nat) : Nat :=
  sorry

/-- Theorem stating that at most 12 moves are needed to sort any permutation of numbers from 1 to 9 -/
theorem at_most_12_moves_for_9 :
  ∀ perm : List Nat, perm.Nodup → perm.length = 9 → (∀ n, n ∈ perm ↔ 1 ≤ n ∧ n ≤ 9) →
  min_moves_to_sort perm ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_at_most_12_moves_for_9_l3304_330456


namespace NUMINAMATH_CALUDE_glenn_total_expenditure_l3304_330483

/-- Represents the cost of movie tickets and concessions -/
structure MovieCosts where
  monday_ticket : ℕ
  wednesday_ticket : ℕ
  saturday_ticket : ℕ
  concession : ℕ

/-- Represents discount percentages -/
structure Discounts where
  wednesday : ℕ
  group : ℕ

/-- Represents the number of people in Glenn's group for each day -/
structure GroupSize where
  wednesday : ℕ
  saturday : ℕ

/-- Calculates the total cost of Glenn's movie outings -/
def calculate_total_cost (costs : MovieCosts) (discounts : Discounts) (group : GroupSize) : ℕ :=
  let wednesday_cost := costs.wednesday_ticket * (100 - discounts.wednesday) / 100 * group.wednesday
  let saturday_cost := costs.saturday_ticket * group.saturday + costs.concession
  wednesday_cost + saturday_cost

/-- Theorem stating that Glenn's total expenditure is $93 -/
theorem glenn_total_expenditure (costs : MovieCosts) (discounts : Discounts) (group : GroupSize) :
  costs.monday_ticket = 5 →
  costs.wednesday_ticket = 2 * costs.monday_ticket →
  costs.saturday_ticket = 5 * costs.monday_ticket →
  costs.concession = 7 →
  discounts.wednesday = 10 →
  discounts.group = 20 →
  group.wednesday = 4 →
  group.saturday = 2 →
  calculate_total_cost costs discounts group = 93 := by
  sorry


end NUMINAMATH_CALUDE_glenn_total_expenditure_l3304_330483


namespace NUMINAMATH_CALUDE_june_election_win_l3304_330486

theorem june_election_win (total_students : ℕ) (boy_percentage : ℚ) 
  (june_boy_vote_percentage : ℚ) (june_girl_vote_percentage : ℚ) :
  total_students = 200 →
  boy_percentage = 60 / 100 →
  june_boy_vote_percentage = 675 / 1000 →
  june_girl_vote_percentage = 1 / 4 →
  ∃ (june_total_vote_percentage : ℚ), 
    june_total_vote_percentage = 505 / 1000 ∧ 
    june_total_vote_percentage > 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_june_election_win_l3304_330486


namespace NUMINAMATH_CALUDE_two_true_propositions_l3304_330474

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x = 3 → x^2 = 9

-- Define the converse proposition
def converse_prop (x : ℝ) : Prop := x^2 = 9 → x = 3

-- Define the inverse proposition
def inverse_prop (x : ℝ) : Prop := x ≠ 3 → x^2 ≠ 9

-- Define the contrapositive proposition
def contrapositive_prop (x : ℝ) : Prop := x^2 ≠ 9 → x ≠ 3

-- Define the negation proposition
def negation_prop (x : ℝ) : Prop := ¬(x = 3 → x^2 = 9)

-- Theorem statement
theorem two_true_propositions :
  ∃ (A B : (ℝ → Prop)), 
    (A = original_prop ∨ A = converse_prop ∨ A = inverse_prop ∨ A = contrapositive_prop ∨ A = negation_prop) ∧
    (B = original_prop ∨ B = converse_prop ∨ B = inverse_prop ∨ B = contrapositive_prop ∨ B = negation_prop) ∧
    A ≠ B ∧
    (∀ x, A x) ∧ 
    (∀ x, B x) ∧
    (∀ C, (C = original_prop ∨ C = converse_prop ∨ C = inverse_prop ∨ C = contrapositive_prop ∨ C = negation_prop) →
      C ≠ A → C ≠ B → ∃ x, ¬(C x)) :=
by sorry

end NUMINAMATH_CALUDE_two_true_propositions_l3304_330474


namespace NUMINAMATH_CALUDE_sum_of_squares_l3304_330455

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 6*y = -17)
  (eq2 : y^2 + 4*z = 1)
  (eq3 : z^2 + 2*x = 2) :
  x^2 + y^2 + z^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3304_330455


namespace NUMINAMATH_CALUDE_max_dimes_in_piggy_banks_l3304_330450

/-- Represents the number of coins a piggy bank can hold -/
def PiggyBankCapacity : ℕ := 100

/-- Represents the total number of coins in two piggy banks -/
def TotalCoins : ℕ := 200

/-- Represents the total value of coins in cents -/
def TotalValue : ℕ := 1200

/-- Represents the value of a dime in cents -/
def DimeValue : ℕ := 10

/-- Represents the value of a penny in cents -/
def PennyValue : ℕ := 1

/-- Theorem stating the maximum number of dimes that can be held in the piggy banks -/
theorem max_dimes_in_piggy_banks :
  ∃ (d : ℕ), d ≤ 111 ∧
  d * DimeValue + (TotalCoins - d) * PennyValue = TotalValue ∧
  (∀ (x : ℕ), x > d →
    x * DimeValue + (TotalCoins - x) * PennyValue ≠ TotalValue) :=
by sorry

#check max_dimes_in_piggy_banks

end NUMINAMATH_CALUDE_max_dimes_in_piggy_banks_l3304_330450


namespace NUMINAMATH_CALUDE_student_average_age_l3304_330417

theorem student_average_age (n : ℕ) (teacher_age : ℕ) (avg_increase : ℚ) :
  n = 19 →
  teacher_age = 40 →
  avg_increase = 1 →
  ∃ (student_avg : ℚ),
    (n : ℚ) * student_avg + teacher_age = (n + 1 : ℚ) * (student_avg + avg_increase) ∧
    student_avg = 20 :=
by sorry

end NUMINAMATH_CALUDE_student_average_age_l3304_330417


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l3304_330467

theorem initial_number_of_persons
  (average_weight_increase : ℝ)
  (weight_of_leaving_person : ℝ)
  (weight_of_new_person : ℝ)
  (h1 : average_weight_increase = 5.5)
  (h2 : weight_of_leaving_person = 68)
  (h3 : weight_of_new_person = 95.5) :
  ∃ N : ℕ, N = 5 ∧ 
  N * average_weight_increase = weight_of_new_person - weight_of_leaving_person :=
by sorry

end NUMINAMATH_CALUDE_initial_number_of_persons_l3304_330467


namespace NUMINAMATH_CALUDE_students_in_grades_2_and_3_l3304_330406

theorem students_in_grades_2_and_3 (boys_grade_2 girls_grade_2 : ℕ) 
  (h1 : boys_grade_2 = 20)
  (h2 : girls_grade_2 = 11)
  (h3 : ∀ x, x = boys_grade_2 + girls_grade_2 → 2 * x = students_grade_3) :
  boys_grade_2 + girls_grade_2 + students_grade_3 = 93 :=
by
  sorry

#check students_in_grades_2_and_3

end NUMINAMATH_CALUDE_students_in_grades_2_and_3_l3304_330406


namespace NUMINAMATH_CALUDE_c_range_theorem_l3304_330491

/-- Proposition p: c^2 < c -/
def p (c : ℝ) : Prop := c^2 < c

/-- Proposition q: ∀x∈ℝ, x^2 + 4cx + 1 > 0 -/
def q (c : ℝ) : Prop := ∀ x : ℝ, x^2 + 4*c*x + 1 > 0

/-- The range of c given the conditions -/
def c_range (c : ℝ) : Prop := c ∈ Set.Ioc (-1/2) 0 ∪ Set.Icc (1/2) 1

theorem c_range_theorem (c : ℝ) :
  (p c ∨ q c) ∧ ¬(p c ∧ q c) → c_range c :=
by sorry

end NUMINAMATH_CALUDE_c_range_theorem_l3304_330491


namespace NUMINAMATH_CALUDE_grocery_shopping_problem_l3304_330409

theorem grocery_shopping_problem (initial_budget : ℚ) (bread_cost : ℚ) (candy_cost : ℚ) 
  (h1 : initial_budget = 32)
  (h2 : bread_cost = 3)
  (h3 : candy_cost = 2) : 
  let remaining_after_bread_candy := initial_budget - (bread_cost + candy_cost)
  let turkey_cost := (1 / 3) * remaining_after_bread_candy
  initial_budget - (bread_cost + candy_cost + turkey_cost) = 18 := by
sorry

end NUMINAMATH_CALUDE_grocery_shopping_problem_l3304_330409


namespace NUMINAMATH_CALUDE_largest_package_size_l3304_330401

theorem largest_package_size (ming_markers : ℕ) (catherine_markers : ℕ)
  (h1 : ming_markers = 72)
  (h2 : catherine_markers = 48) :
  ∃ (package_size : ℕ),
    package_size ∣ ming_markers ∧
    package_size ∣ catherine_markers ∧
    ∀ (n : ℕ), n ∣ ming_markers → n ∣ catherine_markers → n ≤ package_size :=
by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l3304_330401


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l3304_330478

theorem sqrt_product_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (30 * p) * Real.sqrt (5 * p) * Real.sqrt (6 * p) = 30 * p * Real.sqrt p :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l3304_330478


namespace NUMINAMATH_CALUDE_cosine_one_third_irrational_l3304_330492

theorem cosine_one_third_irrational (a : ℝ) (h : Real.cos (π * a) = (1 : ℝ) / 3) : 
  Irrational a := by sorry

end NUMINAMATH_CALUDE_cosine_one_third_irrational_l3304_330492


namespace NUMINAMATH_CALUDE_correct_statements_l3304_330480

theorem correct_statements :
  (∀ x : ℝ, x^2 > 0 → x ≠ 0) ∧
  (∀ x : ℝ, x > 1 → x^2 > x) :=
by sorry

end NUMINAMATH_CALUDE_correct_statements_l3304_330480


namespace NUMINAMATH_CALUDE_pebble_distribution_correct_l3304_330426

/-- The number of friends who received pebbles from Janice --/
def num_friends : ℕ := 17

/-- The total weight of pebbles in grams --/
def total_weight : ℕ := 36000

/-- The weight of a small pebble in grams --/
def small_pebble_weight : ℕ := 200

/-- The weight of a large pebble in grams --/
def large_pebble_weight : ℕ := 300

/-- The number of small pebbles given to each friend --/
def small_pebbles_per_friend : ℕ := 3

/-- The number of large pebbles given to each friend --/
def large_pebbles_per_friend : ℕ := 5

/-- Theorem stating that the number of friends who received pebbles is correct --/
theorem pebble_distribution_correct : 
  num_friends * (small_pebbles_per_friend * small_pebble_weight + 
                 large_pebbles_per_friend * large_pebble_weight) ≤ total_weight ∧
  (num_friends + 1) * (small_pebbles_per_friend * small_pebble_weight + 
                       large_pebbles_per_friend * large_pebble_weight) > total_weight :=
by sorry

end NUMINAMATH_CALUDE_pebble_distribution_correct_l3304_330426


namespace NUMINAMATH_CALUDE_unique_n_for_prime_ones_and_seven_l3304_330433

def has_n_minus_one_ones_and_one_seven (n : ℕ) (x : ℕ) : Prop :=
  ∃ k : ℕ, k < n ∧ x = (10^n - 1) / 9 + 6 * 10^k

theorem unique_n_for_prime_ones_and_seven :
  ∃! n : ℕ, n > 0 ∧ ∀ x : ℕ, has_n_minus_one_ones_and_one_seven n x → Nat.Prime x :=
by sorry

end NUMINAMATH_CALUDE_unique_n_for_prime_ones_and_seven_l3304_330433


namespace NUMINAMATH_CALUDE_union_of_given_sets_l3304_330407

theorem union_of_given_sets :
  let A : Set Int := {-3, 1, 2}
  let B : Set Int := {0, 1, 2, 3}
  A ∪ B = {-3, 0, 1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_union_of_given_sets_l3304_330407


namespace NUMINAMATH_CALUDE_technician_salary_l3304_330408

/-- The average salary of technicians in a workshop --/
theorem technician_salary (total_workers : ℝ) (total_avg_salary : ℝ) 
  (num_technicians : ℝ) (non_tech_avg_salary : ℝ) :
  total_workers = 21.11111111111111 →
  total_avg_salary = 1000 →
  num_technicians = 10 →
  non_tech_avg_salary = 820 →
  (total_workers * total_avg_salary - (total_workers - num_technicians) * non_tech_avg_salary) 
    / num_technicians = 1200 := by
  sorry

end NUMINAMATH_CALUDE_technician_salary_l3304_330408


namespace NUMINAMATH_CALUDE_snow_leopard_arrangement_l3304_330494

theorem snow_leopard_arrangement (n : ℕ) (h : n = 9) : 
  (2 : ℕ) * (Nat.factorial (n - 3)) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_snow_leopard_arrangement_l3304_330494


namespace NUMINAMATH_CALUDE_solution_system1_solution_system2_l3304_330493

-- Define the systems of equations
def system1 (x y : ℝ) : Prop := (3 * x + 2 * y = 5) ∧ (y = 2 * x - 8)
def system2 (x y : ℝ) : Prop := (2 * x - y = 10) ∧ (2 * x + 3 * y = 2)

-- Theorem for System 1
theorem solution_system1 : ∃ x y : ℝ, system1 x y ∧ x = 3 ∧ y = -2 := by
  sorry

-- Theorem for System 2
theorem solution_system2 : ∃ x y : ℝ, system2 x y ∧ x = 4 ∧ y = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_system1_solution_system2_l3304_330493
