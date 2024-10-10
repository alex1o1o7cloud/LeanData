import Mathlib

namespace unknown_towel_rate_unknown_towel_rate_solution_l4055_405508

/-- Proves that the unknown rate of two towels is 300, given the conditions of the problem -/
theorem unknown_towel_rate : ℕ → Prop :=
  fun (x : ℕ) ↦
    let total_towels : ℕ := 3 + 5 + 2
    let known_cost : ℕ := 3 * 100 + 5 * 150
    let total_cost : ℕ := 165 * total_towels
    (known_cost + 2 * x = total_cost) → (x = 300)

/-- Solution to the unknown_towel_rate theorem -/
theorem unknown_towel_rate_solution : unknown_towel_rate 300 := by
  sorry

end unknown_towel_rate_unknown_towel_rate_solution_l4055_405508


namespace problem_stack_total_l4055_405576

/-- Represents a stack of logs -/
structure LogStack where
  topRow : ℕ
  bottomRow : ℕ

/-- Calculates the total number of logs in a stack -/
def totalLogs (stack : LogStack) : ℕ :=
  let n := stack.bottomRow - stack.topRow + 1
  n * (stack.topRow + stack.bottomRow) / 2

/-- The specific log stack described in the problem -/
def problemStack : LogStack := { topRow := 5, bottomRow := 15 }

/-- Theorem stating that the total number of logs in the problem stack is 110 -/
theorem problem_stack_total : totalLogs problemStack = 110 := by
  sorry

end problem_stack_total_l4055_405576


namespace arithmetic_sequence_sum_l4055_405587

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of five consecutive terms equals 450 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 3 + a 4 + a 5 + a 6 + a 7 = 450

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → SumCondition a → a 2 + a 8 = 180 := by
  sorry

end arithmetic_sequence_sum_l4055_405587


namespace christinas_walking_speed_l4055_405582

/-- Prove that Christina's walking speed is 8 feet per second given the initial conditions and the total distance traveled by Lindy. -/
theorem christinas_walking_speed 
  (initial_distance : ℝ) 
  (jack_speed : ℝ) 
  (lindy_speed : ℝ) 
  (lindy_total_distance : ℝ) 
  (h1 : initial_distance = 150)
  (h2 : jack_speed = 7)
  (h3 : lindy_speed = 10)
  (h4 : lindy_total_distance = 100) :
  ∃ christina_speed : ℝ, christina_speed = 8 ∧ 
    (lindy_total_distance / lindy_speed) * (jack_speed + christina_speed) = initial_distance :=
by sorry

end christinas_walking_speed_l4055_405582


namespace correct_calculation_l4055_405561

theorem correct_calculation (x : ℝ) : 2 * (x + 6) = 28 → 6 * x = 48 := by
  sorry

end correct_calculation_l4055_405561


namespace relay_race_total_time_l4055_405583

/-- The total time for a relay race with four athletes -/
def relay_race_time (athlete1_time athlete2_time athlete3_time athlete4_time : ℕ) : ℕ :=
  athlete1_time + athlete2_time + athlete3_time + athlete4_time

/-- Theorem stating the total time for the relay race is 200 seconds -/
theorem relay_race_total_time : 
  ∀ (athlete1_time : ℕ),
    athlete1_time = 55 →
    ∀ (athlete2_time : ℕ),
      athlete2_time = athlete1_time + 10 →
      ∀ (athlete3_time : ℕ),
        athlete3_time = athlete2_time - 15 →
        ∀ (athlete4_time : ℕ),
          athlete4_time = athlete1_time - 25 →
          relay_race_time athlete1_time athlete2_time athlete3_time athlete4_time = 200 :=
by
  sorry

end relay_race_total_time_l4055_405583


namespace f_inverse_property_implies_c_plus_d_eq_nine_halves_l4055_405513

-- Define the piecewise function f
noncomputable def f (c d : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then c * x + d else 10 - 2 * x

-- State the theorem
theorem f_inverse_property_implies_c_plus_d_eq_nine_halves
  (c d : ℝ)
  (h : ∀ x, f c d (f c d x) = x) :
  c + d = 9/2 := by
sorry

end f_inverse_property_implies_c_plus_d_eq_nine_halves_l4055_405513


namespace constant_value_l4055_405525

theorem constant_value (x y z : ℝ) : 
  ∃ (c : ℝ), ∀ (x y z : ℝ), 
    ((x - y)^3 + (y - z)^3 + (z - x)^3) / (c * (x - y) * (y - z) * (z - x)) = 0.2 → c = 15 := by
  sorry

end constant_value_l4055_405525


namespace equation_solutions_l4055_405523

theorem equation_solutions :
  (∀ x : ℝ, (x - 1) * (x + 3) = x - 1 ↔ x = 1 ∨ x = -2) ∧
  (∀ x : ℝ, 2 * x^2 - 6 * x = -3 ↔ x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2) :=
by sorry

end equation_solutions_l4055_405523


namespace emily_quiz_score_l4055_405570

/-- Emily's quiz scores -/
def emily_scores : List ℕ := [85, 88, 90, 94, 96, 92]

/-- The required arithmetic mean -/
def required_mean : ℕ := 91

/-- The number of quizzes including the new one -/
def total_quizzes : ℕ := 7

/-- The score Emily needs on her seventh quiz -/
def seventh_score : ℕ := 92

theorem emily_quiz_score :
  (emily_scores.sum + seventh_score) / total_quizzes = required_mean := by
  sorry

end emily_quiz_score_l4055_405570


namespace vector_difference_magnitude_l4055_405580

theorem vector_difference_magnitude (a b : ℝ × ℝ) :
  ‖a‖ = 1 →
  ‖b‖ = 3 →
  a + b = (Real.sqrt 3, 1) →
  ‖a - b‖ = 4 := by
  sorry

end vector_difference_magnitude_l4055_405580


namespace smallest_cube_root_with_small_fraction_l4055_405555

theorem smallest_cube_root_with_small_fraction (m n : ℕ) (s : ℝ) : 
  (∀ k < n, ¬ ∃ (t : ℝ) (l : ℕ), t > 0 ∧ t < 1/500 ∧ l^(1/3 : ℝ) = k + t) →
  s > 0 → 
  s < 1/500 → 
  m^(1/3 : ℝ) = n + s → 
  n = 13 := by
sorry

end smallest_cube_root_with_small_fraction_l4055_405555


namespace oldest_child_age_l4055_405586

/-- Represents the ages of 7 children -/
def ChildrenAges := Fin 7 → ℕ

/-- The property that each child has a different age -/
def AllDifferent (ages : ChildrenAges) : Prop :=
  ∀ i j : Fin 7, i ≠ j → ages i ≠ ages j

/-- The property that the difference in age between consecutive children is 1 year -/
def ConsecutiveDifference (ages : ChildrenAges) : Prop :=
  ∀ i : Fin 6, ages (Fin.succ i) = ages i + 1

/-- The average age of the children is 8 years -/
def AverageAge (ages : ChildrenAges) : Prop :=
  (ages 0 + ages 1 + ages 2 + ages 3 + ages 4 + ages 5 + ages 6) / 7 = 8

theorem oldest_child_age
  (ages : ChildrenAges)
  (h_diff : AllDifferent ages)
  (h_cons : ConsecutiveDifference ages)
  (h_avg : AverageAge ages) :
  ages 6 = 11 := by
  sorry

end oldest_child_age_l4055_405586


namespace right_triangle_sides_l4055_405530

/-- A right triangle with perimeter 30 and height to hypotenuse 6 has sides 10, 7.5, and 12.5 -/
theorem right_triangle_sides (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) : 
  a^2 + b^2 = c^2 →  -- right triangle condition
  a + b + c = 30 →   -- perimeter condition
  a * b = 6 * c →    -- height to hypotenuse condition
  ((a = 10 ∧ b = 7.5 ∧ c = 12.5) ∨ (a = 7.5 ∧ b = 10 ∧ c = 12.5)) := by
  sorry

#check right_triangle_sides

end right_triangle_sides_l4055_405530


namespace discount_clinic_savings_l4055_405519

theorem discount_clinic_savings (normal_cost : ℚ) (discount_percentage : ℚ) (discount_visits : ℕ) : 
  normal_cost = 200 →
  discount_percentage = 70 →
  discount_visits = 2 →
  normal_cost - (discount_visits * (normal_cost * (1 - discount_percentage / 100))) = 80 := by
sorry

end discount_clinic_savings_l4055_405519


namespace min_distance_curve_line_l4055_405591

/-- The minimum distance between a curve and a line -/
theorem min_distance_curve_line (a m n : ℝ) (h1 : a > 0) :
  let b := -1/2 * a^2 + 3 * Real.log a
  let line := {p : ℝ × ℝ | p.2 = 2 * p.1 + 1/2}
  let Q := (m, n)
  Q ∈ line →
  ∃ (min_dist : ℝ), min_dist = 9/5 ∧
    ∀ (p : ℝ × ℝ), p ∈ line → (a - p.1)^2 + (b - p.2)^2 ≥ min_dist :=
by sorry

end min_distance_curve_line_l4055_405591


namespace correct_calculation_l4055_405571

theorem correct_calculation (a : ℝ) : 2 * a * (1 - a) = 2 * a - 2 * a^2 := by
  sorry

end correct_calculation_l4055_405571


namespace dave_lisa_slices_l4055_405551

/-- Represents the number of slices in a pizza -/
structure Pizza where
  small : ℕ
  large : ℕ

/-- Represents the number of pizzas purchased -/
structure PizzaOrder where
  small : ℕ
  large : ℕ

/-- Represents the number of slices eaten by each person -/
structure SlicesEaten where
  george : ℕ
  bob : ℕ
  susie : ℕ
  bill : ℕ
  fred : ℕ
  mark : ℕ
  ann : ℕ
  kelly : ℕ

def pizza_sizes : Pizza := ⟨4, 8⟩
def george_order : PizzaOrder := ⟨4, 3⟩
def slices_eaten : SlicesEaten := ⟨3, 4, 2, 3, 3, 3, 2, 4⟩

def total_slices (p : Pizza) (o : PizzaOrder) : ℕ :=
  p.small * o.small + p.large * o.large

def total_eaten (s : SlicesEaten) : ℕ :=
  s.george + s.bob + s.susie + s.bill + s.fred + s.mark + s.ann + s.kelly

theorem dave_lisa_slices :
  (total_slices pizza_sizes george_order - total_eaten slices_eaten) / 2 = 8 := by
  sorry

end dave_lisa_slices_l4055_405551


namespace decrypt_ciphertext_l4055_405544

/-- Represents the encryption rule --/
def encrypt (a b c d : ℤ) : ℤ × ℤ × ℤ × ℤ :=
  (a + 2*b, 2*b + c, 2*c + 3*d, 4*d)

/-- Represents the given ciphertext --/
def ciphertext : ℤ × ℤ × ℤ × ℤ := (14, 9, 23, 28)

/-- Theorem stating that the plaintext (6, 4, 1, 7) corresponds to the given ciphertext --/
theorem decrypt_ciphertext :
  encrypt 6 4 1 7 = ciphertext := by sorry

end decrypt_ciphertext_l4055_405544


namespace seventeen_in_both_competitions_l4055_405536

/-- The number of students who participated in both math and physics competitions -/
def students_in_both_competitions (total : ℕ) (math : ℕ) (physics : ℕ) (none : ℕ) : ℕ :=
  math + physics + none - total

/-- Theorem stating that 17 students participated in both competitions -/
theorem seventeen_in_both_competitions :
  students_in_both_competitions 37 30 20 4 = 17 := by
  sorry

end seventeen_in_both_competitions_l4055_405536


namespace dads_strawberries_weight_l4055_405509

/-- The weight of Marco's dad's strawberries -/
def dads_strawberries (marcos_weight total_weight : ℕ) : ℕ :=
  total_weight - marcos_weight

/-- Theorem: Marco's dad's strawberries weigh 22 pounds -/
theorem dads_strawberries_weight :
  dads_strawberries 15 37 = 22 := by
  sorry

end dads_strawberries_weight_l4055_405509


namespace charitable_gentleman_proof_l4055_405547

def charitable_donation (initial : ℕ) : Prop :=
  let after_first := initial - (initial / 2 + 1)
  let after_second := after_first - (after_first / 2 + 2)
  let after_third := after_second - (after_second / 2 + 3)
  after_third = 1

theorem charitable_gentleman_proof :
  ∃ (initial : ℕ), charitable_donation initial ∧ initial = 42 := by
  sorry

end charitable_gentleman_proof_l4055_405547


namespace nicky_running_time_l4055_405592

-- Define the race parameters
def race_distance : ℝ := 400
def head_start : ℝ := 12
def cristina_speed : ℝ := 5
def nicky_speed : ℝ := 3

-- Define the theorem
theorem nicky_running_time (t : ℝ) : 
  t * cristina_speed = head_start * nicky_speed + (t + head_start) * nicky_speed → 
  t + head_start = 48 :=
by sorry

end nicky_running_time_l4055_405592


namespace smallest_s_for_E_l4055_405577

/-- Definition of the function E --/
def E (a b c : ℕ) : ℕ := a * b^c

/-- The smallest positive integer s that satisfies E(s, s, 4) = 2401 is 7 --/
theorem smallest_s_for_E : (∃ s : ℕ, s > 0 ∧ E s s 4 = 2401 ∧ ∀ t : ℕ, t > 0 → E t t 4 = 2401 → s ≤ t) ∧ 
                           (∃ s : ℕ, s > 0 ∧ E s s 4 = 2401 ∧ s = 7) := by
  sorry

end smallest_s_for_E_l4055_405577


namespace quadratic_equation_two_distinct_roots_l4055_405550

theorem quadratic_equation_two_distinct_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (2 * x₁^2 - 6 * x₁ = 7) ∧ (2 * x₂^2 - 6 * x₂ = 7) := by
  sorry

end quadratic_equation_two_distinct_roots_l4055_405550


namespace quadratic_polynomial_theorem_l4055_405594

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on the graph of a quadratic polynomial -/
def pointLiesOnPolynomial (p : Point) (q : QuadraticPolynomial) : Prop :=
  p.y = q.a * p.x^2 + q.b * p.x + q.c

/-- The main theorem -/
theorem quadratic_polynomial_theorem 
  (points : Finset Point) 
  (h_count : points.card = 100)
  (h_four_points : ∀ (p₁ p₂ p₃ p₄ : Point), p₁ ∈ points → p₂ ∈ points → p₃ ∈ points → p₄ ∈ points →
    p₁ ≠ p₂ → p₁ ≠ p₃ → p₁ ≠ p₄ → p₂ ≠ p₃ → p₂ ≠ p₄ → p₃ ≠ p₄ →
    ∃ (q : QuadraticPolynomial), pointLiesOnPolynomial p₁ q ∧ pointLiesOnPolynomial p₂ q ∧
      pointLiesOnPolynomial p₃ q ∧ pointLiesOnPolynomial p₄ q) :
  ∃ (q : QuadraticPolynomial), ∀ (p : Point), p ∈ points → pointLiesOnPolynomial p q :=
by sorry

end quadratic_polynomial_theorem_l4055_405594


namespace simplify_square_roots_l4055_405581

theorem simplify_square_roots : 
  (Real.sqrt 800 / Real.sqrt 200) * ((Real.sqrt 180 / Real.sqrt 72) - (Real.sqrt 224 / Real.sqrt 56)) = Real.sqrt 10 - 4 := by
  sorry

end simplify_square_roots_l4055_405581


namespace regular_dodecahedron_vertex_count_l4055_405510

/-- A regular dodecahedron has 20 vertices. -/
def regular_dodecahedron_vertices : ℕ := 20

/-- The number of vertices in a regular dodecahedron is 20. -/
theorem regular_dodecahedron_vertex_count : 
  regular_dodecahedron_vertices = 20 := by sorry

end regular_dodecahedron_vertex_count_l4055_405510


namespace cone_surface_area_and_volume_l4055_405573

/-- Represents a cone with given height and sector angle -/
structure Cone where
  height : ℝ
  sectorAngle : ℝ

/-- Calculates the surface area of a cone -/
def surfaceArea (c : Cone) : ℝ := sorry

/-- Calculates the volume of a cone -/
def volume (c : Cone) : ℝ := sorry

/-- Theorem stating the surface area and volume of a specific cone -/
theorem cone_surface_area_and_volume :
  let c : Cone := { height := 12, sectorAngle := 100.8 * π / 180 }
  surfaceArea c = 56 * π ∧ volume c = 49 * π := by sorry

end cone_surface_area_and_volume_l4055_405573


namespace three_numbers_average_l4055_405501

theorem three_numbers_average : 
  ∀ (x y z : ℝ), 
    x = 18 ∧ 
    y = 4 * x ∧ 
    z = 2 * y → 
    (x + y + z) / 3 = 78 := by
  sorry

end three_numbers_average_l4055_405501


namespace complex_product_real_l4055_405565

theorem complex_product_real (x : ℝ) : 
  let z₁ : ℂ := 1 + Complex.I
  let z₂ : ℂ := 2 + x * Complex.I
  (z₁ * z₂).im = 0 → x = -2 := by
sorry

end complex_product_real_l4055_405565


namespace cube_root_to_square_l4055_405584

theorem cube_root_to_square (y : ℝ) : 
  (y + 5) ^ (1/3 : ℝ) = 3 → (y + 5)^2 = 729 := by
  sorry

end cube_root_to_square_l4055_405584


namespace total_gift_wrapping_combinations_l4055_405527

/-- The number of different gift wrapping combinations -/
def gift_wrapping_combinations (wrapping_paper : ℕ) (ribbon : ℕ) (gift_card : ℕ) (gift_tag : ℕ) : ℕ :=
  wrapping_paper * ribbon * gift_card * gift_tag

/-- Theorem stating that the total number of gift wrapping combinations is 600 -/
theorem total_gift_wrapping_combinations :
  gift_wrapping_combinations 10 5 6 2 = 600 := by
  sorry

end total_gift_wrapping_combinations_l4055_405527


namespace offer_price_per_year_is_half_l4055_405535

/-- Represents a magazine subscription offer -/
structure MagazineOffer where
  regularYearlyFee : ℕ
  offerYears : ℕ
  offerPrice : ℕ
  issuesPerYear : ℕ

/-- The Parents magazine offer -/
def parentsOffer : MagazineOffer :=
  { regularYearlyFee := 12
  , offerYears := 2
  , offerPrice := 12
  , issuesPerYear := 12
  }

/-- Theorem stating that the offer price per year is half of the regular price per year -/
theorem offer_price_per_year_is_half (o : MagazineOffer) 
    (h1 : o.offerYears = 2)
    (h2 : o.offerPrice = o.regularYearlyFee) :
    o.offerPrice / o.offerYears = o.regularYearlyFee / 2 := by
  sorry

#check offer_price_per_year_is_half parentsOffer

end offer_price_per_year_is_half_l4055_405535


namespace asha_win_probability_l4055_405574

theorem asha_win_probability (lose_prob : ℚ) (win_prob : ℚ) : 
  lose_prob = 7/12 → win_prob + lose_prob = 1 → win_prob = 5/12 := by
  sorry

end asha_win_probability_l4055_405574


namespace log_inequality_l4055_405516

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + 2*x) < 2*x := by
  sorry

end log_inequality_l4055_405516


namespace semicircle_in_square_l4055_405537

theorem semicircle_in_square (d m n : ℝ) : 
  d > 0 →                           -- d is positive (diameter)
  8 > 0 →                           -- square side length is positive
  d ≤ 8 →                           -- semicircle fits in square
  d ≤ m - Real.sqrt n →             -- maximum value of d
  m - Real.sqrt n ≤ 8 →             -- maximum value fits in square
  (∀ x, x > 0 → x - Real.sqrt (4 * x) < m - Real.sqrt n) →  -- m - √n is indeed the maximum
  m + n = 544 := by
sorry

end semicircle_in_square_l4055_405537


namespace point_coordinates_l4055_405568

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the 2D plane -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: A point in the second quadrant with distance 2 to the x-axis
    and distance 3 to the y-axis has coordinates (-3, 2) -/
theorem point_coordinates (p : Point) 
    (h1 : SecondQuadrant p) 
    (h2 : DistanceToXAxis p = 2) 
    (h3 : DistanceToYAxis p = 3) : 
    p = Point.mk (-3) 2 := by
  sorry

end point_coordinates_l4055_405568


namespace west_movement_l4055_405596

-- Define a type for direction
inductive Direction
| East
| West

-- Define a function to represent movement
def movement (dir : Direction) (distance : ℤ) : ℤ :=
  match dir with
  | Direction.East => distance
  | Direction.West => -distance

-- State the theorem
theorem west_movement :
  (movement Direction.East 50 = 50) →
  (∀ (d : Direction) (x : ℤ), movement d x = -movement (match d with
    | Direction.East => Direction.West
    | Direction.West => Direction.East) x) →
  (movement Direction.West 60 = -60) :=
by
  sorry

end west_movement_l4055_405596


namespace lottery_probability_l4055_405572

/-- The number of people participating in the lottery drawing -/
def num_people : ℕ := 4

/-- The total number of tickets in the box -/
def total_tickets : ℕ := 4

/-- The number of winning tickets -/
def winning_tickets : ℕ := 2

/-- The probability that the event ends right after the third person has finished drawing -/
def prob_end_after_third : ℚ := 1/3

theorem lottery_probability :
  (num_people = 4) →
  (total_tickets = 4) →
  (winning_tickets = 2) →
  (prob_end_after_third = 1/3) := by
  sorry

#check lottery_probability

end lottery_probability_l4055_405572


namespace perfect_square_condition_l4055_405521

/-- If x^2 + 6x + k^2 is exactly the square of a polynomial, then k = ±3 -/
theorem perfect_square_condition (k : ℝ) : 
  (∃ (p : ℝ → ℝ), ∀ x, x^2 + 6*x + k^2 = (p x)^2) → k = 3 ∨ k = -3 := by
  sorry

end perfect_square_condition_l4055_405521


namespace apple_picking_fraction_l4055_405515

theorem apple_picking_fraction (total_apples : ℕ) (remaining_apples : ℕ) : 
  total_apples = 200 →
  remaining_apples = 20 →
  ∃ f : ℚ, 
    f > 0 ∧ 
    f < 1 ∧
    (f * total_apples : ℚ) + (2 * f * total_apples : ℚ) + (f * total_apples + 20 : ℚ) = total_apples - remaining_apples ∧
    f = 1/5 := by
  sorry

end apple_picking_fraction_l4055_405515


namespace negation_of_proposition_l4055_405505

theorem negation_of_proposition (x y : ℝ) : 
  ¬(x > 0 ∧ y > 0 → x * y > 0) ↔ ((x ≤ 0 ∨ y ≤ 0) → x * y ≤ 0) := by sorry

end negation_of_proposition_l4055_405505


namespace participant_age_l4055_405500

/-- Represents the initial state of the lecture rooms -/
structure LectureRooms where
  room1_count : ℕ
  room1_avg_age : ℕ
  room2_count : ℕ
  room2_avg_age : ℕ

/-- Calculates the total age sum of all participants -/
def total_age_sum (rooms : LectureRooms) : ℕ :=
  rooms.room1_count * rooms.room1_avg_age + rooms.room2_count * rooms.room2_avg_age

/-- Calculates the total number of participants -/
def total_count (rooms : LectureRooms) : ℕ :=
  rooms.room1_count + rooms.room2_count

/-- Theorem stating the age of the participant who left -/
theorem participant_age (rooms : LectureRooms) 
  (h1 : rooms.room1_count = 8)
  (h2 : rooms.room1_avg_age = 20)
  (h3 : rooms.room2_count = 12)
  (h4 : rooms.room2_avg_age = 45)
  (h5 : (total_age_sum rooms - x) / (total_count rooms - 1) = (total_age_sum rooms) / (total_count rooms) + 1) :
  x = 16 :=
sorry


end participant_age_l4055_405500


namespace quadratic_distinct_roots_l4055_405539

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
sorry

end quadratic_distinct_roots_l4055_405539


namespace distance_after_skating_l4055_405528

/-- Calculates the distance between two skaters moving in opposite directions -/
def distance_between_skaters (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed1 * time) + (speed2 * time)

/-- Theorem: The distance between Ann and Glenda after skating for 3 hours -/
theorem distance_after_skating :
  let ann_speed : ℝ := 6
  let glenda_speed : ℝ := 8
  let skating_time : ℝ := 3
  distance_between_skaters ann_speed glenda_speed skating_time = 42 := by
  sorry

#check distance_after_skating

end distance_after_skating_l4055_405528


namespace geometric_sequence_common_ratio_l4055_405578

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a1 : a 1 = 3)
  (h_arith : arithmetic_sequence (λ n => match n with
    | 1 => 4 * (a 1)
    | 2 => 2 * (a 2)
    | 3 => a 3
    | _ => 0
  )) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 :=
by sorry

end geometric_sequence_common_ratio_l4055_405578


namespace f_2018_equals_neg_2018_l4055_405593

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -1 / f (x + 3)

theorem f_2018_equals_neg_2018
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_eq : satisfies_equation f)
  (h_f4 : f 4 = -2018) :
  f 2018 = -2018 :=
sorry

end f_2018_equals_neg_2018_l4055_405593


namespace total_eggs_needed_l4055_405545

def eggs_from_andrew : ℕ := 155
def eggs_to_buy : ℕ := 67

theorem total_eggs_needed : 
  eggs_from_andrew + eggs_to_buy = 222 := by sorry

end total_eggs_needed_l4055_405545


namespace arithmetic_calculations_l4055_405532

theorem arithmetic_calculations : 
  (12 - (-18) + (-7) - 15 = 8) ∧ 
  ((-1/2) * (-8) + (-6) / (-1/3)^2 = -50) := by
  sorry

end arithmetic_calculations_l4055_405532


namespace inverse_function_decreasing_l4055_405512

theorem inverse_function_decreasing :
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → x₁ > x₂ → (1 / x₁) < (1 / x₂) := by
  sorry

end inverse_function_decreasing_l4055_405512


namespace sqrt_x_minus_one_meaningful_l4055_405549

theorem sqrt_x_minus_one_meaningful (x : ℝ) : x = 2 → ∃ y : ℝ, y ^ 2 = x - 1 := by
  sorry

end sqrt_x_minus_one_meaningful_l4055_405549


namespace wall_building_theorem_l4055_405552

/-- The number of men in the first group that can build a 112-metre wall in 6 days,
    given that 40 men can build a similar wall in 3 days. -/
def number_of_men : ℕ := 80

/-- The length of the wall in metres. -/
def wall_length : ℕ := 112

/-- The number of days it takes the first group to build the wall. -/
def days_first_group : ℕ := 6

/-- The number of men in the second group. -/
def men_second_group : ℕ := 40

/-- The number of days it takes the second group to build the wall. -/
def days_second_group : ℕ := 3

theorem wall_building_theorem :
  number_of_men * days_second_group = men_second_group * days_first_group :=
sorry

end wall_building_theorem_l4055_405552


namespace intersection_slope_l4055_405514

/-- Given two lines p and q that intersect at (-4, -7), 
    prove that the slope of line q is 2.5 -/
theorem intersection_slope (k : ℝ) : 
  (∀ x y, y = 3 * x + 5 → y = k * x + 3 → x = -4 ∧ y = -7) → 
  k = 2.5 := by
  sorry

end intersection_slope_l4055_405514


namespace peters_pond_depth_l4055_405526

theorem peters_pond_depth :
  ∀ (mark_depth peter_depth : ℝ),
    mark_depth = 3 * peter_depth + 4 →
    mark_depth = 19 →
    peter_depth = 5 := by
  sorry

end peters_pond_depth_l4055_405526


namespace snack_combinations_l4055_405529

def num_items : ℕ := 4
def items_to_choose : ℕ := 2

theorem snack_combinations : 
  Nat.choose num_items items_to_choose = 6 := by sorry

end snack_combinations_l4055_405529


namespace set_problem_l4055_405548

theorem set_problem (U A B : Finset ℕ) (h1 : U.card = 190) (h2 : B.card = 49)
  (h3 : (U \ (A ∪ B)).card = 59) (h4 : (A ∩ B).card = 23) :
  A.card = 105 := by
  sorry

end set_problem_l4055_405548


namespace trisector_triangle_angles_l4055_405531

/-- Given a triangle ABC with angles α, β, and γ, if the triangle formed by the first angle trisectors
    has two angles of 45° and 55°, then the triangle formed by the second angle trisectors
    has angles of 40°, 65°, and 75°. -/
theorem trisector_triangle_angles 
  (α β γ : Real) 
  (h_sum : α + β + γ = 180)
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ)
  (h_first_trisector : 
    ((β + 2*γ)/3 = 45 ∧ (γ + 2*α)/3 = 55) ∨ 
    ((β + 2*γ)/3 = 55 ∧ (γ + 2*α)/3 = 45) ∨
    ((γ + 2*α)/3 = 45 ∧ (α + 2*β)/3 = 55) ∨
    ((γ + 2*α)/3 = 55 ∧ (α + 2*β)/3 = 45) ∨
    ((α + 2*β)/3 = 45 ∧ (β + 2*γ)/3 = 55) ∨
    ((α + 2*β)/3 = 55 ∧ (β + 2*γ)/3 = 45)) :
  (2*β + γ)/3 = 65 ∧ (2*γ + α)/3 = 40 ∧ (2*α + β)/3 = 75 := by
  sorry


end trisector_triangle_angles_l4055_405531


namespace sum_of_squares_on_sides_l4055_405546

/-- Given a right triangle XYZ with XY = 8 and YZ = 17, 
    the sum of the areas of squares constructed on sides YZ and XZ is 514. -/
theorem sum_of_squares_on_sides (X Y Z : ℝ × ℝ) : 
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 8^2 →
  (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 = 17^2 →
  (X.1 - Z.1)^2 + (X.2 - Z.2)^2 = ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) + ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) →
  17^2 + ((X.1 - Z.1)^2 + (X.2 - Z.2)^2) = 514 := by
  sorry

end sum_of_squares_on_sides_l4055_405546


namespace quadratic_two_distinct_roots_l4055_405504

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 + m*x₁ - 8 = 0) ∧ (x₂^2 + m*x₂ - 8 = 0) :=
sorry

end quadratic_two_distinct_roots_l4055_405504


namespace ending_number_proof_l4055_405542

/-- The ending number for a sequence of even numbers -/
def ending_number : ℕ := 20

/-- The average of the sequence -/
def average : ℕ := 16

/-- The starting point of the sequence -/
def start : ℕ := 11

theorem ending_number_proof :
  ∀ n : ℕ,
  n > start →
  n ≤ ending_number →
  n % 2 = 0 →
  2 * average = 12 + ending_number :=
sorry

end ending_number_proof_l4055_405542


namespace arithmetic_sequence_nth_term_l4055_405540

/-- An arithmetic sequence with first term 1 and common difference 3 -/
def arithmetic_sequence (n : ℕ) : ℤ := 1 + 3 * (n - 1)

/-- Theorem: If the nth term of the arithmetic sequence is 2014, then n is 672 -/
theorem arithmetic_sequence_nth_term (n : ℕ) :
  arithmetic_sequence n = 2014 → n = 672 := by
  sorry

end arithmetic_sequence_nth_term_l4055_405540


namespace acme_cheaper_at_min_shirts_l4055_405507

/-- Acme T-Shirt Company's setup fee -/
def acme_setup : ℕ := 70

/-- Acme T-Shirt Company's per-shirt cost -/
def acme_per_shirt : ℕ := 11

/-- Beta T-Shirt Company's setup fee -/
def beta_setup : ℕ := 10

/-- Beta T-Shirt Company's per-shirt cost -/
def beta_per_shirt : ℕ := 15

/-- The minimum number of shirts for which Acme is cheaper than Beta -/
def min_shirts_for_acme : ℕ := 16

theorem acme_cheaper_at_min_shirts :
  acme_setup + acme_per_shirt * min_shirts_for_acme < 
  beta_setup + beta_per_shirt * min_shirts_for_acme ∧
  ∀ n : ℕ, n < min_shirts_for_acme → 
    acme_setup + acme_per_shirt * n ≥ beta_setup + beta_per_shirt * n :=
by sorry

end acme_cheaper_at_min_shirts_l4055_405507


namespace award_sequences_eq_sixteen_l4055_405518

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 5

/-- Represents the number of rounds in the tournament -/
def num_rounds : ℕ := 4

/-- Calculates the number of possible award sequences -/
def award_sequences : ℕ := 2^num_rounds

/-- Theorem stating that the number of award sequences is 16 -/
theorem award_sequences_eq_sixteen : award_sequences = 16 := by
  sorry

end award_sequences_eq_sixteen_l4055_405518


namespace nonnegative_integer_representation_l4055_405599

theorem nonnegative_integer_representation (n : ℕ) : 
  ∃ (a b c : ℕ+), n = a^2 + b^2 - c^2 ∧ a < b ∧ b < c := by
  sorry

end nonnegative_integer_representation_l4055_405599


namespace sufficient_not_necessary_condition_l4055_405566

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x > y ∧ y > 0 → x/y > 1) ∧
  ∃ a b : ℝ, a/b > 1 ∧ ¬(a > b ∧ b > 0) :=
sorry

end sufficient_not_necessary_condition_l4055_405566


namespace marked_price_calculation_l4055_405541

theorem marked_price_calculation (cost_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  cost_price = 100 →
  discount_rate = 0.2 →
  profit_rate = 0.2 →
  ∃ (marked_price : ℝ), 
    marked_price * (1 - discount_rate) = cost_price * (1 + profit_rate) ∧
    marked_price = 150 :=
by sorry

end marked_price_calculation_l4055_405541


namespace tangent_perpendicular_to_line_l4055_405590

-- Define the curve
def curve (x : ℝ) : ℝ := 2 * x^2

-- Define the line perpendicular to the tangent
def perp_line (x y : ℝ) : Prop := x + 4*y - 8 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 4*x - y - 2 = 0

-- Theorem statement
theorem tangent_perpendicular_to_line :
  ∀ (x₀ y₀ : ℝ),
  y₀ = curve x₀ →
  (∃ (m : ℝ), ∀ (x y : ℝ), y - y₀ = m * (x - x₀) → 
    (perp_line x y ↔ (x - x₀) * 1 + (y - y₀) * 4 = 0)) →
  tangent_line x₀ y₀ :=
sorry

end tangent_perpendicular_to_line_l4055_405590


namespace samuel_initial_skittles_l4055_405534

/-- The number of friends Samuel gave Skittles to -/
def num_friends : ℕ := 4

/-- The number of Skittles each person (including Samuel) ate -/
def skittles_per_person : ℕ := 3

/-- The initial number of Skittles Samuel had -/
def initial_skittles : ℕ := num_friends * skittles_per_person + skittles_per_person

/-- Theorem stating that Samuel initially had 15 Skittles -/
theorem samuel_initial_skittles : initial_skittles = 15 := by
  sorry

end samuel_initial_skittles_l4055_405534


namespace profit_at_80_max_profit_profit_range_l4055_405520

/-- Represents the clothing sale scenario with given constraints -/
structure ClothingSale where
  cost : ℝ
  demand : ℝ → ℝ
  profit_function : ℝ → ℝ
  max_profit_percentage : ℝ

/-- The specific clothing sale scenario from the problem -/
def sale : ClothingSale :=
  { cost := 60
  , demand := λ x => -x + 120
  , profit_function := λ x => (x - 60) * (-x + 120)
  , max_profit_percentage := 0.4 }

/-- Theorem stating the profit when selling price is 80 -/
theorem profit_at_80 (s : ClothingSale) (h : s = sale) :
  s.profit_function 80 = 800 :=
sorry

/-- Theorem stating the maximum profit and corresponding selling price -/
theorem max_profit (s : ClothingSale) (h : s = sale) :
  ∃ x, x ≤ (1 + s.max_profit_percentage) * s.cost ∧
      s.profit_function x = 864 ∧
      ∀ y, y ≤ (1 + s.max_profit_percentage) * s.cost →
        s.profit_function y ≤ s.profit_function x :=
sorry

/-- Theorem stating the range of selling prices for profit not less than 500 -/
theorem profit_range (s : ClothingSale) (h : s = sale) :
  ∀ x, s.cost ≤ x ∧ x ≤ (1 + s.max_profit_percentage) * s.cost →
    (s.profit_function x ≥ 500 ↔ 70 ≤ x ∧ x ≤ 84) :=
sorry

end profit_at_80_max_profit_profit_range_l4055_405520


namespace two_sevenths_as_distinct_unit_fractions_l4055_405563

theorem two_sevenths_as_distinct_unit_fractions :
  ∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (2 : ℚ) / 7 = 1 / a + 1 / b + 1 / c :=
sorry

end two_sevenths_as_distinct_unit_fractions_l4055_405563


namespace expected_asthma_cases_l4055_405560

theorem expected_asthma_cases (total_sample : ℕ) (asthma_rate : ℚ) 
  (h1 : total_sample = 320) 
  (h2 : asthma_rate = 1 / 8) : 
  ⌊total_sample * asthma_rate⌋ = 40 := by
  sorry

end expected_asthma_cases_l4055_405560


namespace inverse_matrices_sum_l4055_405558

def A (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![x, 2, x^2],
    ![3, y, 4],
    ![z, 3, z^2]]

def B (x y z k l m n : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![-8, k, -x^3],
    ![l, -y^2, m],
    ![3, n, z^3]]

theorem inverse_matrices_sum (x y z k l m n : ℝ) :
  A x y z * B x y z k l m n = 1 →
  x + y + z + k + l + m + n = -1/3 := by sorry

end inverse_matrices_sum_l4055_405558


namespace binomial_divisibility_l4055_405554

theorem binomial_divisibility (p m : ℕ) (hp : Prime p) (hm : m > 0) :
  p^m ∣ (Nat.choose (p^m) p - p^(m-1)) := by
  sorry

end binomial_divisibility_l4055_405554


namespace max_value_trig_expression_l4055_405597

theorem max_value_trig_expression :
  ∀ x : ℝ, 2 * Real.cos x + 3 * Real.sin x ≤ Real.sqrt 13 := by
  sorry

end max_value_trig_expression_l4055_405597


namespace factorialLastNonzeroDigitSeq_not_periodic_l4055_405567

/-- The last nonzero digit of a natural number -/
def lastNonzeroDigit (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10

/-- The sequence of last nonzero digits of factorials -/
def factorialLastNonzeroDigitSeq : ℕ → ℕ :=
  fun n => lastNonzeroDigit (Nat.factorial n)

/-- The sequence of last nonzero digits of factorials is not periodic -/
theorem factorialLastNonzeroDigitSeq_not_periodic :
  ¬ ∃ (p : ℕ), p > 0 ∧ ∀ (n : ℕ), factorialLastNonzeroDigitSeq (n + p) = factorialLastNonzeroDigitSeq n :=
sorry

end factorialLastNonzeroDigitSeq_not_periodic_l4055_405567


namespace tan_difference_l4055_405562

theorem tan_difference (α β : Real) (h1 : Real.tan α = 3) (h2 : Real.tan β = 4/3) :
  Real.tan (α - β) = 1/3 := by
  sorry

end tan_difference_l4055_405562


namespace geometric_sequence_sum_l4055_405575

/-- Given a geometric sequence of real numbers {a_n}, prove that if the sum of the first three terms is 2
    and the sum of the 4th, 5th, and 6th terms is 16, then the sum of the 7th, 8th, and 9th terms is 128. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geometric : ∀ n : ℕ, ∃ q : ℝ, a (n + 1) = q * a n)
    (h_sum1 : a 1 + a 2 + a 3 = 2) (h_sum2 : a 4 + a 5 + a 6 = 16) : a 7 + a 8 + a 9 = 128 := by
  sorry


end geometric_sequence_sum_l4055_405575


namespace largest_solution_of_equation_l4055_405585

theorem largest_solution_of_equation : 
  ∃ (x : ℝ), x = 6 ∧ 3 * x^2 + 18 * x - 84 = x * (x + 10) ∧
  ∀ (y : ℝ), 3 * y^2 + 18 * y - 84 = y * (y + 10) → y ≤ x :=
sorry

end largest_solution_of_equation_l4055_405585


namespace polynomial_properties_l4055_405553

theorem polynomial_properties (p q : ℤ) :
  (∀ x : ℤ, ∃ k : ℤ, x^2 + p*x + q = 2*k ↔ Even q ∧ Odd p) ∧ 
  (∀ x : ℤ, ∃ k : ℤ, x^2 + p*x + q = 2*k + 1 ↔ Odd q ∧ Odd p) ∧
  (∀ x : ℤ, ∃ k : ℤ, x^3 + p*x + q = 3*k ↔ q % 3 = 0 ∧ p % 3 = 2) := by
  sorry

end polynomial_properties_l4055_405553


namespace right_triangle_with_consecutive_sides_l4055_405589

theorem right_triangle_with_consecutive_sides (a b c : ℕ) : 
  a = 11 → b + 1 = c → a^2 + b^2 = c^2 → c = 61 := by sorry

end right_triangle_with_consecutive_sides_l4055_405589


namespace log_expression_equality_l4055_405595

theorem log_expression_equality : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 + 8^(1/4) * 2^(1/4) = 4 := by
  sorry

end log_expression_equality_l4055_405595


namespace bacteria_growth_problem_l4055_405556

/-- Bacteria growth problem -/
theorem bacteria_growth_problem (initial_count : ℕ) : 
  (∀ (period : ℕ), initial_count * (4 ^ period) = initial_count * 4 ^ period) →
  initial_count * 4 ^ 4 = 262144 →
  initial_count = 1024 := by
  sorry

end bacteria_growth_problem_l4055_405556


namespace select_five_from_fifteen_l4055_405543

theorem select_five_from_fifteen (n : Nat) (r : Nat) : n = 15 ∧ r = 5 →
  Nat.choose n r = 3003 := by
  sorry

end select_five_from_fifteen_l4055_405543


namespace equation_solutions_l4055_405506

theorem equation_solutions :
  (∀ x : ℝ, x * (x + 1) = x + 1 ↔ x = -1 ∨ x = 1) ∧
  (∀ x : ℝ, 2 * x^2 - 4 * x + 1 = 0 ↔ x = 1 + Real.sqrt 2 / 2 ∨ x = 1 - Real.sqrt 2 / 2) :=
by sorry

end equation_solutions_l4055_405506


namespace power_three_405_mod_13_l4055_405557

theorem power_three_405_mod_13 : 3^405 ≡ 1 [ZMOD 13] := by
  sorry

end power_three_405_mod_13_l4055_405557


namespace f_comparison_l4055_405598

-- Define f as a function from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f (-x) = f x)
variable (h_decreasing : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)

-- State the theorem
theorem f_comparison (a : ℝ) : f (-3/4) ≥ f (a^2 - a + 1) := by
  sorry

end f_comparison_l4055_405598


namespace linear_inequality_m_value_l4055_405503

/-- If 3m - 5x^(3+m) > 4 is a linear inequality in x, then m = -2 -/
theorem linear_inequality_m_value (m : ℝ) : 
  (∃ (a b : ℝ), ∀ x, 3*m - 5*x^(3+m) > 4 ↔ a*x + b > 0) → m = -2 :=
sorry

end linear_inequality_m_value_l4055_405503


namespace triangle_existence_condition_l4055_405522

/-- A triangle with side lengths 3, 2x+1, and 10 exists if and only if 3 < x < 6 -/
theorem triangle_existence_condition (x : ℝ) :
  (3 : ℝ) < x ∧ x < 6 ↔ 
  (3 : ℝ) + (2*x + 1) > 10 ∧
  (3 : ℝ) + 10 > 2*x + 1 ∧
  10 + (2*x + 1) > 3 := by
sorry

end triangle_existence_condition_l4055_405522


namespace dave_added_sixty_apps_l4055_405588

/-- Calculates the number of apps Dave added to his phone -/
def apps_added (initial : ℕ) (removed : ℕ) (final : ℕ) : ℕ :=
  final - (initial - removed)

/-- Proves that Dave added 60 apps to his phone -/
theorem dave_added_sixty_apps :
  apps_added 50 10 100 = 60 := by
  sorry

end dave_added_sixty_apps_l4055_405588


namespace chord_length_l4055_405502

/-- In a circle with radius 15 units, a chord that is a perpendicular bisector of the radius has a length of 26√3 units. -/
theorem chord_length (r : ℝ) (c : ℝ) : 
  r = 15 → -- The radius is 15 units
  c^2 = 4 * (r^2 - (r/2)^2) → -- The chord is a perpendicular bisector of the radius
  c = 26 * Real.sqrt 3 := by -- The length of the chord is 26√3 units
sorry

end chord_length_l4055_405502


namespace dollar_evaluation_l4055_405511

def dollar (a b : ℝ) : ℝ := (a - b)^2

theorem dollar_evaluation (x y : ℝ) :
  dollar (2*x + 3*y) (3*x - 4*y) = x^2 - 14*x*y + 49*y^2 := by
  sorry

end dollar_evaluation_l4055_405511


namespace f_properties_l4055_405517

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_properties :
  let e := Real.exp 1
  (∀ x ∈ Set.Ioo 0 e, ∀ y ∈ Set.Ioo 0 e, x < y → f x < f y) ∧
  (∀ x ∈ Set.Ioi e, ∀ y ∈ Set.Ioi e, x < y → f x > f y) ∧
  (∀ x ∈ Set.Ioo 0 (Real.exp 1), f x ≤ f e) ∧
  (∀ x ∈ Set.Ioi (Real.exp 1), f x < f e) ∧
  (∀ a : ℝ, (∀ x ≥ 1, f x ≤ a * (1 - 1 / x^2)) ↔ a ≥ 1/2) :=
by sorry


end f_properties_l4055_405517


namespace expression_evaluation_l4055_405569

theorem expression_evaluation : 
  (3^2 + 5^2 + 7^2) / (2^2 + 4^2 + 6^2) - (2^2 + 4^2 + 6^2) / (3^2 + 5^2 + 7^2) = 3753/4648 := by
  sorry

end expression_evaluation_l4055_405569


namespace correct_exponent_calculation_l4055_405564

theorem correct_exponent_calculation (a : ℝ) : (-a)^6 / a^3 = a^3 := by
  sorry

end correct_exponent_calculation_l4055_405564


namespace min_area_two_rectangles_l4055_405579

/-- Given a wire of length l, cut into two pieces x and (l-x), forming two rectangles
    with length-to-width ratios of 2:1 and 3:2 respectively, the minimum value of 
    the sum of their areas is 3/104 * l^2 --/
theorem min_area_two_rectangles (l : ℝ) (h : l > 0) :
  ∃ (x : ℝ), 0 < x ∧ x < l ∧
  (∀ (y : ℝ), 0 < y → y < l →
    x^2 / 18 + 3 * (l - x)^2 / 50 ≤ y^2 / 18 + 3 * (l - y)^2 / 50) ∧
  x^2 / 18 + 3 * (l - x)^2 / 50 = 3 * l^2 / 104 :=
sorry

end min_area_two_rectangles_l4055_405579


namespace sum_of_squares_for_specific_conditions_l4055_405524

theorem sum_of_squares_for_specific_conditions : 
  ∃ (S : Finset ℕ), 
    (∀ s ∈ S, ∃ x y z : ℕ, 
      x > 0 ∧ y > 0 ∧ z > 0 ∧
      x + y + z = 30 ∧ 
      Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 12 ∧
      s = x^2 + y^2 + z^2) ∧
    (∀ x y z : ℕ, 
      x > 0 → y > 0 → z > 0 →
      x + y + z = 30 → 
      Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 12 →
      (x^2 + y^2 + z^2) ∈ S) ∧
    S.sum id = 710 :=
by sorry

end sum_of_squares_for_specific_conditions_l4055_405524


namespace xiaoming_money_l4055_405559

/-- Proves that Xiaoming brought 108 yuan to the supermarket -/
theorem xiaoming_money (fresh_milk_cost yogurt_cost : ℕ) 
  (fresh_milk_cartons yogurt_cartons total_money : ℕ) : 
  fresh_milk_cost = 6 →
  yogurt_cost = 9 →
  fresh_milk_cost * fresh_milk_cartons = total_money →
  yogurt_cost * yogurt_cartons = total_money →
  fresh_milk_cartons = yogurt_cartons + 6 →
  total_money = 108 := by
  sorry

#check xiaoming_money

end xiaoming_money_l4055_405559


namespace rational_equation_solution_l4055_405538

theorem rational_equation_solution (k : ℝ) (x : ℝ) (h : x ≠ 4) :
  (x^2 - 3*x - 4) / (x - 4) = 3*x + k → x = (1 - k) / 2 :=
by sorry

end rational_equation_solution_l4055_405538


namespace f_sum_symmetric_l4055_405533

def f (x : ℝ) : ℝ := x^3 + 2*x

theorem f_sum_symmetric : f 5 + f (-5) = 0 := by
  sorry

end f_sum_symmetric_l4055_405533
