import Mathlib

namespace NUMINAMATH_CALUDE_average_running_distance_l1396_139655

def monday_distance : ℝ := 4.2
def tuesday_distance : ℝ := 3.8
def wednesday_distance : ℝ := 3.6
def thursday_distance : ℝ := 4.4
def number_of_days : ℕ := 4

theorem average_running_distance :
  (monday_distance + tuesday_distance + wednesday_distance + thursday_distance) / number_of_days = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_running_distance_l1396_139655


namespace NUMINAMATH_CALUDE_units_digit_of_product_division_l1396_139629

theorem units_digit_of_product_division : 
  (12 * 13 * 14 * 15 * 16 * 17) / 2000 % 10 = 6 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_division_l1396_139629


namespace NUMINAMATH_CALUDE_symmetry_point_l1396_139676

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- A line in the form y = mx + b -/
structure Line where
  m : ℚ
  b : ℚ

/-- Check if two points are symmetric with respect to a line -/
def areSymmetric (P Q : Point) (l : Line) : Prop :=
  -- The product of the slopes of PQ and l is -1
  ((Q.y - P.y) / (Q.x - P.x)) * l.m = -1 ∧
  -- The midpoint of PQ lies on l
  ((Q.y + P.y) / 2) = l.m * ((Q.x + P.x) / 2) + l.b

theorem symmetry_point :
  let P : Point := ⟨-1, 2⟩
  let Q : Point := ⟨7/5, 4/5⟩
  let l : Line := ⟨2, 1⟩  -- y = 2x + 1
  areSymmetric P Q l := by sorry

end NUMINAMATH_CALUDE_symmetry_point_l1396_139676


namespace NUMINAMATH_CALUDE_range_of_2a_plus_3b_l1396_139646

theorem range_of_2a_plus_3b (a b : ℝ) 
  (h1 : -1 < a + b) (h2 : a + b < 3) 
  (h3 : 2 < a - b) (h4 : a - b < 4) : 
  ∀ x, (2*a + 3*b = x) → (-9/2 < x ∧ x < 13/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_2a_plus_3b_l1396_139646


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l1396_139644

/-- Given a geometric sequence {a_n} where the first three terms are x, x-1, and 2x-2 respectively,
    prove that the general term is a_n = -2^(n-1) -/
theorem geometric_sequence_general_term (a : ℕ → ℝ) (x : ℝ) (h1 : a 1 = x) (h2 : a 2 = x - 1) (h3 : a 3 = 2*x - 2) :
  ∀ n : ℕ, a n = -2^(n-1) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l1396_139644


namespace NUMINAMATH_CALUDE_function_composition_equality_l1396_139663

/-- Given f(x) = x/3 + 4 and g(x) = 7 - x, if f(g(a)) = 6, then a = 1 -/
theorem function_composition_equality (f g : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = x / 3 + 4)
  (hg : ∀ x, g x = 7 - x)
  (h : f (g a) = 6) : 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_function_composition_equality_l1396_139663


namespace NUMINAMATH_CALUDE_log_system_solutions_l1396_139616

noncomputable def solve_log_system (x y : ℝ) : Prop :=
  x > y ∧ y > 0 ∧
  Real.log (x - y) + Real.log 2 = (1 / 2) * (Real.log x - Real.log y) ∧
  Real.log (x + y) - Real.log 3 = (1 / 2) * (Real.log y - Real.log x)

theorem log_system_solutions :
  (∃ (x y : ℝ), solve_log_system x y ∧ 
    ((x = Real.sqrt 2 ∧ y = 1 / Real.sqrt 2) ∨ 
     (x = 3 * Real.sqrt 3 / 4 ∧ y = Real.sqrt 3 / 4))) ∧
  (∀ (x y : ℝ), solve_log_system x y → 
    ((x = Real.sqrt 2 ∧ y = 1 / Real.sqrt 2) ∨ 
     (x = 3 * Real.sqrt 3 / 4 ∧ y = Real.sqrt 3 / 4))) :=
by sorry

end NUMINAMATH_CALUDE_log_system_solutions_l1396_139616


namespace NUMINAMATH_CALUDE_recurrence_sequence_uniqueness_l1396_139691

/-- A sequence defined by a recurrence relation -/
def RecurrenceSequence (p q : ℝ) (a₀ a₁ : ℝ) : ℕ → ℝ
| 0 => a₀
| 1 => a₁
| (n + 2) => p * RecurrenceSequence p q a₀ a₁ (n + 1) + q * RecurrenceSequence p q a₀ a₁ n

/-- Theorem: All terms in the sequence are uniquely determined -/
theorem recurrence_sequence_uniqueness (p q : ℝ) (a₀ a₁ : ℝ) :
  ∀ n : ℕ, ∃! x : ℝ, x = RecurrenceSequence p q a₀ a₁ n :=
by sorry

end NUMINAMATH_CALUDE_recurrence_sequence_uniqueness_l1396_139691


namespace NUMINAMATH_CALUDE_parallel_perpendicular_transitivity_l1396_139653

/-- A structure representing a 3D space with lines and planes -/
structure Space3D where
  Line : Type
  Plane : Type
  parallel_lines : Line → Line → Prop
  perpendicular_line_plane : Line → Plane → Prop

/-- The theorem stating that if a line is parallel to another line that is perpendicular to a plane, 
    then the first line is also perpendicular to that plane -/
theorem parallel_perpendicular_transitivity 
  {S : Space3D} {m n : S.Line} {α : S.Plane} :
  S.parallel_lines m n → S.perpendicular_line_plane m α → S.perpendicular_line_plane n α :=
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_transitivity_l1396_139653


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l1396_139664

-- Define the trapezoid EFGH
structure Trapezoid :=
  (EF : ℝ) (GH : ℝ) (height : ℝ)

-- Define the properties of the trapezoid
def isIsoscelesTrapezoid (t : Trapezoid) : Prop :=
  t.EF = t.GH

-- Theorem statement
theorem trapezoid_perimeter 
  (t : Trapezoid) 
  (h1 : isIsoscelesTrapezoid t) 
  (h2 : t.height = 5) 
  (h3 : t.GH = 10) 
  (h4 : t.EF = 4) : 
  ∃ (perimeter : ℝ), perimeter = 14 + 2 * Real.sqrt 34 :=
by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l1396_139664


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l1396_139619

open Complex

theorem modulus_of_complex_number : ∃ z : ℂ, z = (2 - I)^2 / I ∧ Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l1396_139619


namespace NUMINAMATH_CALUDE_alex_age_difference_l1396_139650

/-- Proves the number of years ago when Alex was one-third as old as his father -/
theorem alex_age_difference (alex_current_age : ℝ) (alex_father_age : ℝ) (years_ago : ℝ) : 
  alex_current_age = 16.9996700066 →
  alex_father_age = 2 * alex_current_age + 5 →
  alex_current_age - years_ago = (1 / 3) * (alex_father_age - years_ago) →
  years_ago = 6.4998350033 := by
sorry

end NUMINAMATH_CALUDE_alex_age_difference_l1396_139650


namespace NUMINAMATH_CALUDE_price_reduction_for_same_profit_no_solution_for_460_profit_l1396_139681

/-- Represents the fruit sales scenario at Huimin Fresh Supermarket -/
structure FruitSales where
  cost_price : ℝ
  initial_selling_price : ℝ
  initial_daily_sales : ℝ
  sales_increase_rate : ℝ

/-- Calculates the daily profit given a price reduction -/
def daily_profit (fs : FruitSales) (price_reduction : ℝ) : ℝ :=
  (fs.initial_selling_price - price_reduction - fs.cost_price) *
  (fs.initial_daily_sales + fs.sales_increase_rate * price_reduction)

/-- The scenario described in the problem -/
def huimin_scenario : FruitSales := {
  cost_price := 20
  initial_selling_price := 40
  initial_daily_sales := 20
  sales_increase_rate := 2
}

theorem price_reduction_for_same_profit :
  daily_profit huimin_scenario 10 = daily_profit huimin_scenario 0 := by sorry

theorem no_solution_for_460_profit :
  ∀ x : ℝ, daily_profit huimin_scenario x ≠ 460 := by sorry

end NUMINAMATH_CALUDE_price_reduction_for_same_profit_no_solution_for_460_profit_l1396_139681


namespace NUMINAMATH_CALUDE_parabola_vertex_l1396_139610

/-- The vertex of a parabola defined by y = -(x+1)^2 is the point (-1, 0) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -(x + 1)^2 → (∃ (a : ℝ), y = a * (x + 1)^2 ∧ a = -1) → 
  (∃ (h k : ℝ), y = -(x - h)^2 + k ∧ h = -1 ∧ k = 0) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1396_139610


namespace NUMINAMATH_CALUDE_folded_paper_triangle_perimeter_l1396_139665

/-- A square piece of paper with side length 2 is folded such that vertex C meets edge AB at point C',
    making C'B = 2/3. Edge BC intersects edge AD at point E. -/
theorem folded_paper_triangle_perimeter :
  ∀ (A B C D C' E : ℝ × ℝ),
    -- Square conditions
    A = (0, 2) ∧ B = (0, 0) ∧ C = (2, 0) ∧ D = (2, 2) →
    -- Folding conditions
    C' = (0, 4/3) →
    -- Intersection condition
    E = (2, 0) →
    -- Perimeter calculation
    dist A E + dist E C' + dist C' A = 4 :=
by sorry

end NUMINAMATH_CALUDE_folded_paper_triangle_perimeter_l1396_139665


namespace NUMINAMATH_CALUDE_limit_cubic_difference_quotient_l1396_139680

theorem limit_cubic_difference_quotient :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → |(x^3 - 1) / (x - 1) - 3| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_cubic_difference_quotient_l1396_139680


namespace NUMINAMATH_CALUDE_winning_candidate_percentage_l1396_139641

def election_votes : List Nat := [1000, 2000, 4000]

theorem winning_candidate_percentage :
  let total_votes := election_votes.sum
  let winning_votes := election_votes.maximum?
  winning_votes.map (λ w => (w : ℚ) / total_votes * 100) = some (4000 / 7000 * 100) := by
  sorry

end NUMINAMATH_CALUDE_winning_candidate_percentage_l1396_139641


namespace NUMINAMATH_CALUDE_passengers_on_time_l1396_139689

theorem passengers_on_time (total : ℕ) (late : ℕ) (h1 : total = 14720) (h2 : late = 213) :
  total - late = 14507 := by
  sorry

end NUMINAMATH_CALUDE_passengers_on_time_l1396_139689


namespace NUMINAMATH_CALUDE_planes_parallel_or_intersect_l1396_139633

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- Parallel relation between lines -/
def Line.parallel (l1 l2 : Line) : Prop := sorry

/-- A line is contained in a plane -/
def Line.contained_in (l : Line) (p : Plane) : Prop := sorry

/-- Two planes are parallel -/
def Plane.parallel (p1 p2 : Plane) : Prop := sorry

/-- Two planes intersect -/
def Plane.intersect (p1 p2 : Plane) : Prop := sorry

/-- Main theorem: Given the conditions, planes α and β are either parallel or intersecting -/
theorem planes_parallel_or_intersect (α β : Plane) (a b c : Line) 
  (h1 : a.parallel b) (h2 : b.parallel c)
  (h3 : a.contained_in α) (h4 : b.contained_in β) (h5 : c.contained_in β) :
  Plane.parallel α β ∨ Plane.intersect α β := by sorry

end NUMINAMATH_CALUDE_planes_parallel_or_intersect_l1396_139633


namespace NUMINAMATH_CALUDE_inequality_proof_l1396_139631

theorem inequality_proof (a b : ℝ) (h1 : a + b > 0) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  a / b^2 + b / a^2 ≥ 1 / a + 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1396_139631


namespace NUMINAMATH_CALUDE_matrix_determinant_l1396_139605

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 4, 7]

theorem matrix_determinant :
  Matrix.det matrix = 47 := by sorry

end NUMINAMATH_CALUDE_matrix_determinant_l1396_139605


namespace NUMINAMATH_CALUDE_weight_difference_after_one_year_l1396_139609

/-- Calculates the final weight of the labrador puppy after one year -/
def labrador_final_weight (initial_weight : ℝ) : ℝ :=
  let weight1 := initial_weight * 1.1
  let weight2 := weight1 * 1.2
  let weight3 := weight2 * 1.25
  weight3 + 5

/-- Calculates the final weight of the dachshund puppy after one year -/
def dachshund_final_weight (initial_weight : ℝ) : ℝ :=
  let weight1 := initial_weight * 1.05
  let weight2 := weight1 * 1.15
  let weight3 := weight2 - 1
  let weight4 := weight3 * 1.2
  weight4 + 3

/-- The difference in weight between the labrador and dachshund puppies after one year -/
theorem weight_difference_after_one_year :
  labrador_final_weight 40 - dachshund_final_weight 12 = 51.812 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_after_one_year_l1396_139609


namespace NUMINAMATH_CALUDE_tiger_escape_distance_l1396_139695

/-- Represents the speed and duration of each phase of the tiger's escape --/
structure EscapePhase where
  speed : ℝ
  duration : ℝ

/-- Calculates the total distance traveled by the tiger --/
def totalDistance (phases : List EscapePhase) : ℝ :=
  phases.foldl (fun acc phase => acc + phase.speed * phase.duration) 0

/-- The escape phases of the tiger --/
def tigerEscapePhases : List EscapePhase := [
  { speed := 25, duration := 1 },
  { speed := 35, duration := 2 },
  { speed := 20, duration := 1.5 },
  { speed := 10, duration := 1 },
  { speed := 50, duration := 0.5 }
]

theorem tiger_escape_distance :
  totalDistance tigerEscapePhases = 160 := by
  sorry

end NUMINAMATH_CALUDE_tiger_escape_distance_l1396_139695


namespace NUMINAMATH_CALUDE_big_sixteen_game_count_l1396_139611

/-- Represents a basketball league with the given structure -/
structure BasketballLeague where
  totalTeams : Nat
  divisionsCount : Nat
  intraGameCount : Nat
  interGameCount : Nat

/-- Calculates the total number of scheduled games in the league -/
def totalGames (league : BasketballLeague) : Nat :=
  let teamsPerDivision := league.totalTeams / league.divisionsCount
  let intraGamesPerDivision := teamsPerDivision * (teamsPerDivision - 1) / 2 * league.intraGameCount
  let totalIntraGames := intraGamesPerDivision * league.divisionsCount
  let totalInterGames := league.totalTeams * teamsPerDivision * league.interGameCount / 2
  totalIntraGames + totalInterGames

/-- Theorem stating that the Big Sixteen Basketball League schedules 296 games -/
theorem big_sixteen_game_count :
  let bigSixteen : BasketballLeague := {
    totalTeams := 16
    divisionsCount := 2
    intraGameCount := 3
    interGameCount := 2
  }
  totalGames bigSixteen = 296 := by
  sorry

end NUMINAMATH_CALUDE_big_sixteen_game_count_l1396_139611


namespace NUMINAMATH_CALUDE_isosceles_triangles_count_l1396_139682

/-- A triangle represented by its three vertices in 2D space -/
structure Triangle where
  v1 : (Int × Int)
  v2 : (Int × Int)
  v3 : (Int × Int)

/-- Check if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Bool :=
  let d12 := (t.v1.1 - t.v2.1)^2 + (t.v1.2 - t.v2.2)^2
  let d23 := (t.v2.1 - t.v3.1)^2 + (t.v2.2 - t.v3.2)^2
  let d31 := (t.v3.1 - t.v1.1)^2 + (t.v3.2 - t.v1.2)^2
  d12 = d23 || d23 = d31 || d31 = d12

/-- The list of triangles from the problem -/
def triangles : List Triangle := [
  { v1 := (0, 8), v2 := (2, 8), v3 := (1, 6) },
  { v1 := (3, 5), v2 := (3, 8), v3 := (6, 5) },
  { v1 := (0, 2), v2 := (4, 3), v3 := (8, 2) },
  { v1 := (7, 5), v2 := (6, 8), v3 := (10, 5) },
  { v1 := (7, 2), v2 := (8, 4), v3 := (10, 1) },
  { v1 := (3, 1), v2 := (5, 1), v3 := (4, 3) }
]

theorem isosceles_triangles_count : 
  (triangles.filter isIsosceles).length = 5 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangles_count_l1396_139682


namespace NUMINAMATH_CALUDE_ali_flower_sales_l1396_139659

def flower_problem (monday_sales : ℕ) (friday_multiplier : ℕ) (total_sales : ℕ) : Prop :=
  let friday_sales := friday_multiplier * monday_sales
  let tuesday_sales := total_sales - monday_sales - friday_sales
  tuesday_sales = 8

theorem ali_flower_sales : flower_problem 4 2 20 := by
  sorry

end NUMINAMATH_CALUDE_ali_flower_sales_l1396_139659


namespace NUMINAMATH_CALUDE_prob_800_to_1000_l1396_139660

/-- Probability that a light bulb works after 800 hours -/
def prob_800 : ℝ := 0.8

/-- Probability that a light bulb works after 1000 hours -/
def prob_1000 : ℝ := 0.5

/-- Theorem stating the probability of a light bulb continuing to work from 800 to 1000 hours -/
theorem prob_800_to_1000 : (prob_1000 / prob_800 : ℝ) = 5/8 := by sorry

end NUMINAMATH_CALUDE_prob_800_to_1000_l1396_139660


namespace NUMINAMATH_CALUDE_total_spent_proof_l1396_139697

/-- The price of each flower in dollars -/
def flower_price : ℕ := 3

/-- The number of roses Zoe bought -/
def roses_bought : ℕ := 8

/-- The number of daisies Zoe bought -/
def daisies_bought : ℕ := 2

/-- Theorem: Given the price of each flower and the number of roses and daisies bought,
    prove that the total amount spent is 30 dollars -/
theorem total_spent_proof :
  flower_price * (roses_bought + daisies_bought) = 30 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_proof_l1396_139697


namespace NUMINAMATH_CALUDE_vector_sum_proof_l1396_139623

def v1 : Fin 3 → ℤ := ![- 7, 3, 5]
def v2 : Fin 3 → ℤ := ![4, - 1, - 6]
def v3 : Fin 3 → ℤ := ![1, 8, 2]

theorem vector_sum_proof :
  (v1 + v2 + v3) = ![- 2, 10, 1] := by sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l1396_139623


namespace NUMINAMATH_CALUDE_justin_jersey_cost_l1396_139640

/-- The total cost of jerseys bought by Justin -/
def total_cost (long_sleeve_count : ℕ) (long_sleeve_price : ℕ) (striped_count : ℕ) (striped_price : ℕ) : ℕ :=
  long_sleeve_count * long_sleeve_price + striped_count * striped_price

/-- Theorem stating that Justin's total cost for jerseys is $80 -/
theorem justin_jersey_cost :
  total_cost 4 15 2 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_justin_jersey_cost_l1396_139640


namespace NUMINAMATH_CALUDE_b_income_percentage_over_c_l1396_139634

/-- Prove that B's monthly income is 12% more than C's monthly income given the specified conditions --/
theorem b_income_percentage_over_c (a_annual_income b_monthly_income c_monthly_income : ℚ) : 
  c_monthly_income = 16000 →
  a_annual_income = 537600 →
  a_annual_income / 12 / b_monthly_income = 5 / 2 →
  (b_monthly_income - c_monthly_income) / c_monthly_income = 12 / 100 := by
  sorry

end NUMINAMATH_CALUDE_b_income_percentage_over_c_l1396_139634


namespace NUMINAMATH_CALUDE_right_triangle_identification_l1396_139600

def is_right_triangle (a b c : ℕ) : Prop := a * a + b * b = c * c

theorem right_triangle_identification :
  (¬ is_right_triangle 2 3 4) ∧
  (is_right_triangle 3 4 5) ∧
  (¬ is_right_triangle 4 5 6) ∧
  (¬ is_right_triangle 5 6 7) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_identification_l1396_139600


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l1396_139624

def a (n : ℕ) : ℕ := 2 * Nat.factorial n + n

theorem max_gcd_consecutive_terms (n : ℕ) : Nat.gcd (a n) (a (n + 1)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l1396_139624


namespace NUMINAMATH_CALUDE_imaginary_part_of_fraction_l1396_139674

theorem imaginary_part_of_fraction (i : ℂ) : i * i = -1 → Complex.im ((1 - i) / (1 + i)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_fraction_l1396_139674


namespace NUMINAMATH_CALUDE_cos_75_degrees_l1396_139657

theorem cos_75_degrees :
  Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_degrees_l1396_139657


namespace NUMINAMATH_CALUDE_sum_15_terms_l1396_139608

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- Sum of the first 5 terms -/
  sum5 : ℝ
  /-- Sum of the first 10 terms -/
  sum10 : ℝ
  /-- The sequence is arithmetic -/
  is_arithmetic : True
  /-- The sum of the first 5 terms is 10 -/
  sum5_eq_10 : sum5 = 10
  /-- The sum of the first 10 terms is 50 -/
  sum10_eq_50 : sum10 = 50

/-- Theorem: The sum of the first 15 terms is 120 -/
theorem sum_15_terms (seq : ArithmeticSequence) : ∃ (sum15 : ℝ), sum15 = 120 := by
  sorry

end NUMINAMATH_CALUDE_sum_15_terms_l1396_139608


namespace NUMINAMATH_CALUDE_min_problems_for_45_points_l1396_139672

/-- Represents the possible point values for each problem -/
inductive PointValue
  | Three
  | Eight
  | Ten

/-- Represents a solution to the olympiad problem -/
structure Solution :=
  (threes : Nat)
  (eights : Nat)
  (tens : Nat)

/-- Calculates the total points for a given solution -/
def totalPoints (s : Solution) : Nat :=
  3 * s.threes + 8 * s.eights + 10 * s.tens

/-- Calculates the total number of problems solved for a given solution -/
def totalProblems (s : Solution) : Nat :=
  s.threes + s.eights + s.tens

/-- Defines a valid solution that achieves exactly 45 points -/
def isValidSolution (s : Solution) : Prop :=
  totalPoints s = 45

/-- Theorem stating that the minimum number of problems to achieve 45 points is 6 -/
theorem min_problems_for_45_points :
  ∃ (s : Solution), isValidSolution s ∧
  (∀ (s' : Solution), isValidSolution s' → totalProblems s ≤ totalProblems s') ∧
  totalProblems s = 6 :=
sorry

end NUMINAMATH_CALUDE_min_problems_for_45_points_l1396_139672


namespace NUMINAMATH_CALUDE_median_mode_difference_l1396_139656

def data : List ℕ := [21, 23, 23, 24, 24, 33, 33, 33, 33, 42, 42, 47, 48, 51, 52, 53, 54, 62, 67, 68]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem median_mode_difference (h : data.length = 20) : 
  |median data - (mode data : ℚ)| = 0 := by sorry

end NUMINAMATH_CALUDE_median_mode_difference_l1396_139656


namespace NUMINAMATH_CALUDE_crayon_theorem_l1396_139654

/-- The number of crayons the other friend has -/
def other_friend_crayons (lizzie_crayons : ℕ) : ℕ :=
  lizzie_crayons * 4 / 3

theorem crayon_theorem (lizzie_crayons : ℕ) 
  (h1 : lizzie_crayons = 27) : 
  other_friend_crayons lizzie_crayons = 18 :=
by
  sorry

#eval other_friend_crayons 27

end NUMINAMATH_CALUDE_crayon_theorem_l1396_139654


namespace NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l1396_139627

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
structure ArithmeticSequence (α : Type*) [Add α] where
  a : ℕ → α  -- The sequence
  d : α      -- The common difference
  h : ∀ n, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 3rd term is 3/8 and the 15th term is 7/9, 
    the 9th term is equal to 83/144. -/
theorem ninth_term_of_arithmetic_sequence 
  (seq : ArithmeticSequence ℚ) 
  (h3 : seq.a 3 = 3/8) 
  (h15 : seq.a 15 = 7/9) : 
  seq.a 9 = 83/144 := by
sorry

end NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l1396_139627


namespace NUMINAMATH_CALUDE_stating_dual_polyhedra_equal_spheres_l1396_139607

/-- Represents a regular polyhedron with its associated sphere radii -/
structure RegularPolyhedron where
  R : ℝ  -- radius of circumscribed sphere
  r : ℝ  -- radius of inscribed sphere
  p : ℝ  -- radius of half-inscribed sphere

/-- Represents a pair of dual regular polyhedra -/
structure DualPolyhedraPair where
  T1 : RegularPolyhedron
  T2 : RegularPolyhedron

/-- 
Theorem stating that for dual regular polyhedra with equal inscribed spheres,
their circumscribed spheres are also equal.
-/
theorem dual_polyhedra_equal_spheres (pair : DualPolyhedraPair) :
  pair.T1.r = pair.T2.r → pair.T1.R = pair.T2.R := by
  sorry


end NUMINAMATH_CALUDE_stating_dual_polyhedra_equal_spheres_l1396_139607


namespace NUMINAMATH_CALUDE_checkers_draw_fraction_l1396_139667

theorem checkers_draw_fraction (dan_wins eve_wins : ℚ) (h1 : dan_wins = 4/9) (h2 : eve_wins = 1/3) :
  1 - (dan_wins + eve_wins) = 2/9 := by
sorry

end NUMINAMATH_CALUDE_checkers_draw_fraction_l1396_139667


namespace NUMINAMATH_CALUDE_problem_solution_l1396_139642

def p (a : ℝ) : Prop := ∀ x ≥ 1, x - x^2 ≤ a

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a*x + 1 = 0

theorem problem_solution (a : ℝ) :
  (¬(¬(p a)) → a ≥ 0) ∧
  ((¬(p a ∧ q a) ∧ (p a ∨ q a)) → (a ≤ -2 ∨ (0 ≤ a ∧ a < 2))) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1396_139642


namespace NUMINAMATH_CALUDE_new_light_wattage_l1396_139622

theorem new_light_wattage (original_wattage : ℝ) (increase_percentage : ℝ) : 
  original_wattage = 110 → 
  increase_percentage = 30 → 
  original_wattage * (1 + increase_percentage / 100) = 143 := by
sorry

end NUMINAMATH_CALUDE_new_light_wattage_l1396_139622


namespace NUMINAMATH_CALUDE_real_world_length_l1396_139679

/-- Represents the scale factor of the model -/
def scale_factor : ℝ := 50

/-- Represents the length of the line segment in the model (in cm) -/
def model_length : ℝ := 7.5

/-- Theorem stating that the real-world length represented by the model line segment is 375 meters -/
theorem real_world_length : model_length * scale_factor = 375 := by
  sorry

end NUMINAMATH_CALUDE_real_world_length_l1396_139679


namespace NUMINAMATH_CALUDE_cars_meeting_time_l1396_139637

/-- Two cars traveling towards each other on a highway meet after a certain time -/
theorem cars_meeting_time (highway_length : ℝ) (speed1 speed2 : ℝ) (meeting_time : ℝ) :
  highway_length = 500 →
  speed1 = 40 →
  speed2 = 60 →
  meeting_time * (speed1 + speed2) = highway_length →
  meeting_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_cars_meeting_time_l1396_139637


namespace NUMINAMATH_CALUDE_september_to_august_ratio_l1396_139652

def july_earnings : ℕ := 150
def august_earnings : ℕ := 3 * july_earnings
def total_earnings : ℕ := 1500

def september_earnings_ratio (x : ℚ) : Prop :=
  july_earnings + august_earnings + x * august_earnings = total_earnings

theorem september_to_august_ratio :
  ∃ x : ℚ, september_earnings_ratio x ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_september_to_august_ratio_l1396_139652


namespace NUMINAMATH_CALUDE_sequence_repeat_value_l1396_139675

theorem sequence_repeat_value (p q n : ℕ+) (x : Fin (n + 1) → ℤ)
  (h1 : p + q < n)
  (h2 : x 0 = 0 ∧ x n = 0)
  (h3 : ∀ i : Fin n, x (i + 1) - x i = p ∨ x (i + 1) - x i = -q) :
  ∃ i j : Fin (n + 1), i < j ∧ (i, j) ≠ (0, n) ∧ x i = x j := by
  sorry

end NUMINAMATH_CALUDE_sequence_repeat_value_l1396_139675


namespace NUMINAMATH_CALUDE_line_intersection_bound_l1396_139630

/-- Given points A(2,7) and B(9,6) in the Cartesian plane, and a line y = kx (k ≠ 0) that
    intersects the line segment AB, prove that k is bounded by 2/3 ≤ k ≤ 7/2. -/
theorem line_intersection_bound (k : ℝ) : k ≠ 0 → 
  (∃ x y : ℝ, x ∈ Set.Icc 2 9 ∧ y ∈ Set.Icc 6 7 ∧ y = k * x ∧ y - 7 = (6 - 7) / (9 - 2) * (x - 2)) →
  2/3 ≤ k ∧ k ≤ 7/2 := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_bound_l1396_139630


namespace NUMINAMATH_CALUDE_line_perp_to_plane_l1396_139699

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and between a line and a plane
variable (perp : Line → Line → Prop)
variable (perpToPlane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

-- Define the intersection operation for lines
variable (intersect : Line → Line → Set Point)

-- Theorem statement
theorem line_perp_to_plane 
  (a b c : Line) (α : Plane) (A : Point) :
  perp c a → 
  perp c b → 
  subset a α → 
  subset b α → 
  intersect a b = {A} → 
  perpToPlane c α :=
sorry

end NUMINAMATH_CALUDE_line_perp_to_plane_l1396_139699


namespace NUMINAMATH_CALUDE_orange_bin_theorem_l1396_139698

/-- Calculates the final number of oranges in a bin after changes. -/
def final_oranges (initial : ℕ) (thrown_away : ℕ) (added : ℕ) : ℕ :=
  initial - thrown_away + added

/-- Proves that the final number of oranges is correct given the initial conditions. -/
theorem orange_bin_theorem (initial : ℕ) (thrown_away : ℕ) (added : ℕ) :
  final_oranges initial thrown_away added = initial - thrown_away + added :=
by sorry

end NUMINAMATH_CALUDE_orange_bin_theorem_l1396_139698


namespace NUMINAMATH_CALUDE_ducks_in_lake_l1396_139669

theorem ducks_in_lake (initial_ducks joining_ducks : ℕ) 
  (h1 : initial_ducks = 13)
  (h2 : joining_ducks = 20) : 
  initial_ducks + joining_ducks = 33 := by
sorry

end NUMINAMATH_CALUDE_ducks_in_lake_l1396_139669


namespace NUMINAMATH_CALUDE_units_digit_of_17_to_1995_l1396_139692

theorem units_digit_of_17_to_1995 : (17 ^ 1995 : ℕ) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_17_to_1995_l1396_139692


namespace NUMINAMATH_CALUDE_students_not_in_biology_l1396_139635

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) 
  (h1 : total_students = 880)
  (h2 : biology_percentage = 40 / 100) :
  ↑total_students * (1 - biology_percentage) = 528 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l1396_139635


namespace NUMINAMATH_CALUDE_square_sum_equals_six_l1396_139606

theorem square_sum_equals_six (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : 
  x^2 + y^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_six_l1396_139606


namespace NUMINAMATH_CALUDE_root_quadratic_equation_l1396_139621

theorem root_quadratic_equation (m : ℝ) : 
  m^2 - m - 110 = 0 → (m - 1)^2 + m = 111 := by
  sorry

end NUMINAMATH_CALUDE_root_quadratic_equation_l1396_139621


namespace NUMINAMATH_CALUDE_investment_rate_proof_l1396_139662

theorem investment_rate_proof (total_investment : ℝ) (first_investment : ℝ) (second_investment : ℝ)
  (first_rate : ℝ) (second_rate : ℝ) (desired_income : ℝ) :
  total_investment = 12000 ∧
  first_investment = 5000 ∧
  second_investment = 4000 ∧
  first_rate = 0.05 ∧
  second_rate = 0.035 ∧
  desired_income = 600 →
  ∃ (remaining_rate : ℝ),
    remaining_rate = 0.07 ∧
    (total_investment - first_investment - second_investment) * remaining_rate +
    first_investment * first_rate + second_investment * second_rate = desired_income :=
by sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l1396_139662


namespace NUMINAMATH_CALUDE_vector_equality_implies_x_value_l1396_139636

/-- Given that the vector (x+3, x^2-3x-4) is equal to (2, 0), prove that x = -1 -/
theorem vector_equality_implies_x_value : 
  ∀ x : ℝ, (x + 3 = 2 ∧ x^2 - 3*x - 4 = 0) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_implies_x_value_l1396_139636


namespace NUMINAMATH_CALUDE_composition_value_l1396_139643

/-- Given two functions g and h, prove that their composition at x = 2 equals 3890 -/
theorem composition_value :
  let g (x : ℝ) := 3 * x^2 + 2
  let h (x : ℝ) := -5 * x^3 + 4
  g (h 2) = 3890 := by
  sorry

end NUMINAMATH_CALUDE_composition_value_l1396_139643


namespace NUMINAMATH_CALUDE_solve_equation_l1396_139683

theorem solve_equation (x : ℚ) : (3 * x - 2) / 4 = 14 → x = 58 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1396_139683


namespace NUMINAMATH_CALUDE_square_root_squared_l1396_139626

theorem square_root_squared (x : ℝ) : (Real.sqrt x)^2 = 49 → x = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_root_squared_l1396_139626


namespace NUMINAMATH_CALUDE_sum_even_digits_1_to_200_l1396_139694

/-- E(n) represents the sum of even digits in the number n -/
def E (n : ℕ) : ℕ := sorry

/-- The sum of E(n) for n from 1 to 200 -/
def sumE : ℕ := (Finset.range 200).sum E + E 200

theorem sum_even_digits_1_to_200 : sumE = 800 := by sorry

end NUMINAMATH_CALUDE_sum_even_digits_1_to_200_l1396_139694


namespace NUMINAMATH_CALUDE_factorization_identities_l1396_139614

theorem factorization_identities (x y m n p : ℝ) : 
  (x^2 + 2*x + 1 - y^2 = (x + y + 1)*(x - y + 1)) ∧ 
  (m^2 - n^2 - 2*n*p - p^2 = (m + n + p)*(m - n - p)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identities_l1396_139614


namespace NUMINAMATH_CALUDE_sum_remainder_theorem_l1396_139618

/-- Calculates the sum S as defined in the problem -/
def calculate_sum : ℚ := sorry

/-- Finds the closest natural number to a given rational number -/
def closest_natural (q : ℚ) : ℕ := sorry

/-- The main theorem stating that the remainder when the closest natural number
    to the sum S is divided by 5 is equal to 4 -/
theorem sum_remainder_theorem : 
  (closest_natural calculate_sum) % 5 = 4 := by sorry

end NUMINAMATH_CALUDE_sum_remainder_theorem_l1396_139618


namespace NUMINAMATH_CALUDE_sqrt_inequalities_l1396_139649

theorem sqrt_inequalities (x : ℝ) :
  (∀ x, (Real.sqrt (x - 1) < 1) ↔ (1 ≤ x ∧ x < 2)) ∧
  (∀ x, (Real.sqrt (2*x - 3) ≤ Real.sqrt (x - 1)) ↔ ((3/2) ≤ x ∧ x ≤ 2)) := by
sorry

end NUMINAMATH_CALUDE_sqrt_inequalities_l1396_139649


namespace NUMINAMATH_CALUDE_stating_cube_coloring_theorem_l1396_139625

/-- Represents the number of faces on a cube -/
def num_faces : ℕ := 6

/-- Represents the number of available colors -/
def num_colors : ℕ := 8

/-- Represents the number of rotational symmetries of a cube -/
def cube_symmetries : ℕ := 24

/-- 
Calculates the number of distinguishable ways to paint a cube
given the number of faces, colors, and rotational symmetries
-/
def distinguishable_colorings (faces : ℕ) (colors : ℕ) (symmetries : ℕ) : ℕ :=
  faces * (Nat.factorial (colors - 1)) / symmetries

/-- 
Theorem stating that the number of distinguishable ways to paint a cube
with 8 different colors, where each face is painted a different color, is 1260
-/
theorem cube_coloring_theorem : 
  distinguishable_colorings num_faces num_colors cube_symmetries = 1260 := by
  sorry

end NUMINAMATH_CALUDE_stating_cube_coloring_theorem_l1396_139625


namespace NUMINAMATH_CALUDE_odd_function_properties_l1396_139670

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + (a - 1)*x^2 + a*x + b

-- Define the property of f being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Main theorem
theorem odd_function_properties (a b : ℝ) 
  (h : is_odd_function (f a b)) : 
  (a + b = 1) ∧ 
  (∃ m c : ℝ, m = 4 ∧ c = -2 ∧ 
    ∀ x y : ℝ, y = f a b x → (y - f a b 1 = m * (x - 1) ↔ m*x - y + c = 0)) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_properties_l1396_139670


namespace NUMINAMATH_CALUDE_min_value_xy_l1396_139686

theorem min_value_xy (x y : ℝ) (h : x > 0 ∧ y > 0) (eq : 1/x + 2/y = Real.sqrt (x*y)) : 
  x * y ≥ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_xy_l1396_139686


namespace NUMINAMATH_CALUDE_fifteenth_term_is_negative_one_l1396_139651

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  first_term : ℤ
  common_diff : ℤ

/-- The nth term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.first_term + (n - 1 : ℤ) * seq.common_diff

theorem fifteenth_term_is_negative_one
  (seq : ArithmeticSequence)
  (h21 : nth_term seq 21 = 17)
  (h22 : nth_term seq 22 = 20) :
  nth_term seq 15 = -1 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_is_negative_one_l1396_139651


namespace NUMINAMATH_CALUDE_halfway_point_between_fractions_l1396_139678

theorem halfway_point_between_fractions :
  let a := (1 : ℚ) / 7
  let b := (1 : ℚ) / 9
  let midpoint := (a + b) / 2
  midpoint = 8 / 63 := by sorry

end NUMINAMATH_CALUDE_halfway_point_between_fractions_l1396_139678


namespace NUMINAMATH_CALUDE_equation_solution_l1396_139666

def solution_set : Set (ℕ × ℕ) :=
  {(0, 1), (1, 1), (3, 25), (4, 31), (5, 41), (8, 85)}

theorem equation_solution :
  {(a, b) : ℕ × ℕ | a * b + 2 = a ^ 3 + 2 * b} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1396_139666


namespace NUMINAMATH_CALUDE_prob_at_least_three_out_of_five_is_half_l1396_139658

def probability_at_least_three_out_of_five : ℚ :=
  let n : ℕ := 5  -- total number of games
  let p : ℚ := 1/2  -- probability of winning a single game
  let winning_prob : ℕ → ℚ := λ k => Nat.choose n k * p^k * (1-p)^(n-k)
  (winning_prob 3) + (winning_prob 4) + (winning_prob 5)

theorem prob_at_least_three_out_of_five_is_half :
  probability_at_least_three_out_of_five = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_three_out_of_five_is_half_l1396_139658


namespace NUMINAMATH_CALUDE_consecutive_numbers_divisible_by_2014_l1396_139688

theorem consecutive_numbers_divisible_by_2014 :
  ∃ (n : ℕ), n < 96 ∧ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 2014 = 0 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_divisible_by_2014_l1396_139688


namespace NUMINAMATH_CALUDE_concert_audience_fraction_l1396_139628

/-- The fraction of the audience for the second band at a concert -/
def fraction_second_band (total_audience : ℕ) (under_30_percent : ℚ) 
  (women_percent : ℚ) (men_under_30 : ℕ) : ℚ :=
  2 / 3

theorem concert_audience_fraction 
  (total_audience : ℕ) 
  (under_30_percent : ℚ) 
  (women_percent : ℚ) 
  (men_under_30 : ℕ) : 
  fraction_second_band total_audience under_30_percent women_percent men_under_30 = 2 / 3 :=
by
  sorry

#check concert_audience_fraction 150 (1/2) (3/5) 20

end NUMINAMATH_CALUDE_concert_audience_fraction_l1396_139628


namespace NUMINAMATH_CALUDE_cookie_distribution_l1396_139671

/-- The number of ways to distribute n identical items into k distinct groups -/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of friends -/
def numFriends : ℕ := 4

/-- The total number of cookies -/
def totalCookies : ℕ := 10

/-- The minimum number of cookies each friend must have -/
def minCookies : ℕ := 2

/-- The number of ways to distribute the cookies -/
def numWays : ℕ := starsAndBars (totalCookies - minCookies * numFriends) numFriends

theorem cookie_distribution :
  numWays = 10 := by sorry

end NUMINAMATH_CALUDE_cookie_distribution_l1396_139671


namespace NUMINAMATH_CALUDE_function_inequality_l1396_139687

theorem function_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x y, x < y ∧ y ≤ 2 → f x < f y) 
  (h2 : ∀ x, f (-x + 2) = f (x + 2)) : 
  f (-1) < f 3 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l1396_139687


namespace NUMINAMATH_CALUDE_books_sold_l1396_139632

theorem books_sold (initial_books : ℕ) (added_books : ℕ) (final_books : ℕ) : 
  initial_books = 4 → added_books = 10 → final_books = 11 → 
  ∃ (sold_books : ℕ), initial_books - sold_books + added_books = final_books ∧ sold_books = 3 :=
by sorry

end NUMINAMATH_CALUDE_books_sold_l1396_139632


namespace NUMINAMATH_CALUDE_problem_statement_l1396_139696

theorem problem_statement : |Real.sqrt 3 - 2| + 2 * Real.sin (60 * π / 180) - 2023^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1396_139696


namespace NUMINAMATH_CALUDE_debby_candy_l1396_139693

def initial_candy : ℕ → ℕ → ℕ
  | remaining, eaten => remaining + eaten

theorem debby_candy (remaining eaten : ℕ) 
  (h1 : remaining = 3) 
  (h2 : eaten = 9) : 
  initial_candy remaining eaten = 12 := by
  sorry

end NUMINAMATH_CALUDE_debby_candy_l1396_139693


namespace NUMINAMATH_CALUDE_square_roots_problem_l1396_139604

theorem square_roots_problem (x a b : ℝ) (hx : x > 0) 
  (h_roots : x = a^2 ∧ x = (a + b)^2) (h_sum : 2*a + b = 0) :
  (a = -2 → b = 4 ∧ x = 4) ∧
  (b = 6 → a = -3 ∧ x = 9) ∧
  (a^2*x + (a + b)^2*x = 8 → x = 2) := by
sorry

end NUMINAMATH_CALUDE_square_roots_problem_l1396_139604


namespace NUMINAMATH_CALUDE_factorization_problem_value_problem_l1396_139615

-- Problem 1
theorem factorization_problem (a : ℝ) : 
  a^3 - 3*a^2 - 4*a + 12 = (a - 3) * (a - 2) * (a + 2) := by sorry

-- Problem 2
theorem value_problem (m n : ℝ) (h1 : m + n = 5) (h2 : m - n = 1) : 
  m^2 - n^2 + 2*m - 2*n = 7 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_value_problem_l1396_139615


namespace NUMINAMATH_CALUDE_salem_poem_words_per_line_l1396_139601

/-- A poem with a given structure and word count -/
structure Poem where
  stanzas : ℕ
  lines_per_stanza : ℕ
  total_words : ℕ

/-- Calculate the number of words per line in a poem -/
def words_per_line (p : Poem) : ℕ :=
  p.total_words / (p.stanzas * p.lines_per_stanza)

/-- Theorem: Given a poem with 20 stanzas, 10 lines per stanza, and 1600 total words,
    the number of words per line is 8 -/
theorem salem_poem_words_per_line :
  let p : Poem := { stanzas := 20, lines_per_stanza := 10, total_words := 1600 }
  words_per_line p = 8 := by
  sorry

#check salem_poem_words_per_line

end NUMINAMATH_CALUDE_salem_poem_words_per_line_l1396_139601


namespace NUMINAMATH_CALUDE_hamburger_combinations_l1396_139661

/-- The number of available condiments -/
def num_condiments : ℕ := 8

/-- The number of choices for meat patties -/
def num_patty_choices : ℕ := 4

/-- The total number of hamburger combinations -/
def total_combinations : ℕ := 2^num_condiments * num_patty_choices

theorem hamburger_combinations :
  total_combinations = 1024 :=
by sorry

end NUMINAMATH_CALUDE_hamburger_combinations_l1396_139661


namespace NUMINAMATH_CALUDE_continued_fraction_value_l1396_139647

theorem continued_fraction_value : 
  ∃ y : ℝ, y = 3 + 5 / (1 + 5 / y) → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_value_l1396_139647


namespace NUMINAMATH_CALUDE_min_value_expression_l1396_139668

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * (a + b + c) + b * c = 4 - 2 * Real.sqrt 3) :
  2 * a + b + c ≥ 2 * Real.sqrt 3 - 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1396_139668


namespace NUMINAMATH_CALUDE_expand_polynomial_l1396_139690

theorem expand_polynomial (x : ℝ) : 
  (x - 3) * (x + 3) * (x^2 + 9) * (x - 1) = x^5 - x^4 - 81*x + 81 := by
sorry

end NUMINAMATH_CALUDE_expand_polynomial_l1396_139690


namespace NUMINAMATH_CALUDE_total_cars_is_seventeen_l1396_139612

/-- The number of cars Tommy has -/
def tommy_cars : ℕ := 3

/-- The number of cars Jessie has -/
def jessie_cars : ℕ := 3

/-- The number of additional cars Jessie's older brother has compared to Tommy and Jessie combined -/
def brother_additional_cars : ℕ := 5

/-- The total number of cars for all three of them -/
def total_cars : ℕ := tommy_cars + jessie_cars + (tommy_cars + jessie_cars + brother_additional_cars)

theorem total_cars_is_seventeen : total_cars = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_is_seventeen_l1396_139612


namespace NUMINAMATH_CALUDE_biancas_books_l1396_139673

/-- The number of coloring books Bianca has after giving some away and buying more -/
def final_book_count (initial : ℕ) (given_away : ℕ) (bought : ℕ) : ℕ :=
  initial - given_away + bought

/-- Theorem stating that Bianca's final book count is 59 -/
theorem biancas_books : final_book_count 45 6 20 = 59 := by
  sorry

end NUMINAMATH_CALUDE_biancas_books_l1396_139673


namespace NUMINAMATH_CALUDE_surface_area_of_specific_cut_tetrahedron_l1396_139620

/-- Represents a right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  base_side_length : ℝ

/-- Represents the tetrahedron formed by cutting the prism -/
structure CutTetrahedron where
  prism : RightPrism

/-- Calculate the surface area of the cut tetrahedron -/
noncomputable def surface_area (tetra : CutTetrahedron) : ℝ :=
  sorry

/-- Theorem statement for the surface area of the specific cut tetrahedron -/
theorem surface_area_of_specific_cut_tetrahedron :
  let prism := RightPrism.mk 20 10
  let tetra := CutTetrahedron.mk prism
  surface_area tetra = 50 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 118.75) / 2 :=
by sorry

end NUMINAMATH_CALUDE_surface_area_of_specific_cut_tetrahedron_l1396_139620


namespace NUMINAMATH_CALUDE_pie_rows_theorem_l1396_139684

def pecan_pies : ℕ := 16
def apple_pies : ℕ := 14
def pies_per_row : ℕ := 5

theorem pie_rows_theorem : 
  (pecan_pies + apple_pies) / pies_per_row = 6 := by
  sorry

end NUMINAMATH_CALUDE_pie_rows_theorem_l1396_139684


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1396_139639

theorem sufficient_but_not_necessary (a : ℝ) :
  (∀ x : ℝ, x > 1 → x ≠ 1) ∧ (∃ y : ℝ, y ≠ 1 ∧ ¬(y > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1396_139639


namespace NUMINAMATH_CALUDE_walts_age_inconsistency_l1396_139645

theorem walts_age_inconsistency :
  ¬ ∃ (w : ℕ), 
    (3 * w + 12 = 2 * (w + 12)) ∧ 
    (4 * w + 15 = 3 * (w + 15)) := by
  sorry

end NUMINAMATH_CALUDE_walts_age_inconsistency_l1396_139645


namespace NUMINAMATH_CALUDE_power_sum_constant_implies_zero_or_one_l1396_139677

/-- Given a natural number n > 1 and a list of real numbers x,
    if the sum of the k-th powers of these numbers is constant for k from 1 to n+1,
    then each number in the list is either 0 or 1. -/
theorem power_sum_constant_implies_zero_or_one (n : ℕ) (x : List ℝ) :
  n > 1 →
  x.length = n →
  (∀ k : ℕ, k ≥ 1 → k ≤ n + 1 →
    (List.sum (List.map (fun xi => xi ^ k) x)) = (List.sum (List.map (fun xi => xi ^ 1) x))) →
  ∀ xi ∈ x, xi = 0 ∨ xi = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_constant_implies_zero_or_one_l1396_139677


namespace NUMINAMATH_CALUDE_walkway_problem_l1396_139603

/-- Represents the scenario of a person walking on a moving walkway -/
structure WalkwayScenario where
  length : ℝ
  time_with : ℝ
  time_against : ℝ

/-- Calculates the time taken to walk when the walkway is stopped -/
def time_when_stopped (scenario : WalkwayScenario) : ℝ :=
  -- The actual calculation is not provided here
  sorry

/-- Theorem stating the correct time when the walkway is stopped -/
theorem walkway_problem (scenario : WalkwayScenario) 
  (h1 : scenario.length = 80)
  (h2 : scenario.time_with = 40)
  (h3 : scenario.time_against = 120) :
  time_when_stopped scenario = 60 := by
  sorry

end NUMINAMATH_CALUDE_walkway_problem_l1396_139603


namespace NUMINAMATH_CALUDE_smallest_angle_in_characteristic_triangle_l1396_139617

/-- A characteristic triangle is a triangle where one interior angle is twice another. -/
structure CharacteristicTriangle where
  α : ℝ  -- The larger angle (characteristic angle)
  β : ℝ  -- The smaller angle
  γ : ℝ  -- The third angle
  angle_sum : α + β + γ = 180
  characteristic : α = 2 * β

/-- The smallest angle in a characteristic triangle with characteristic angle 100° is 30°. -/
theorem smallest_angle_in_characteristic_triangle :
  ∀ (t : CharacteristicTriangle), t.α = 100 → min t.α (min t.β t.γ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_in_characteristic_triangle_l1396_139617


namespace NUMINAMATH_CALUDE_consecutive_numbers_lcm_660_l1396_139648

theorem consecutive_numbers_lcm_660 (x : ℕ) :
  (Nat.lcm x (Nat.lcm (x + 1) (x + 2)) = 660) →
  x = 10 ∧ (x + 1) = 11 ∧ (x + 2) = 12 := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_lcm_660_l1396_139648


namespace NUMINAMATH_CALUDE_arccos_gt_twice_arcsin_l1396_139613

theorem arccos_gt_twice_arcsin (x : ℝ) : 
  -1 ≤ x ∧ x ≤ 1 → (Real.arccos x > 2 * Real.arcsin x ↔ -1 < x ∧ x ≤ (1 - Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_arccos_gt_twice_arcsin_l1396_139613


namespace NUMINAMATH_CALUDE_square_implies_four_right_angles_but_not_conversely_l1396_139602

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a square
def is_square (q : Quadrilateral) : Prop :=
  -- A square has four equal sides and four right angles
  sorry

-- Define a quadrilateral with four right angles
def has_four_right_angles (q : Quadrilateral) : Prop :=
  -- A quadrilateral has four right angles
  sorry

-- Theorem statement
theorem square_implies_four_right_angles_but_not_conversely :
  (∀ q : Quadrilateral, is_square q → has_four_right_angles q) ∧
  (∃ q : Quadrilateral, has_four_right_angles q ∧ ¬is_square q) :=
sorry

end NUMINAMATH_CALUDE_square_implies_four_right_angles_but_not_conversely_l1396_139602


namespace NUMINAMATH_CALUDE_sum_and_count_equals_431_l1396_139638

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_equals_431 : 
  sum_of_integers 10 30 + count_even_integers 10 30 = 431 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_equals_431_l1396_139638


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_representation_l1396_139685

theorem quadratic_form_ratio_representation (x y u v : ℤ) :
  (∃ k : ℤ, (x^2 + 3*y^2) = k * (u^2 + 3*v^2)) →
  ∃ a b : ℤ, (x^2 + 3*y^2) / (u^2 + 3*v^2) = a^2 + 3*b^2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_representation_l1396_139685
