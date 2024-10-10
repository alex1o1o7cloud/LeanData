import Mathlib

namespace largest_divisor_of_n_l1183_118326

theorem largest_divisor_of_n (n : ℕ+) (h : 100 ∣ n^3) :
  100 = Nat.gcd 100 n ∧ ∀ m : ℕ, m > 100 → ¬(m ∣ n) :=
sorry

end largest_divisor_of_n_l1183_118326


namespace smallest_prime_is_two_l1183_118321

theorem smallest_prime_is_two (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  p ≠ q → p ≠ r → q ≠ r →
  p^3 + q^3 + 3*p*q*r = r^3 →
  min p (min q r) = 2 := by
sorry

end smallest_prime_is_two_l1183_118321


namespace min_ceiling_sum_squares_l1183_118391

theorem min_ceiling_sum_squares (A B C D E F G H I J K L M N O P Q R S T U V W X Y Z : ℝ) 
  (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0) (hE : E ≠ 0) (hF : F ≠ 0) 
  (hG : G ≠ 0) (hH : H ≠ 0) (hI : I ≠ 0) (hJ : J ≠ 0) (hK : K ≠ 0) (hL : L ≠ 0) 
  (hM : M ≠ 0) (hN : N ≠ 0) (hO : O ≠ 0) (hP : P ≠ 0) (hQ : Q ≠ 0) (hR : R ≠ 0) 
  (hS : S ≠ 0) (hT : T ≠ 0) (hU : U ≠ 0) (hV : V ≠ 0) (hW : W ≠ 0) (hX : X ≠ 0) 
  (hY : Y ≠ 0) (hZ : Z ≠ 0) : 
  26 = ⌈(A^2 + B^2 + C^2 + D^2 + E^2 + F^2 + G^2 + H^2 + I^2 + J^2 + K^2 + L^2 + 
         M^2 + N^2 + O^2 + P^2 + Q^2 + R^2 + S^2 + T^2 + U^2 + V^2 + W^2 + X^2 + Y^2 + Z^2)⌉ :=
by sorry

end min_ceiling_sum_squares_l1183_118391


namespace function_decomposition_l1183_118395

theorem function_decomposition (f : ℝ → ℝ) :
  ∃ (g h : ℝ → ℝ), 
    (∀ x, g (-x) = g x) ∧ 
    (∀ x, h (-x) = -h x) ∧ 
    (∀ x, f x = g x + h x) := by
  sorry

end function_decomposition_l1183_118395


namespace amithab_january_expenditure_l1183_118357

/-- Amithab's monthly expenditure problem -/
theorem amithab_january_expenditure
  (avg_jan_to_jun : ℝ)
  (july_expenditure : ℝ)
  (avg_feb_to_jul : ℝ)
  (h1 : avg_jan_to_jun = 4200)
  (h2 : july_expenditure = 1500)
  (h3 : avg_feb_to_jul = 4250) :
  6 * avg_jan_to_jun + july_expenditure = 6 * avg_feb_to_jul + 1800 :=
by sorry

end amithab_january_expenditure_l1183_118357


namespace lcm_gcd_product_l1183_118396

theorem lcm_gcd_product (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h_lcm : Nat.lcm a b = 60) (h_gcd : Nat.gcd a b = 5) : 
  a * b = 300 := by
  sorry

end lcm_gcd_product_l1183_118396


namespace sum_parity_of_nine_consecutive_naturals_l1183_118390

theorem sum_parity_of_nine_consecutive_naturals (n : ℕ) :
  Even (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8)) ↔ Even n :=
by sorry

end sum_parity_of_nine_consecutive_naturals_l1183_118390


namespace sum_of_fourth_powers_l1183_118394

theorem sum_of_fourth_powers (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 1) : x^4 + y^4 = 2 := by
  sorry

end sum_of_fourth_powers_l1183_118394


namespace quadratic_equation_solution_l1183_118353

theorem quadratic_equation_solution (x : ℝ) 
  (eq : 2 * x^2 = 9 * x - 4) 
  (neq : x ≠ 4) : 
  2 * x = 1 := by
sorry

end quadratic_equation_solution_l1183_118353


namespace triangle_angle_proof_l1183_118329

theorem triangle_angle_proof (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  -- Side-angle relationship
  c * Real.sin A = a * Real.cos C →
  -- Conclusion
  C = π / 4 := by
sorry

end triangle_angle_proof_l1183_118329


namespace vendor_drink_problem_l1183_118398

theorem vendor_drink_problem (maaza sprite cans : ℕ) (pepsi : ℕ) : 
  maaza = 50 →
  sprite = 368 →
  cans = 281 →
  (maaza + sprite + pepsi) % cans = 0 →
  pepsi = 144 :=
by sorry

end vendor_drink_problem_l1183_118398


namespace drum_capacity_ratio_l1183_118304

theorem drum_capacity_ratio :
  ∀ (C_X C_Y : ℝ),
  C_X > 0 → C_Y > 0 →
  (1/2 * C_X) + (1/4 * C_Y) = 1/2 * C_Y →
  C_Y / C_X = 2 := by
sorry

end drum_capacity_ratio_l1183_118304


namespace probability_one_girl_two_boys_l1183_118373

/-- The probability of having a boy or a girl for each child -/
def child_probability : ℝ := 0.5

/-- The number of children in the family -/
def num_children : ℕ := 3

/-- The number of ways to arrange 1 girl and 2 boys in 3 positions -/
def num_arrangements : ℕ := 3

/-- Theorem: The probability of having exactly 1 girl and 2 boys in a family with 3 children,
    where each child has an equal probability of being a boy or a girl, is 0.375 -/
theorem probability_one_girl_two_boys :
  (child_probability ^ num_children) * num_arrangements = 0.375 := by
  sorry

end probability_one_girl_two_boys_l1183_118373


namespace river_road_cars_l1183_118336

theorem river_road_cars (buses cars : ℕ) : 
  buses * 10 = cars ∧ cars - buses = 90 → cars = 100 := by
  sorry

end river_road_cars_l1183_118336


namespace power_problem_l1183_118375

theorem power_problem (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3*a + b) = 24 := by
  sorry

end power_problem_l1183_118375


namespace seventeen_flavors_l1183_118366

/-- Represents the number of different flavors possible given blue and orange candies. -/
def number_of_flavors (blue : ℕ) (orange : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that given 5 blue candies and 4 orange candies, 
    the number of different possible flavors is 17. -/
theorem seventeen_flavors : number_of_flavors 5 4 = 17 := by
  sorry

end seventeen_flavors_l1183_118366


namespace third_smallest_sum_is_four_l1183_118369

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (n / 10) % 10 = 1

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + (n % 10)

theorem third_smallest_sum_is_four :
  ∃ (n : ℕ), is_valid_number n ∧
  (∀ m, is_valid_number m → m < n) ∧
  (∃ k₁ k₂, is_valid_number k₁ ∧ is_valid_number k₂ ∧ k₁ < k₂ ∧ k₂ < n) ∧
  digit_sum n = 4 :=
sorry

end third_smallest_sum_is_four_l1183_118369


namespace smallest_four_digit_prime_divisible_proof_l1183_118388

def smallest_four_digit_prime_divisible : ℕ := 2310

theorem smallest_four_digit_prime_divisible_proof :
  (smallest_four_digit_prime_divisible ≥ 1000) ∧
  (smallest_four_digit_prime_divisible < 10000) ∧
  (smallest_four_digit_prime_divisible % 2 = 0) ∧
  (smallest_four_digit_prime_divisible % 3 = 0) ∧
  (smallest_four_digit_prime_divisible % 5 = 0) ∧
  (smallest_four_digit_prime_divisible % 7 = 0) ∧
  (smallest_four_digit_prime_divisible % 11 = 0) ∧
  (∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 → n ≥ smallest_four_digit_prime_divisible) :=
by sorry

end smallest_four_digit_prime_divisible_proof_l1183_118388


namespace coefficient_x_squared_in_expansion_l1183_118399

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℤ := sorry

-- Define the expansion of (x-1)^6
def expansion_x_minus_1_power_6 (r : ℕ) : ℤ := binomial 6 r * (-1)^r

-- Theorem statement
theorem coefficient_x_squared_in_expansion :
  expansion_x_minus_1_power_6 3 = -20 := by sorry

end coefficient_x_squared_in_expansion_l1183_118399


namespace quadratic_inequality_transformation_l1183_118301

theorem quadratic_inequality_transformation (a b c : ℝ) :
  (∀ x, 1 < x ∧ x < 2 → a * x^2 + b * x + c > 0) →
  (∀ x, c * x^2 + b * x + a > 0 ↔ 1/2 < x ∧ x < 1) :=
by sorry

end quadratic_inequality_transformation_l1183_118301


namespace road_repair_workers_l1183_118364

theorem road_repair_workers (group1_people : ℕ) (group1_days : ℕ) (group1_hours : ℕ)
                             (group2_days : ℕ) (group2_hours : ℕ) :
  group1_people = 69 →
  group1_days = 12 →
  group1_hours = 5 →
  group2_days = 23 →
  group2_hours = 6 →
  group1_people * group1_days * group1_hours = 
    ((group1_people * group1_days * group1_hours) / (group2_days * group2_hours) : ℕ) * group2_days * group2_hours →
  ((group1_people * group1_days * group1_hours) / (group2_days * group2_hours) : ℕ) = 30 :=
by sorry

end road_repair_workers_l1183_118364


namespace square_area_is_169_l1183_118365

/-- Square with intersecting segments --/
structure SquareWithIntersection where
  -- Side length of the square
  s : ℝ
  -- Length of BR
  br : ℝ
  -- Length of PR
  pr : ℝ
  -- Length of CQ
  cq : ℝ
  -- Conditions
  br_positive : br > 0
  pr_positive : pr > 0
  cq_positive : cq > 0
  right_angle : True  -- Represents that BP and CQ intersect at right angles
  br_eq : br = 8
  pr_eq : pr = 5
  cq_eq : cq = 12

/-- The area of the square is 169 --/
theorem square_area_is_169 (square : SquareWithIntersection) : square.s^2 = 169 := by
  sorry

end square_area_is_169_l1183_118365


namespace comprehensive_survey_suitability_l1183_118334

/-- Represents a survey scenario --/
inductive SurveyScenario
  | CalculatorServiceLife
  | BeijingStudentsSpaceflightLogo
  | ClassmatesBadalingGreatWall
  | FoodPigmentContent

/-- Determines if a survey scenario is suitable for a comprehensive survey --/
def isSuitableForComprehensiveSurvey (scenario : SurveyScenario) : Prop :=
  match scenario with
  | SurveyScenario.ClassmatesBadalingGreatWall => True
  | _ => False

/-- Theorem stating that the ClassmatesBadalingGreatWall scenario is the only one suitable for a comprehensive survey --/
theorem comprehensive_survey_suitability :
  ∀ (scenario : SurveyScenario),
    isSuitableForComprehensiveSurvey scenario ↔ scenario = SurveyScenario.ClassmatesBadalingGreatWall :=
by
  sorry

#check comprehensive_survey_suitability

end comprehensive_survey_suitability_l1183_118334


namespace system_solution_l1183_118393

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(1, 5, 2, 3), (1, 5, 3, 2), (5, 1, 2, 3), (5, 1, 3, 2), (2, 2, 2, 2),
   (2, 3, 1, 5), (2, 3, 5, 1), (3, 2, 1, 5), (3, 2, 5, 1)}

theorem system_solution (a b c d : ℕ) :
  (a * b = c + d ∧ c * d = a + b) ↔ (a, b, c, d) ∈ solution_set := by
  sorry

#check system_solution

end system_solution_l1183_118393


namespace intersection_of_P_and_Q_l1183_118354

def P : Set ℝ := {0, 1, 2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = 3^x}

theorem intersection_of_P_and_Q : P ∩ Q = {1} := by sorry

end intersection_of_P_and_Q_l1183_118354


namespace min_shots_13x13_grid_l1183_118315

/-- Represents a grid with side length n -/
def Grid (n : ℕ) := Fin n × Fin n

/-- The set of possible moves for the target -/
def neighborMoves : List (ℤ × ℤ) :=
  [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

/-- Check if a move is valid within the grid -/
def isValidMove (n : ℕ) (pos : Grid n) (move : ℤ × ℤ) : Bool :=
  let (x, y) := pos
  let (dx, dy) := move
  0 ≤ x.val + dx ∧ x.val + dx < n ∧ 0 ≤ y.val + dy ∧ y.val + dy < n

/-- The minimum number of shots required to guarantee hitting the target twice -/
def minShotsToDestroy (n : ℕ) : ℕ :=
  n * n + (n * n + 1) / 2

/-- Theorem stating the minimum number of shots required for a 13x13 grid -/
theorem min_shots_13x13_grid :
  minShotsToDestroy 13 = 254 :=
sorry

end min_shots_13x13_grid_l1183_118315


namespace fraction_simplification_l1183_118351

theorem fraction_simplification (x : ℝ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (18 - 13 * x) / 12 := by
  sorry

end fraction_simplification_l1183_118351


namespace pizza_toppings_l1183_118363

/-- Given a pizza with the following properties:
  * Total slices: 16
  * Slices with pepperoni: 8
  * Slices with mushrooms: 12
  * Plain slices: 2
  Prove that the number of slices with both pepperoni and mushrooms is 6. -/
theorem pizza_toppings (total : Nat) (pepperoni : Nat) (mushrooms : Nat) (plain : Nat)
    (h_total : total = 16)
    (h_pepperoni : pepperoni = 8)
    (h_mushrooms : mushrooms = 12)
    (h_plain : plain = 2) :
    ∃ (both : Nat), both = 6 ∧
      pepperoni + mushrooms - both = total - plain :=
by sorry

end pizza_toppings_l1183_118363


namespace perpendicular_bisector_equation_l1183_118362

/-- The perpendicular bisector of a line segment connecting two points -/
theorem perpendicular_bisector_equation (A B : ℝ × ℝ) :
  let M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let m_AB : ℝ := (B.2 - A.2) / (B.1 - A.1)
  let m_perp : ℝ := -1 / m_AB
  A = (0, 1) →
  B = (4, 3) →
  (λ (x y : ℝ) => 2 * x + y - 6 = 0) =
    (λ (x y : ℝ) => y - M.2 = m_perp * (x - M.1)) :=
by sorry

end perpendicular_bisector_equation_l1183_118362


namespace closest_multiple_of_15_to_2023_l1183_118302

theorem closest_multiple_of_15_to_2023 :
  ∀ k : ℤ, |15 * k - 2023| ≥ |2025 - 2023| := by
  sorry

end closest_multiple_of_15_to_2023_l1183_118302


namespace largest_integer_in_interval_l1183_118389

theorem largest_integer_in_interval : 
  ∃ (x : ℤ), (1/4 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 11/15 ∧ 
  ∀ (y : ℤ), ((1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 11/15) → y ≤ x :=
by sorry

end largest_integer_in_interval_l1183_118389


namespace min_value_of_f_on_interval_l1183_118314

-- Define the function f(x) = x^3 - 12x
def f (x : ℝ) : ℝ := x^3 - 12*x

-- Define the interval [-3, 1]
def interval : Set ℝ := Set.Icc (-3) 1

-- Theorem statement
theorem min_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ interval ∧ f x = -11 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x :=
sorry

end min_value_of_f_on_interval_l1183_118314


namespace angle_sum_at_point_l1183_118367

theorem angle_sum_at_point (x y : ℝ) : 
  3 * x + 6 * x + (x + y) + 4 * y = 360 → x = 0 ∧ y = 72 := by
  sorry

end angle_sum_at_point_l1183_118367


namespace focal_chord_length_l1183_118323

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  parabola_eq : y^2 = 4*x

/-- Represents a line passing through the focal point of the parabola -/
structure FocalLine where
  A : ParabolaPoint
  B : ParabolaPoint
  sum_x : A.x + B.x = 6

/-- Theorem: The length of AB is 8 for the given conditions -/
theorem focal_chord_length (line : FocalLine) : 
  Real.sqrt ((line.B.x - line.A.x)^2 + (line.B.y - line.A.y)^2) = 8 := by
  sorry

end focal_chord_length_l1183_118323


namespace min_value_expression_l1183_118317

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (min : ℝ), min = 3 * Real.sqrt (5 / 13) ∧
  ∀ (x : ℝ), x = (Real.sqrt ((a^2 + 2*b^2) * (4*a^2 + b^2))) / (a * b) → x ≥ min :=
sorry

end min_value_expression_l1183_118317


namespace group_size_calculation_l1183_118313

theorem group_size_calculation (iceland : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : iceland = 55)
  (h2 : norway = 33)
  (h3 : both = 51)
  (h4 : neither = 53) :
  iceland + norway - both + neither = 90 := by
  sorry

end group_size_calculation_l1183_118313


namespace pencil_eraser_cost_l1183_118384

theorem pencil_eraser_cost :
  ∀ (p e : ℕ),
  15 * p + 5 * e = 125 →
  p > e →
  p > 0 →
  e > 0 →
  p + e = 11 :=
by
  sorry

end pencil_eraser_cost_l1183_118384


namespace r_value_when_n_is_3_l1183_118332

theorem r_value_when_n_is_3 (n m s r : ℕ) :
  m = 3 ∧ s = 2^n - m ∧ r = 3^s + s ∧ n = 3 → r = 248 := by
  sorry

end r_value_when_n_is_3_l1183_118332


namespace min_tenth_game_score_l1183_118330

/-- Represents the scores of a basketball player in a series of games -/
structure BasketballScores where
  first_five : ℝ  -- Total score of first 5 games
  sixth : ℝ
  seventh : ℝ
  eighth : ℝ
  ninth : ℝ
  tenth : ℝ

/-- Theorem stating the minimum score required for the 10th game -/
theorem min_tenth_game_score (scores : BasketballScores) 
  (h1 : scores.sixth = 23)
  (h2 : scores.seventh = 14)
  (h3 : scores.eighth = 11)
  (h4 : scores.ninth = 20)
  (h5 : (scores.first_five + scores.sixth + scores.seventh + scores.eighth + scores.ninth) / 9 > 
        scores.first_five / 5)
  (h6 : (scores.first_five + scores.sixth + scores.seventh + scores.eighth + scores.ninth + scores.tenth) / 10 > 18) :
  scores.tenth ≥ 29 := by
  sorry

end min_tenth_game_score_l1183_118330


namespace point_coordinates_wrt_origin_l1183_118350

/-- Given a point P(-3, -5) in the Cartesian coordinate system,
    its coordinates with respect to the origin are (3, 5). -/
theorem point_coordinates_wrt_origin :
  let P : ℝ × ℝ := (-3, -5)
  (|P.1|, |P.2|) = (3, 5) := by sorry

end point_coordinates_wrt_origin_l1183_118350


namespace lines_intersect_at_point_l1183_118359

/-- The first line parameterized by t -/
def line1 (t : ℚ) : ℚ × ℚ := (3 - t, 2 + 4*t)

/-- The second line parameterized by u -/
def line2 (u : ℚ) : ℚ × ℚ := (-1 + 3*u, 3 + 5*u)

/-- The proposed intersection point -/
def intersection_point : ℚ × ℚ := (39/17, 74/17)

theorem lines_intersect_at_point :
  ∃! (t u : ℚ), line1 t = line2 u ∧ line1 t = intersection_point :=
sorry

end lines_intersect_at_point_l1183_118359


namespace chicken_adventure_feathers_l1183_118387

/-- Calculates the number of feathers remaining after a chicken's thrill-seeking adventure. -/
def remaining_feathers (initial_feathers : ℕ) (cars_dodged : ℕ) : ℕ :=
  initial_feathers - 2 * cars_dodged

/-- Theorem stating the number of feathers remaining after the chicken's adventure. -/
theorem chicken_adventure_feathers :
  remaining_feathers 5263 23 = 5217 := by
  sorry

#eval remaining_feathers 5263 23

end chicken_adventure_feathers_l1183_118387


namespace square_area_from_perimeter_l1183_118325

theorem square_area_from_perimeter (p : ℝ) : 
  let perimeter : ℝ := 12 * p
  let side_length : ℝ := perimeter / 4
  let area : ℝ := side_length ^ 2
  area = 9 * p ^ 2 := by
sorry

end square_area_from_perimeter_l1183_118325


namespace line_increase_percentage_l1183_118360

/-- Given an increase of 450 lines resulting in a total of 1350 lines, 
    prove that the percentage increase is 50%. -/
theorem line_increase_percentage (increase : ℕ) (total : ℕ) : 
  increase = 450 → total = 1350 → (increase : ℚ) / ((total - increase) : ℚ) * 100 = 50 := by
  sorry

end line_increase_percentage_l1183_118360


namespace polynomial_coefficient_difference_l1183_118328

theorem polynomial_coefficient_difference (a b : ℝ) : 
  (∀ x, (1 + x) + (1 + x)^4 = 2 + 5*x + a*x^2 + b*x^3 + x^4) → 
  a - b = 2 := by
sorry

end polynomial_coefficient_difference_l1183_118328


namespace complex_inequality_l1183_118381

theorem complex_inequality (z : ℂ) (n : ℕ) (h1 : z.re ≥ 1) (h2 : n ≥ 4) :
  Complex.abs (z^(n+1) - 1) ≥ Complex.abs (z^n) * Complex.abs (z - 1) := by
  sorry

end complex_inequality_l1183_118381


namespace larger_number_l1183_118379

theorem larger_number (x y : ℝ) (h_sum : x + y = 40) (h_diff : x - y = 6) : x = 23 := by
  sorry

end larger_number_l1183_118379


namespace inequality_proof_l1183_118355

-- Define the set M
def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

-- State the theorem
theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : 
  (|(1/3 : ℝ) * a + (1/6 : ℝ) * b| < (1/4 : ℝ)) ∧ 
  (|1 - 4 * a * b| > 2 * |a - b|) := by
  sorry

end inequality_proof_l1183_118355


namespace sum_of_coefficients_l1183_118300

def polynomial (x : ℝ) : ℝ := 5 * (2 * x^8 - 3 * x^5 + 9) + 6 * (x^6 + 4 * x^3 - 6)

theorem sum_of_coefficients : (polynomial 1) = 34 := by
  sorry

end sum_of_coefficients_l1183_118300


namespace vector_on_line_l1183_118322

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- A line passing through two vectors p and q can be parameterized as p + t(q - p) for some real t -/
def line_through (p q : V) (t : ℝ) : V := p + t • (q - p)

/-- The theorem states that if m*p + 5/8*q lies on the line through p and q, then m = 3/8 -/
theorem vector_on_line (p q : V) (m : ℝ) 
  (h : ∃ t : ℝ, m • p + (5/8) • q = line_through p q t) : 
  m = 3/8 := by
sorry

end vector_on_line_l1183_118322


namespace probability_of_drawing_two_l1183_118386

def card_set : Finset ℕ := {1, 2, 2, 3, 5}

theorem probability_of_drawing_two (s : Finset ℕ := card_set) :
  (s.filter (· = 2)).card / s.card = 2 / 5 := by sorry

end probability_of_drawing_two_l1183_118386


namespace max_product_constraint_l1183_118349

theorem max_product_constraint (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_constraint : 5 * x + 8 * y + 3 * z = 90) :
  x * y * z ≤ 225 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
    5 * x₀ + 8 * y₀ + 3 * z₀ = 90 ∧ x₀ * y₀ * z₀ = 225 := by
  sorry

end max_product_constraint_l1183_118349


namespace pears_given_by_mike_l1183_118320

theorem pears_given_by_mike (initial_pears : ℕ) (pears_given_away : ℕ) (final_pears : ℕ) :
  initial_pears = 46 →
  pears_given_away = 47 →
  final_pears = 11 →
  pears_given_away - initial_pears + final_pears = 12 :=
by sorry

end pears_given_by_mike_l1183_118320


namespace statement_A_statement_C_statement_D_l1183_118368

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := (x - 1)^3 - a*x - b + 1

-- Define the function g
def g (a b x : ℝ) : ℝ := f a b x - 3*x + a*x + b

-- Statement A
theorem statement_A (a b : ℝ) :
  a = 3 → (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a b x = 0 ∧ f a b y = 0 ∧ f a b z = 0) →
  -4 < b ∧ b < 0 := by sorry

-- Statement C
theorem statement_C (a b m : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    ∃ k₁ k₂ k₃ : ℝ, 
      k₁ * (2 - x) + m = g a b x ∧
      k₂ * (2 - y) + m = g a b y ∧
      k₃ * (2 - z) + m = g a b z) →
  -5 < m ∧ m < -4 := by sorry

-- Statement D
theorem statement_D (a b : ℝ) :
  (∃ x₀ x₁ : ℝ, x₀ ≠ x₁ ∧
    (∀ x : ℝ, f a b x₀ ≤ f a b x ∨ f a b x₀ ≥ f a b x) ∧
    f a b x₀ = f a b x₁) →
  ∃ x₀ x₁ : ℝ, x₀ ≠ x₁ ∧ x₁ + 2*x₀ = 3 := by sorry

end statement_A_statement_C_statement_D_l1183_118368


namespace east_west_convention_l1183_118383

-- Define the direction type
inductive Direction
| West
| East

-- Define a function to convert distance and direction to a signed number
def signedDistance (dist : ℝ) (dir : Direction) : ℝ :=
  match dir with
  | Direction.West => dist
  | Direction.East => -dist

-- State the theorem
theorem east_west_convention (westDistance : ℝ) (eastDistance : ℝ) :
  westDistance > 0 →
  signedDistance westDistance Direction.West = westDistance →
  signedDistance eastDistance Direction.East = -eastDistance :=
by
  sorry

-- Example with the given values
example : signedDistance 3 Direction.East = -3 :=
by
  sorry

end east_west_convention_l1183_118383


namespace greek_cross_dissection_l1183_118345

/-- Represents a Greek cross -/
structure GreekCross where
  area : ℝ
  squares : Fin 5 → Square

/-- Represents a square piece of a Greek cross -/
structure Square where
  side_length : ℝ

/-- Represents a piece obtained from cutting a Greek cross -/
inductive Piece
| Square : Square → Piece
| Composite : List Square → Piece

/-- Theorem stating that a Greek cross can be dissected into 12 pieces 
    to form three identical smaller Greek crosses -/
theorem greek_cross_dissection (original : GreekCross) :
  ∃ (pieces : List Piece) (small_crosses : Fin 3 → GreekCross),
    (pieces.length = 12) ∧
    (∀ i : Fin 3, (small_crosses i).area = original.area / 3) ∧
    (∀ i j : Fin 3, i ≠ j → small_crosses i = small_crosses j) ∧
    (∃ (reassembly : List Piece → Fin 3 → GreekCross), 
      reassembly pieces = small_crosses) :=
sorry

end greek_cross_dissection_l1183_118345


namespace baseball_cards_distribution_l1183_118335

theorem baseball_cards_distribution (total_cards : ℕ) (num_friends : ℕ) (cards_per_friend : ℕ) :
  total_cards = 24 →
  num_friends = 4 →
  total_cards = num_friends * cards_per_friend →
  cards_per_friend = 6 := by
  sorry

end baseball_cards_distribution_l1183_118335


namespace inequality_represents_lower_right_l1183_118361

/-- Represents a point in the 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- The line defined by the equation x - 2y + 6 = 0 -/
def line (p : Point) : Prop :=
  p.x - 2 * p.y + 6 = 0

/-- The area defined by the inequality x - 2y + 6 > 0 -/
def inequality_area (p : Point) : Prop :=
  p.x - 2 * p.y + 6 > 0

/-- A point is on the lower right side of the line if it satisfies the inequality -/
def is_lower_right (p : Point) : Prop :=
  inequality_area p

theorem inequality_represents_lower_right :
  ∀ p : Point, is_lower_right p ↔ inequality_area p :=
sorry

end inequality_represents_lower_right_l1183_118361


namespace apple_tree_production_ratio_l1183_118347

theorem apple_tree_production_ratio : 
  ∀ (first_season second_season third_season : ℕ),
  first_season = 200 →
  second_season = first_season - first_season / 5 →
  first_season + second_season + third_season = 680 →
  third_season / second_season = 2 :=
by
  sorry

end apple_tree_production_ratio_l1183_118347


namespace stratified_sampling_results_l1183_118376

theorem stratified_sampling_results (total_sample : ℕ) (junior_students senior_students : ℕ) :
  total_sample = 60 ∧ junior_students = 400 ∧ senior_students = 200 →
  (Nat.choose junior_students ((total_sample * junior_students) / (junior_students + senior_students))) *
  (Nat.choose senior_students ((total_sample * senior_students) / (junior_students + senior_students))) =
  Nat.choose 400 40 * Nat.choose 200 20 :=
by sorry

end stratified_sampling_results_l1183_118376


namespace fourth_power_sum_l1183_118378

theorem fourth_power_sum (a b c : ℝ) 
  (h1 : a + b + c = 2) 
  (h2 : a^2 + b^2 + c^2 = 5) 
  (h3 : a^3 + b^3 + c^3 = 8) : 
  a^4 + b^4 + c^4 = 19 := by
  sorry

end fourth_power_sum_l1183_118378


namespace shaded_fraction_of_rectangle_l1183_118371

theorem shaded_fraction_of_rectangle (rectangle_length rectangle_width : ℝ) 
  (h_length : rectangle_length = 15)
  (h_width : rectangle_width = 20)
  (h_triangle_area : ∃ (triangle_area : ℝ), triangle_area = (1/3) * rectangle_length * rectangle_width)
  (h_shaded_area : ∃ (shaded_area : ℝ), shaded_area = (1/2) * (1/3) * rectangle_length * rectangle_width) :
  (∃ (shaded_area : ℝ), shaded_area = (1/6) * rectangle_length * rectangle_width) :=
by sorry

end shaded_fraction_of_rectangle_l1183_118371


namespace number_operation_l1183_118306

theorem number_operation (x : ℚ) : x - 7/3 = 3/2 → x + 7/3 = 37/6 := by
  sorry

end number_operation_l1183_118306


namespace binary_representation_of_2015_l1183_118331

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: aux (m / 2)
    aux n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem binary_representation_of_2015 :
  toBinary 2015 = [true, true, true, true, true, false, true, true, true, true, true] :=
by sorry

#eval fromBinary [true, true, true, true, true, false, true, true, true, true, true]

end binary_representation_of_2015_l1183_118331


namespace parabola_line_intersection_l1183_118377

theorem parabola_line_intersection (a k b x₁ x₂ x₃ : ℝ) 
  (ha : a > 0)
  (h₁ : a * x₁^2 = k * x₁ + b)
  (h₂ : a * x₂^2 = k * x₂ + b)
  (h₃ : 0 = k * x₃ + b) :
  x₁ * x₂ = x₂ * x₃ + x₁ * x₃ := by sorry

end parabola_line_intersection_l1183_118377


namespace least_common_denominator_l1183_118374

theorem least_common_denominator : 
  let denominators := [5, 6, 8, 9, 10, 11]
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9) 10) 11 = 3960 := by
  sorry

end least_common_denominator_l1183_118374


namespace reciprocal_inequality_l1183_118316

theorem reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : a * b > 0) :
  (a > 0 ∧ b > 0 → 1 / a > 1 / b) ∧
  (a < 0 ∧ b < 0 → 1 / a < 1 / b) := by
sorry

end reciprocal_inequality_l1183_118316


namespace modulo_equivalence_unique_solution_l1183_118356

theorem modulo_equivalence_unique_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ 123456 [ZMOD 9] ∧ n = 3 := by
  sorry

end modulo_equivalence_unique_solution_l1183_118356


namespace ice_cream_scoop_permutations_l1183_118318

theorem ice_cream_scoop_permutations :
  Nat.factorial 5 = 120 := by
  sorry

end ice_cream_scoop_permutations_l1183_118318


namespace sum_is_composite_l1183_118324

theorem sum_is_composite (m n : ℕ) (h : 88 * m = 81 * n) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ m + n = a * b := by
  sorry

end sum_is_composite_l1183_118324


namespace line_intersects_circle_l1183_118380

/-- Given a point outside a circle, prove that a specific line intersects the circle -/
theorem line_intersects_circle (x₀ y₀ a : ℝ) (h₁ : a > 0) (h₂ : x₀^2 + y₀^2 > a^2) :
  ∃ (x y : ℝ), x^2 + y^2 = a^2 ∧ x₀*x + y₀*y = a^2 := by
  sorry

end line_intersects_circle_l1183_118380


namespace only_two_solutions_l1183_118385

/-- Represents a solution of steers and cows --/
structure Solution :=
  (s : ℕ+)
  (c : ℕ+)

/-- Checks if a solution is valid given the budget constraint --/
def is_valid_solution (sol : Solution) : Prop :=
  30 * sol.s.val + 35 * sol.c.val = 1500

/-- The set of all valid solutions --/
def valid_solutions : Set Solution :=
  {sol : Solution | is_valid_solution sol}

/-- The theorem stating that there are only two valid solutions --/
theorem only_two_solutions :
  valid_solutions = {⟨1, 42⟩, ⟨36, 12⟩} :=
sorry

end only_two_solutions_l1183_118385


namespace largest_negative_integer_l1183_118397

theorem largest_negative_integer :
  ∀ n : ℤ, n < 0 → n ≤ -1 :=
by
  sorry

end largest_negative_integer_l1183_118397


namespace gcd_lcm_product_90_135_l1183_118310

theorem gcd_lcm_product_90_135 : Nat.gcd 90 135 * Nat.lcm 90 135 = 12150 := by
  sorry

end gcd_lcm_product_90_135_l1183_118310


namespace negation_equivalence_l1183_118327

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 + x + 1 > 0) := by
  sorry

end negation_equivalence_l1183_118327


namespace yellow_parrot_count_l1183_118382

theorem yellow_parrot_count (total : ℕ) (red_fraction : ℚ) : 
  total = 120 → red_fraction = 5/8 → (total : ℚ) * (1 - red_fraction) = 45 := by
  sorry

end yellow_parrot_count_l1183_118382


namespace joe_total_cars_l1183_118307

def initial_cars : ℕ := 500
def additional_cars : ℕ := 120

theorem joe_total_cars : initial_cars + additional_cars = 620 := by
  sorry

end joe_total_cars_l1183_118307


namespace quadratic_root_difference_l1183_118312

theorem quadratic_root_difference (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ x - y = Real.sqrt 77) →
  k ≤ Real.sqrt 109 :=
by sorry

end quadratic_root_difference_l1183_118312


namespace triangle_nth_part_area_l1183_118372

theorem triangle_nth_part_area (b h n : ℝ) (h_pos : 0 < h) (n_pos : 0 < n) :
  let original_area := (1 / 2) * b * h
  let cut_height := h / Real.sqrt n
  let cut_area := (1 / 2) * b * cut_height
  cut_area = (1 / n) * original_area := by
  sorry

end triangle_nth_part_area_l1183_118372


namespace value_of_fraction_difference_l1183_118343

theorem value_of_fraction_difference (x y : ℝ) 
  (hx : x = Real.sqrt 5 - 1) 
  (hy : y = Real.sqrt 5 + 1) : 
  1 / x - 1 / y = 1 / 2 := by sorry

end value_of_fraction_difference_l1183_118343


namespace only_B_in_fourth_quadrant_l1183_118303

def point_A : ℝ × ℝ := (2, 3)
def point_B : ℝ × ℝ := (1, -1)
def point_C : ℝ × ℝ := (-2, 1)
def point_D : ℝ × ℝ := (-2, -1)

def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem only_B_in_fourth_quadrant :
  in_fourth_quadrant point_B ∧
  ¬in_fourth_quadrant point_A ∧
  ¬in_fourth_quadrant point_C ∧
  ¬in_fourth_quadrant point_D :=
by sorry

end only_B_in_fourth_quadrant_l1183_118303


namespace bakers_pastry_problem_l1183_118342

/-- Baker's pastry problem -/
theorem bakers_pastry_problem (cakes_sold : ℕ) (difference : ℕ) (pastries_sold : ℕ) :
  cakes_sold = 97 →
  cakes_sold = pastries_sold + difference →
  difference = 89 →
  pastries_sold = 8 := by
  sorry

end bakers_pastry_problem_l1183_118342


namespace connie_markers_total_l1183_118305

theorem connie_markers_total (red : ℕ) (blue : ℕ) (green : ℕ) (yellow : ℕ)
  (h_red : red = 5420)
  (h_blue : blue = 3875)
  (h_green : green = 2910)
  (h_yellow : yellow = 6740) :
  red + blue + green + yellow = 18945 := by
  sorry

end connie_markers_total_l1183_118305


namespace sqrt_difference_equals_seven_l1183_118337

theorem sqrt_difference_equals_seven : 
  Real.sqrt (36 + 64) - Real.sqrt (25 - 16) = 7 := by sorry

end sqrt_difference_equals_seven_l1183_118337


namespace rug_length_is_25_l1183_118352

/-- Represents a rectangular rug with integer dimensions -/
structure Rug where
  length : ℕ
  width : ℕ

/-- Represents a rectangular room -/
structure Room where
  width : ℕ
  length : ℕ

/-- Checks if a rug fits perfectly in a room -/
def fitsInRoom (rug : Rug) (room : Room) : Prop :=
  rug.length ^ 2 + rug.width ^ 2 = room.width ^ 2 + room.length ^ 2

theorem rug_length_is_25 :
  ∃ (rug : Rug) (room1 room2 : Room),
    room1.width = 38 ∧
    room2.width = 50 ∧
    room1.length = room2.length ∧
    fitsInRoom rug room1 ∧
    fitsInRoom rug room2 →
    rug.length = 25 := by
  sorry

end rug_length_is_25_l1183_118352


namespace correct_average_weight_l1183_118341

def class_size : ℕ := 20
def initial_average : ℚ := 58.4
def misread_weight : ℕ := 56
def correct_weight : ℕ := 62

theorem correct_average_weight :
  let incorrect_total := initial_average * class_size
  let weight_difference := correct_weight - misread_weight
  let correct_total := incorrect_total + weight_difference
  (correct_total / class_size : ℚ) = 58.7 := by sorry

end correct_average_weight_l1183_118341


namespace tan_150_degrees_l1183_118348

theorem tan_150_degrees : 
  Real.tan (150 * π / 180) = -1 / Real.sqrt 3 := by
  sorry

end tan_150_degrees_l1183_118348


namespace units_digit_of_factorial_sum_l1183_118346

def factorial (n : ℕ) : ℕ := sorry

def sum_factorials (n : ℕ) : ℕ := sorry

theorem units_digit_of_factorial_sum : 
  ∃ k : ℕ, (sum_factorials 500 + factorial 2 * factorial 4 + factorial 3 * factorial 7) % 10 = 1 + 10 * k :=
by
  sorry

end units_digit_of_factorial_sum_l1183_118346


namespace distribute_and_simplify_l1183_118308

theorem distribute_and_simplify (a : ℝ) : a * (a - 3) = a^2 - 3*a := by
  sorry

end distribute_and_simplify_l1183_118308


namespace driveway_snow_volume_l1183_118338

/-- Calculates the total volume of snow on a driveway with given dimensions and snow depths -/
theorem driveway_snow_volume 
  (driveway_length : ℝ) 
  (driveway_width : ℝ) 
  (section1_length : ℝ) 
  (section1_depth : ℝ) 
  (section2_length : ℝ) 
  (section2_depth : ℝ) 
  (h1 : driveway_length = 30) 
  (h2 : driveway_width = 3) 
  (h3 : section1_length = 10) 
  (h4 : section1_depth = 1) 
  (h5 : section2_length = 20) 
  (h6 : section2_depth = 0.5) 
  (h7 : section1_length + section2_length = driveway_length) : 
  section1_length * driveway_width * section1_depth + 
  section2_length * driveway_width * section2_depth = 60 :=
by
  sorry

#check driveway_snow_volume

end driveway_snow_volume_l1183_118338


namespace alternating_series_sum_equals_minus_30_l1183_118344

def alternatingSeriesSum (a₁ : ℤ) (d : ℤ) (lastTerm : ℤ) : ℤ :=
  -- Definition of the sum of the alternating series
  sorry

theorem alternating_series_sum_equals_minus_30 :
  alternatingSeriesSum 2 6 59 = -30 := by
  sorry

end alternating_series_sum_equals_minus_30_l1183_118344


namespace workshop_theorem_l1183_118311

def workshop_problem (total_members : ℕ) (avg_age_all : ℝ) 
                     (num_girls : ℕ) (num_boys : ℕ) (num_adults : ℕ) 
                     (avg_age_girls : ℝ) (avg_age_boys : ℝ) : Prop :=
  let total_age := total_members * avg_age_all
  let girls_age := num_girls * avg_age_girls
  let boys_age := num_boys * avg_age_boys
  let adults_age := total_age - girls_age - boys_age
  (adults_age / num_adults) = 26.2

theorem workshop_theorem : 
  workshop_problem 50 20 22 18 10 18 19 := by
  sorry

end workshop_theorem_l1183_118311


namespace twelve_buses_required_l1183_118333

/-- The minimum number of buses required to transport all students -/
def min_buses (max_capacity : ℕ) (total_students : ℕ) (available_drivers : ℕ) : ℕ :=
  max (((total_students + max_capacity - 1) / max_capacity) : ℕ) available_drivers

/-- Proof that 12 buses are required given the problem conditions -/
theorem twelve_buses_required :
  min_buses 42 480 12 = 12 := by
  sorry

#eval min_buses 42 480 12  -- Should output 12

end twelve_buses_required_l1183_118333


namespace hall_volume_l1183_118319

/-- Given a rectangular hall with length 15 m and breadth 12 m, if the sum of the areas of
    the floor and ceiling is equal to the sum of the areas of four walls, then the volume
    of the hall is 8004 m³. -/
theorem hall_volume (height : ℝ) : 
  (15 : ℝ) * 12 * height = 8004 ∧ 
  2 * (15 * 12) = 2 * (15 * height) + 2 * (12 * height) := by
  sorry

#check hall_volume

end hall_volume_l1183_118319


namespace calculate_gratuity_percentage_l1183_118370

/-- Calculate the gratuity percentage for a restaurant bill -/
theorem calculate_gratuity_percentage
  (num_people : ℕ)
  (total_bill : ℚ)
  (avg_cost_before_gratuity : ℚ)
  (h_num_people : num_people = 9)
  (h_total_bill : total_bill = 756)
  (h_avg_cost : avg_cost_before_gratuity = 70) :
  (total_bill - num_people * avg_cost_before_gratuity) / (num_people * avg_cost_before_gratuity) = 1/5 :=
sorry

end calculate_gratuity_percentage_l1183_118370


namespace cone_base_radius_l1183_118340

/-- 
Given a cone whose lateral surface, when unfolded, is a semicircle with radius 1,
prove that the radius of the base of the cone is 1/2.
-/
theorem cone_base_radius (r : ℝ) : r > 0 → r = 1 → (2 * π * (1 / 2 : ℝ)) = (π * r) → (1 / 2 : ℝ) = r := by
  sorry

end cone_base_radius_l1183_118340


namespace function_symmetry_and_translation_l1183_118358

-- Define a function type
def RealFunction := ℝ → ℝ

-- Define the translation operation
def translate (f : RealFunction) (h : ℝ) : RealFunction :=
  λ x => f (x + h)

-- Define symmetry with respect to y-axis
def symmetricToYAxis (f g : RealFunction) : Prop :=
  ∀ x, f x = g (-x)

-- State the theorem
theorem function_symmetry_and_translation (f : RealFunction) :
  (symmetricToYAxis (translate f 1) (λ x => 2^x)) →
  (f = λ x => (1/2)^(x-1)) := by
  sorry

end function_symmetry_and_translation_l1183_118358


namespace f_increasing_f_two_zeros_l1183_118339

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * abs (x + 1) + a * x

-- Statement 1: f(x) is increasing when a > 2
theorem f_increasing (a : ℝ) (h : a > 2) : 
  StrictMono (f a) := by sorry

-- Statement 2: f(x) has two zeros iff a ∈ (0, 2)
theorem f_two_zeros (a : ℝ) : 
  (∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ 0 < a ∧ a < 2 := by sorry

end f_increasing_f_two_zeros_l1183_118339


namespace basketball_not_tabletennis_l1183_118392

theorem basketball_not_tabletennis (U A B : Finset ℕ) : 
  Finset.card U = 42 →
  Finset.card A = 20 →
  Finset.card B = 25 →
  Finset.card (U \ (A ∪ B)) = 12 →
  Finset.card (A \ B) = 5 := by
sorry

end basketball_not_tabletennis_l1183_118392


namespace cube_increase_theorem_l1183_118309

def cube_edge_increase_percent : ℝ := 60

theorem cube_increase_theorem (s : ℝ) (h : s > 0) :
  let new_edge := s * (1 + cube_edge_increase_percent / 100)
  let original_surface_area := 6 * s^2
  let new_surface_area := 6 * new_edge^2
  let original_volume := s^3
  let new_volume := new_edge^3
  (new_surface_area - original_surface_area) / original_surface_area * 100 = 156 ∧
  (new_volume - original_volume) / original_volume * 100 = 309.6 := by
sorry


end cube_increase_theorem_l1183_118309
