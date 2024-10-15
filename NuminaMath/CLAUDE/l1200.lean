import Mathlib

namespace NUMINAMATH_CALUDE_advertising_time_l1200_120023

def newscast_duration : ℕ := 30
def national_news_duration : ℕ := 12
def international_news_duration : ℕ := 5
def sports_duration : ℕ := 5
def weather_forecast_duration : ℕ := 2

theorem advertising_time :
  newscast_duration - (national_news_duration + international_news_duration + sports_duration + weather_forecast_duration) = 6 := by
  sorry

end NUMINAMATH_CALUDE_advertising_time_l1200_120023


namespace NUMINAMATH_CALUDE_solve_factorial_equation_l1200_120037

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem solve_factorial_equation : ∃ (n : ℕ), n * factorial n + factorial n = 5040 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_factorial_equation_l1200_120037


namespace NUMINAMATH_CALUDE_minimum_students_with_girl_percentage_l1200_120063

theorem minimum_students_with_girl_percentage (n : ℕ) (g : ℕ) : n > 0 → g > 0 → (25 : ℚ) / 100 < (g : ℚ) / n → (g : ℚ) / n < (30 : ℚ) / 100 → n ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_minimum_students_with_girl_percentage_l1200_120063


namespace NUMINAMATH_CALUDE_tenth_term_is_neg_512_l1200_120061

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℤ  -- The sequence (a₁, a₂, a₃, ...)
  is_geometric : ∀ n : ℕ, n ≥ 2 → a (n + 1) / a n = a 2 / a 1
  product_25 : a 2 * a 5 = -32
  sum_34 : a 3 + a 4 = 4
  integer_ratio : ∃ q : ℤ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = q * a n

/-- The 10th term of the geometric sequence is -512 -/
theorem tenth_term_is_neg_512 (seq : GeometricSequence) : seq.a 10 = -512 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_neg_512_l1200_120061


namespace NUMINAMATH_CALUDE_area_PQR_is_sqrt_35_l1200_120034

/-- Represents a square pyramid with given dimensions and points -/
structure SquarePyramid where
  base_side : ℝ
  altitude : ℝ
  p_ratio : ℝ
  q_ratio : ℝ
  r_ratio : ℝ

/-- Calculates the area of triangle PQR in the square pyramid -/
def area_PQR (pyramid : SquarePyramid) : ℝ :=
  sorry

/-- Theorem stating that the area of triangle PQR is √35 for the given pyramid -/
theorem area_PQR_is_sqrt_35 :
  let pyramid := SquarePyramid.mk 4 8 (1/4) (1/4) (3/4)
  area_PQR pyramid = Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_area_PQR_is_sqrt_35_l1200_120034


namespace NUMINAMATH_CALUDE_power_of_ten_zeros_l1200_120057

theorem power_of_ten_zeros (n : ℕ) : 10000 ^ 50 * 10 ^ 5 = 10 ^ 205 := by
  sorry

end NUMINAMATH_CALUDE_power_of_ten_zeros_l1200_120057


namespace NUMINAMATH_CALUDE_bobby_candy_consumption_l1200_120030

theorem bobby_candy_consumption (initial_candy : ℕ) (remaining_candy : ℕ) 
  (h1 : initial_candy = 30) (h2 : remaining_candy = 7) :
  initial_candy - remaining_candy = 23 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_consumption_l1200_120030


namespace NUMINAMATH_CALUDE_sortable_configurations_after_three_passes_l1200_120017

/-- The number of sortable book configurations after three passes -/
def sortableConfigurations (n : ℕ) : ℕ :=
  6 * 4^(n - 3)

/-- Theorem stating the number of sortable configurations for n ≥ 3 books after three passes -/
theorem sortable_configurations_after_three_passes (n : ℕ) (h : n ≥ 3) :
  sortableConfigurations n = 6 * 4^(n - 3) := by
  sorry

end NUMINAMATH_CALUDE_sortable_configurations_after_three_passes_l1200_120017


namespace NUMINAMATH_CALUDE_population_ratio_problem_l1200_120045

theorem population_ratio_problem (s v : ℝ) 
  (h1 : 0.94 * s = 1.14 * v) : s / v = 57 / 47 := by
  sorry

end NUMINAMATH_CALUDE_population_ratio_problem_l1200_120045


namespace NUMINAMATH_CALUDE_sunflower_seeds_majority_l1200_120028

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : Nat
  sunflowerSeeds : Rat
  otherSeeds : Rat

/-- Calculates the next day's feeder state -/
def nextDay (state : FeederState) : FeederState :=
  { day := state.day + 1,
    sunflowerSeeds := state.sunflowerSeeds * (4/5) + (2/5),
    otherSeeds := 3/5 }

/-- Initial state of the feeder on Monday -/
def initialState : FeederState :=
  { day := 1,
    sunflowerSeeds := 2/5,
    otherSeeds := 3/5 }

/-- Theorem stating that on the third day, more than half the seeds are sunflower seeds -/
theorem sunflower_seeds_majority : 
  let state3 := nextDay (nextDay initialState)
  state3.sunflowerSeeds > (state3.sunflowerSeeds + state3.otherSeeds) / 2 := by
  sorry


end NUMINAMATH_CALUDE_sunflower_seeds_majority_l1200_120028


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_6_squared_l1200_120042

def units_digit_pattern : List Nat := [7, 9, 3, 1]

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_7_pow_6_squared : 
  units_digit (7^(6^2)) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_6_squared_l1200_120042


namespace NUMINAMATH_CALUDE_josie_money_left_l1200_120076

/-- Calculates the amount of money Josie has left after grocery shopping --/
def money_left_after_shopping (initial_amount : ℚ) (milk_price : ℚ) (bread_price : ℚ) 
  (detergent_price : ℚ) (banana_price_per_pound : ℚ) (banana_pounds : ℚ) 
  (milk_discount : ℚ) (detergent_discount : ℚ) : ℚ :=
  let milk_cost := milk_price * (1 - milk_discount)
  let detergent_cost := detergent_price - detergent_discount
  let banana_cost := banana_price_per_pound * banana_pounds
  let total_cost := milk_cost + bread_price + detergent_cost + banana_cost
  initial_amount - total_cost

/-- Theorem stating that Josie has $4.00 left after shopping --/
theorem josie_money_left : 
  money_left_after_shopping 20 4 3.5 10.25 0.75 2 0.5 1.25 = 4 := by
  sorry

end NUMINAMATH_CALUDE_josie_money_left_l1200_120076


namespace NUMINAMATH_CALUDE_shooting_competition_probability_l1200_120038

theorem shooting_competition_probability (p_10 p_9 p_8 p_7 : ℝ) 
  (h1 : p_10 = 0.15)
  (h2 : p_9 = 0.35)
  (h3 : p_8 = 0.2)
  (h4 : p_7 = 0.1) :
  p_7 = 0.3 :=
sorry

end NUMINAMATH_CALUDE_shooting_competition_probability_l1200_120038


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1200_120048

theorem consecutive_integers_sum (n : ℤ) : 
  (n - 1) * n * (n + 1) = 336 → (n - 1) + n + (n + 1) = 21 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1200_120048


namespace NUMINAMATH_CALUDE_solution_sets_equal_l1200_120010

/-- A strictly increasing bijective function from R to R -/
def StrictlyIncreasingBijection (f : ℝ → ℝ) : Prop :=
  Function.Bijective f ∧ StrictMono f

/-- The solution set of x = f(x) -/
def SolutionSetP (f : ℝ → ℝ) : Set ℝ :=
  {x | x = f x}

/-- The solution set of x = f(f(x)) -/
def SolutionSetQ (f : ℝ → ℝ) : Set ℝ :=
  {x | x = f (f x)}

/-- Theorem: For a strictly increasing bijective function f from R to R,
    the solution set P of x = f(x) is equal to the solution set Q of x = f(f(x)) -/
theorem solution_sets_equal (f : ℝ → ℝ) (h : StrictlyIncreasingBijection f) :
  SolutionSetP f = SolutionSetQ f := by
  sorry

end NUMINAMATH_CALUDE_solution_sets_equal_l1200_120010


namespace NUMINAMATH_CALUDE_fastest_student_requires_comprehensive_survey_l1200_120082

/-- Represents a survey method -/
inductive SurveyMethod
| Comprehensive
| Sample

/-- Represents a survey scenario -/
structure SurveyScenario where
  description : String
  requiredMethod : SurveyMethod

/-- Define the four survey scenarios -/
def viewershipSurvey : SurveyScenario :=
  { description := "Investigating the viewership rate of the Spring Festival Gala"
    requiredMethod := SurveyMethod.Sample }

def colorantSurvey : SurveyScenario :=
  { description := "Investigating whether the colorant content of a certain food in the market meets national standards"
    requiredMethod := SurveyMethod.Sample }

def shoeSoleSurvey : SurveyScenario :=
  { description := "Testing the number of times the shoe soles produced by a shoe factory can withstand bending"
    requiredMethod := SurveyMethod.Sample }

def fastestStudentSurvey : SurveyScenario :=
  { description := "Selecting the fastest student in short-distance running at a certain school to participate in the city-wide competition"
    requiredMethod := SurveyMethod.Comprehensive }

/-- Theorem stating that selecting the fastest student requires a comprehensive survey -/
theorem fastest_student_requires_comprehensive_survey :
  fastestStudentSurvey.requiredMethod = SurveyMethod.Comprehensive ∧
  viewershipSurvey.requiredMethod ≠ SurveyMethod.Comprehensive ∧
  colorantSurvey.requiredMethod ≠ SurveyMethod.Comprehensive ∧
  shoeSoleSurvey.requiredMethod ≠ SurveyMethod.Comprehensive :=
sorry

end NUMINAMATH_CALUDE_fastest_student_requires_comprehensive_survey_l1200_120082


namespace NUMINAMATH_CALUDE_cat_mouse_positions_after_196_moves_l1200_120049

/-- Represents the four squares in the grid --/
inductive Square
| TopLeft
| TopRight
| BottomLeft
| BottomRight

/-- Represents the eight outer segments of the squares --/
inductive Segment
| TopLeft
| TopMiddle
| TopRight
| RightMiddle
| BottomRight
| BottomMiddle
| BottomLeft
| LeftMiddle

/-- The cat's position after a given number of moves --/
def catPosition (moves : Nat) : Square :=
  match moves % 4 with
  | 0 => Square.TopLeft
  | 1 => Square.BottomLeft
  | 2 => Square.BottomRight
  | 3 => Square.TopRight
  | _ => Square.TopLeft  -- This case is unreachable, but needed for exhaustiveness

/-- The mouse's position after a given number of moves --/
def mousePosition (moves : Nat) : Segment :=
  match moves % 8 with
  | 0 => Segment.TopMiddle
  | 1 => Segment.TopRight
  | 2 => Segment.RightMiddle
  | 3 => Segment.BottomRight
  | 4 => Segment.BottomMiddle
  | 5 => Segment.BottomLeft
  | 6 => Segment.LeftMiddle
  | 7 => Segment.TopLeft
  | _ => Segment.TopMiddle  -- This case is unreachable, but needed for exhaustiveness

theorem cat_mouse_positions_after_196_moves :
  catPosition 196 = Square.TopLeft ∧ mousePosition 196 = Segment.BottomMiddle := by
  sorry


end NUMINAMATH_CALUDE_cat_mouse_positions_after_196_moves_l1200_120049


namespace NUMINAMATH_CALUDE_arccos_negative_half_l1200_120011

theorem arccos_negative_half : Real.arccos (-1/2) = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_negative_half_l1200_120011


namespace NUMINAMATH_CALUDE_stratified_sample_ninth_grade_l1200_120055

/-- Represents the number of students in each grade and the sample size for 7th grade -/
structure SchoolData where
  total : ℕ
  seventh : ℕ
  eighth : ℕ
  ninth : ℕ
  sample_seventh : ℕ

/-- Calculates the sample size for 9th grade using stratified sampling -/
def stratified_sample (data : SchoolData) : ℕ :=
  (data.sample_seventh * data.ninth) / data.seventh

/-- Theorem stating that the stratified sample for 9th grade is 224 given the school data -/
theorem stratified_sample_ninth_grade 
  (data : SchoolData) 
  (h1 : data.total = 1700)
  (h2 : data.seventh = 600)
  (h3 : data.eighth = 540)
  (h4 : data.ninth = 560)
  (h5 : data.sample_seventh = 240) :
  stratified_sample data = 224 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_ninth_grade_l1200_120055


namespace NUMINAMATH_CALUDE_complex_number_equalities_l1200_120071

/-- Prove complex number equalities -/
theorem complex_number_equalities :
  let i : ℂ := Complex.I
  let z₁ : ℂ := (1 + 2*i)^2 + 3*(1 - i)
  let z₂ : ℂ := 2 + i
  let z₃ : ℂ := 1 - i
  let z₄ : ℂ := 1 + i
  let z₅ : ℂ := 1 - Complex.I * Real.sqrt 3
  let z₆ : ℂ := Complex.I * Real.sqrt 3 + i
  (z₁ / z₂ = 1/5 + 2/5*i) ∧
  (z₃ / z₄^2 + z₄ / z₃^2 = -1) ∧
  (z₅ / z₆^2 = -1/4 - (Real.sqrt 3)/4*i) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equalities_l1200_120071


namespace NUMINAMATH_CALUDE_point_on_ellipse_l1200_120074

/-- The coordinates of a point P on an ellipse satisfying specific conditions -/
theorem point_on_ellipse (x y : ℝ) : 
  x > 0 → -- P is on the right side of y-axis
  x^2 / 5 + y^2 / 4 = 1 → -- P is on the ellipse
  (1/2) * 2 * |y| = 1 → -- Area of triangle PF₁F₂ is 1
  (x = Real.sqrt 15 / 2) ∧ (y = 1) := by
  sorry

end NUMINAMATH_CALUDE_point_on_ellipse_l1200_120074


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1200_120035

theorem sqrt_equation_solution : ∃! x : ℝ, Real.sqrt (2 * x + 3) = x := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1200_120035


namespace NUMINAMATH_CALUDE_even_function_problem_l1200_120091

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_problem (f : ℝ → ℝ) 
  (h1 : EvenFunction f) (h2 : f (-5) = 9) : f 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_even_function_problem_l1200_120091


namespace NUMINAMATH_CALUDE_range_of_m_l1200_120089

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 2*x - 15 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 2*x - m^2 + 1 ≤ 0

-- Define the property that ¬p is a necessary but not sufficient condition for ¬q
def not_p_necessary_not_sufficient_for_not_q (m : ℝ) : Prop :=
  ∀ x, q x m → p x ∧ ∃ y, ¬(p y) ∧ q y m

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, not_p_necessary_not_sufficient_for_not_q m ↔ (m < -4 ∨ m > 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1200_120089


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l1200_120033

theorem two_digit_number_puzzle : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 1000 + 100 * (n / 10) + 10 * (n % 10) + 1 = 23 * n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l1200_120033


namespace NUMINAMATH_CALUDE_circle_equations_from_line_intersections_l1200_120065

-- Define the line
def line (x y : ℝ) : Prop := 2 * x + y - 4 = 0

-- Define the two intersection points
def point_A : ℝ × ℝ := (0, 4)
def point_B : ℝ × ℝ := (2, 0)

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 20
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 20

-- Theorem statement
theorem circle_equations_from_line_intersections :
  (∀ x y : ℝ, line x y → (x = 0 ∧ y = 4) ∨ (x = 2 ∧ y = 0)) ∧
  (circle1 (point_A.1) (point_A.2) ∧ circle1 (point_B.1) (point_B.2)) ∧
  (circle2 (point_A.1) (point_A.2) ∧ circle2 (point_B.1) (point_B.2)) :=
sorry

end NUMINAMATH_CALUDE_circle_equations_from_line_intersections_l1200_120065


namespace NUMINAMATH_CALUDE_train_length_proof_l1200_120084

/-- Proves that the length of each train is 50 meters given the specified conditions -/
theorem train_length_proof (v_fast v_slow : ℝ) (t : ℝ) (h1 : v_fast = 46) (h2 : v_slow = 36) (h3 : t = 36) :
  let v_rel := (v_fast - v_slow) * (5 / 18)  -- Convert km/hr to m/s
  let l := v_rel * t / 2                     -- Length of one train
  l = 50 := by sorry

end NUMINAMATH_CALUDE_train_length_proof_l1200_120084


namespace NUMINAMATH_CALUDE_original_eq_hyperbola_and_ellipse_l1200_120020

-- Define the original equation
def original_equation (x y : ℝ) : Prop := y^4 - 16*x^4 = 8*y^2 - 4

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := y^2 - 4*x^2 = 4

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := y^2 + 4*x^2 = 4

-- Theorem stating that the original equation is equivalent to the union of a hyperbola and an ellipse
theorem original_eq_hyperbola_and_ellipse :
  ∀ x y : ℝ, original_equation x y ↔ (hyperbola_equation x y ∨ ellipse_equation x y) :=
sorry

end NUMINAMATH_CALUDE_original_eq_hyperbola_and_ellipse_l1200_120020


namespace NUMINAMATH_CALUDE_complex_solutions_count_l1200_120096

open Complex

theorem complex_solutions_count : 
  ∃ (S : Finset ℂ), (∀ z ∈ S, (z^4 - 1) / (z^2 + z + 1) = 0) ∧ 
                    (∀ z : ℂ, (z^4 - 1) / (z^2 + z + 1) = 0 → z ∈ S) ∧
                    Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_solutions_count_l1200_120096


namespace NUMINAMATH_CALUDE_characterize_nat_function_l1200_120081

/-- A function from natural numbers to natural numbers -/
def NatFunction := ℕ → ℕ

/-- Predicate that checks if a number is a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- Theorem statement -/
theorem characterize_nat_function (f : NatFunction) :
  (∀ m n : ℕ, IsPerfectSquare (f n + 2 * m * n + f m)) →
  ∃ ℓ : ℕ, ∀ n : ℕ, f n = (n + 2 * ℓ)^2 - 2 * ℓ^2 :=
by sorry

end NUMINAMATH_CALUDE_characterize_nat_function_l1200_120081


namespace NUMINAMATH_CALUDE_max_value_fourth_root_sum_l1200_120097

theorem max_value_fourth_root_sum (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hsum : a + b + c + d ≤ 4) :
  (a^2 + 3*a*b)^(1/4) + (b^2 + 3*b*c)^(1/4) + (c^2 + 3*c*d)^(1/4) + (d^2 + 3*d*a)^(1/4) ≤ 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fourth_root_sum_l1200_120097


namespace NUMINAMATH_CALUDE_largest_angle_of_special_hexagon_l1200_120083

-- Define a hexagon type
structure Hexagon where
  angles : Fin 6 → ℝ
  is_convex : True
  consecutive_integers : ∀ i : Fin 5, ∃ n : ℤ, angles i.succ = angles i + 1
  sum_720 : (Finset.univ.sum angles) = 720

-- Theorem statement
theorem largest_angle_of_special_hexagon (h : Hexagon) :
  Finset.max' (Finset.univ.image h.angles) (by sorry) = 122.5 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_of_special_hexagon_l1200_120083


namespace NUMINAMATH_CALUDE_summer_camp_group_size_l1200_120078

/-- The number of children in Mrs. Generous' summer camp group -/
def num_children : ℕ := 31

/-- The number of jelly beans Mrs. Generous brought -/
def total_jelly_beans : ℕ := 500

/-- The number of jelly beans left after distribution -/
def leftover_jelly_beans : ℕ := 10

/-- The difference between the number of boys and girls -/
def boy_girl_difference : ℕ := 3

theorem summer_camp_group_size :
  ∃ (girls boys : ℕ),
    girls + boys = num_children ∧
    boys = girls + boy_girl_difference ∧
    girls * girls + boys * boys = total_jelly_beans - leftover_jelly_beans :=
by sorry

end NUMINAMATH_CALUDE_summer_camp_group_size_l1200_120078


namespace NUMINAMATH_CALUDE_expression_simplification_l1200_120085

theorem expression_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  ((x + 2*a)^2) / ((a - b)*(a - c)) + ((x + 2*b)^2) / ((b - a)*(b - c)) + ((x + 2*c)^2) / ((c - a)*(c - b)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1200_120085


namespace NUMINAMATH_CALUDE_min_value_fraction_l1200_120024

theorem min_value_fraction (x : ℝ) (h : x ≥ 0) :
  (5 * x^2 + 20 * x + 25) / (8 * (1 + x)) ≥ 65 / 16 ∧
  ∃ y : ℝ, y ≥ 0 ∧ (5 * y^2 + 20 * y + 25) / (8 * (1 + y)) = 65 / 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1200_120024


namespace NUMINAMATH_CALUDE_cabbage_increase_l1200_120007

/-- Represents a square garden where cabbages are grown -/
structure CabbageGarden where
  side : ℕ  -- Side length of the square garden

/-- The number of cabbages in a garden -/
def num_cabbages (g : CabbageGarden) : ℕ := g.side * g.side

/-- Theorem: If the number of cabbages increased by 199 from last year to this year,
    and the garden remained square-shaped, then the number of cabbages this year is 10,000 -/
theorem cabbage_increase (last_year this_year : CabbageGarden) :
  num_cabbages this_year = num_cabbages last_year + 199 →
  num_cabbages this_year = 10000 := by
  sorry

#check cabbage_increase

end NUMINAMATH_CALUDE_cabbage_increase_l1200_120007


namespace NUMINAMATH_CALUDE_teddy_bears_per_shelf_l1200_120066

theorem teddy_bears_per_shelf (total_bears : ℕ) (num_shelves : ℕ) 
  (h1 : total_bears = 98) (h2 : num_shelves = 14) :
  (total_bears / num_shelves : ℕ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_teddy_bears_per_shelf_l1200_120066


namespace NUMINAMATH_CALUDE_max_puzzle_sets_l1200_120047

/-- Represents a set of puzzles -/
structure PuzzleSet where
  logic : ℕ
  visual : ℕ
  word : ℕ

/-- Checks if a PuzzleSet is valid according to the given conditions -/
def isValidSet (s : PuzzleSet) : Prop :=
  7 ≤ s.logic + s.visual + s.word ∧
  s.logic + s.visual + s.word ≤ 12 ∧
  4 * s.visual = 3 * s.logic ∧
  2 * s.word ≥ s.visual

/-- The main theorem stating the maximum number of sets that can be created -/
theorem max_puzzle_sets :
  ∃ (s : PuzzleSet),
    isValidSet s ∧
    (∃ (n : ℕ), n = 5 ∧
      n * s.logic = 36 ∧
      n * s.visual = 27 ∧
      n * s.word = 15) ∧
    (∀ (m : ℕ) (t : PuzzleSet),
      isValidSet t →
      m * t.logic ≤ 36 ∧
      m * t.visual ≤ 27 ∧
      m * t.word ≤ 15 →
      m ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_max_puzzle_sets_l1200_120047


namespace NUMINAMATH_CALUDE_pencil_cost_l1200_120051

/-- Given that 120 pencils cost $36, prove that 3000 pencils cost $900 -/
theorem pencil_cost (cost_120 : ℕ) (quantity : ℕ) (h1 : cost_120 = 36) (h2 : quantity = 3000) :
  (cost_120 * quantity) / 120 = 900 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l1200_120051


namespace NUMINAMATH_CALUDE_compare_sqrt_expressions_l1200_120018

theorem compare_sqrt_expressions : 2 * Real.sqrt 3 < 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_compare_sqrt_expressions_l1200_120018


namespace NUMINAMATH_CALUDE_odd_function_symmetry_l1200_120043

-- Define a real-valued function
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be odd
def IsOdd (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define symmetry about the y-axis for the absolute value of a function
def IsSymmetricAboutYAxis (f : RealFunction) : Prop :=
  ∀ x : ℝ, |f (-x)| = |f x|

-- Theorem statement
theorem odd_function_symmetry :
  (∀ f : RealFunction, IsOdd f → IsSymmetricAboutYAxis f) ∧
  (∃ f : RealFunction, IsSymmetricAboutYAxis f ∧ ¬IsOdd f) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_symmetry_l1200_120043


namespace NUMINAMATH_CALUDE_first_day_over_500_day_is_saturday_l1200_120029

def paperclips (k : ℕ) : ℕ := 5 * 3^k

theorem first_day_over_500 :
  ∃ k : ℕ, paperclips k > 500 ∧ ∀ j : ℕ, j < k → paperclips j ≤ 500 :=
by sorry

theorem day_is_saturday : 
  ∃ k : ℕ, paperclips k > 500 ∧ ∀ j : ℕ, j < k → paperclips j ≤ 500 → k = 5 :=
by sorry

end NUMINAMATH_CALUDE_first_day_over_500_day_is_saturday_l1200_120029


namespace NUMINAMATH_CALUDE_inductive_reasoning_classification_l1200_120013

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| NonInductive

-- Define the types of inductive reasoning
inductive InductiveReasoningType
| Generalization
| Analogy

-- Define a structure for an inference
structure Inference where
  id : Nat
  reasoningType : ReasoningType
  inductiveType : Option InductiveReasoningType

-- Define the inferences
def inference1 : Inference := ⟨1, ReasoningType.Inductive, some InductiveReasoningType.Analogy⟩
def inference2 : Inference := ⟨2, ReasoningType.Inductive, some InductiveReasoningType.Generalization⟩
def inference3 : Inference := ⟨3, ReasoningType.NonInductive, none⟩
def inference4 : Inference := ⟨4, ReasoningType.Inductive, some InductiveReasoningType.Generalization⟩

-- Define a function to check if an inference is inductive
def isInductive (i : Inference) : Prop :=
  i.reasoningType = ReasoningType.Inductive

-- Theorem to prove
theorem inductive_reasoning_classification :
  (isInductive inference1) ∧
  (isInductive inference2) ∧
  (¬isInductive inference3) ∧
  (isInductive inference4) := by
  sorry

end NUMINAMATH_CALUDE_inductive_reasoning_classification_l1200_120013


namespace NUMINAMATH_CALUDE_solution_value_l1200_120073

theorem solution_value (p q : ℝ) : 
  (3 * p^2 - 5 * p = 12) → 
  (3 * q^2 - 5 * q = 12) → 
  p ≠ q →
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1200_120073


namespace NUMINAMATH_CALUDE_smallest_n_for_zero_last_four_digits_l1200_120014

def last_four_digits_zero (n : ℕ) : Prop :=
  ∃ k : ℕ, 225 * 525 * n = k * 10000

theorem smallest_n_for_zero_last_four_digits :
  ∀ n : ℕ, n < 16 → ¬(last_four_digits_zero n) ∧ last_four_digits_zero 16 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_zero_last_four_digits_l1200_120014


namespace NUMINAMATH_CALUDE_knitting_productivity_ratio_l1200_120094

/-- Represents the knitting productivity of a girl -/
structure Knitter where
  work_time : ℕ  -- Time spent working before a break
  break_time : ℕ -- Duration of the break

/-- Calculates the total cycle time for a knitter -/
def cycle_time (k : Knitter) : ℕ := k.work_time + k.break_time

/-- Calculates the actual working time within a given period -/
def working_time (k : Knitter) (period : ℕ) : ℕ :=
  (period / cycle_time k) * k.work_time

theorem knitting_productivity_ratio :
  let girl1 : Knitter := ⟨5, 1⟩
  let girl2 : Knitter := ⟨7, 1⟩
  let common_period := Nat.lcm (cycle_time girl1) (cycle_time girl2)
  (working_time girl2 common_period : ℚ) / (working_time girl1 common_period) = 20/21 :=
sorry

end NUMINAMATH_CALUDE_knitting_productivity_ratio_l1200_120094


namespace NUMINAMATH_CALUDE_sum_of_multiples_is_even_l1200_120032

theorem sum_of_multiples_is_even (a b : ℤ) (ha : 4 ∣ a) (hb : 6 ∣ b) : 2 ∣ (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_is_even_l1200_120032


namespace NUMINAMATH_CALUDE_angle_of_inclination_for_unit_slope_l1200_120053

/-- Given a line with slope of absolute value 1, its angle of inclination is either 45° or 135°. -/
theorem angle_of_inclination_for_unit_slope (slope : ℝ) (h : |slope| = 1) :
  let angle := Real.arctan slope
  angle = π/4 ∨ angle = 3*π/4 := by
sorry

end NUMINAMATH_CALUDE_angle_of_inclination_for_unit_slope_l1200_120053


namespace NUMINAMATH_CALUDE_quadratic_equation_from_root_properties_l1200_120039

theorem quadratic_equation_from_root_properties (a b c : ℝ) :
  (∀ x y : ℝ, x + y = 10 ∧ x * y = 24 → a * x^2 + b * x + c = 0) →
  a = 1 ∧ b = -10 ∧ c = 24 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_root_properties_l1200_120039


namespace NUMINAMATH_CALUDE_root_property_l1200_120070

theorem root_property (a : ℝ) : a^2 - 2*a - 5 = 0 → 2*a^2 - 4*a = 10 := by
  sorry

end NUMINAMATH_CALUDE_root_property_l1200_120070


namespace NUMINAMATH_CALUDE_sunset_time_correct_l1200_120080

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents the length of daylight -/
structure DaylightLength where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the sunset time given sunrise time and daylight length -/
def calculate_sunset (sunrise : Time) (daylight : DaylightLength) : Time :=
  sorry

theorem sunset_time_correct (sunrise : Time) (daylight : DaylightLength) :
  sunrise.hours = 6 ∧ sunrise.minutes = 43 ∧
  daylight.hours = 11 ∧ daylight.minutes = 56 →
  let sunset := calculate_sunset sunrise daylight
  sunset.hours = 18 ∧ sunset.minutes = 39 :=
sorry

end NUMINAMATH_CALUDE_sunset_time_correct_l1200_120080


namespace NUMINAMATH_CALUDE_least_x_for_divisibility_by_three_l1200_120031

theorem least_x_for_divisibility_by_three : 
  (∃ x : ℕ, ∀ n : ℕ, (n * 57) % 3 = 0) ∧ 
  (∀ y : ℕ, y < 0 → ¬(∀ n : ℕ, (n * 57) % 3 = 0)) := by sorry

end NUMINAMATH_CALUDE_least_x_for_divisibility_by_three_l1200_120031


namespace NUMINAMATH_CALUDE_adult_ticket_price_l1200_120060

/-- Given information about ticket sales, prove the price of an adult ticket --/
theorem adult_ticket_price
  (student_price : ℝ)
  (total_tickets : ℕ)
  (total_revenue : ℝ)
  (student_tickets : ℕ)
  (h1 : student_price = 2.5)
  (h2 : total_tickets = 59)
  (h3 : total_revenue = 222.5)
  (h4 : student_tickets = 9) :
  (total_revenue - student_price * student_tickets) / (total_tickets - student_tickets) = 4 :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l1200_120060


namespace NUMINAMATH_CALUDE_average_weight_increase_l1200_120087

theorem average_weight_increase (initial_count : ℕ) (replaced_weight new_weight : ℝ) :
  initial_count = 8 →
  replaced_weight = 70 →
  new_weight = 94 →
  (new_weight - replaced_weight) / initial_count = 3 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l1200_120087


namespace NUMINAMATH_CALUDE_project_completion_time_l1200_120036

/-- Calculates the number of days needed to complete a project given extra hours,
    normal work hours, and total project hours. -/
def days_to_complete_project (extra_hours : ℕ) (normal_hours : ℕ) (project_hours : ℕ) : ℕ :=
  project_hours / (normal_hours + extra_hours)

/-- Theorem stating that under the given conditions, it takes 100 days to complete the project. -/
theorem project_completion_time :
  days_to_complete_project 5 10 1500 = 100 := by
  sorry

#eval days_to_complete_project 5 10 1500

end NUMINAMATH_CALUDE_project_completion_time_l1200_120036


namespace NUMINAMATH_CALUDE_intersection_of_given_sets_l1200_120059

theorem intersection_of_given_sets :
  let A : Set ℕ := {1, 3, 4}
  let B : Set ℕ := {3, 4, 5}
  A ∩ B = {3, 4} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_given_sets_l1200_120059


namespace NUMINAMATH_CALUDE_exists_number_with_specific_digit_sums_l1200_120090

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number n such that the sum of its digits is 100
    and the sum of the digits of n^3 is 1,000,000 -/
theorem exists_number_with_specific_digit_sums :
  ∃ n : ℕ, sum_of_digits n = 100 ∧ sum_of_digits (n^3) = 1000000 := by sorry

end NUMINAMATH_CALUDE_exists_number_with_specific_digit_sums_l1200_120090


namespace NUMINAMATH_CALUDE_inequality_holds_iff_x_leq_3_l1200_120003

theorem inequality_holds_iff_x_leq_3 (x : ℕ+) :
  (x + 1 : ℚ) / 3 - (2 * x - 1 : ℚ) / 4 ≥ (x - 3 : ℚ) / 6 ↔ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_x_leq_3_l1200_120003


namespace NUMINAMATH_CALUDE_trout_division_l1200_120015

theorem trout_division (total_trout : ℕ) (num_people : ℕ) (trout_per_person : ℕ) : 
  total_trout = 18 → num_people = 2 → trout_per_person = total_trout / num_people → trout_per_person = 9 := by
  sorry

end NUMINAMATH_CALUDE_trout_division_l1200_120015


namespace NUMINAMATH_CALUDE_total_salary_is_583_l1200_120019

/-- The total amount paid to two employees per week, given their relative salaries -/
def total_salary (n_salary : ℝ) : ℝ :=
  n_salary + 1.2 * n_salary

/-- Proof that the total salary for two employees is $583 per week -/
theorem total_salary_is_583 :
  total_salary 265 = 583 := by
  sorry

end NUMINAMATH_CALUDE_total_salary_is_583_l1200_120019


namespace NUMINAMATH_CALUDE_parametric_equations_represent_line_l1200_120002

/-- Proves that the given parametric equations represent the straight line 2x - y + 1 = 0 -/
theorem parametric_equations_represent_line :
  ∀ (t : ℝ), 2 * (1 - t) - (3 - 2*t) + 1 = 0 := by
  sorry

#check parametric_equations_represent_line

end NUMINAMATH_CALUDE_parametric_equations_represent_line_l1200_120002


namespace NUMINAMATH_CALUDE_closest_fraction_to_one_l1200_120006

theorem closest_fraction_to_one : 
  let fractions : List ℚ := [7/8, 8/7, 9/10, 10/11, 11/10]
  ∀ f ∈ fractions, |10/11 - 1| ≤ |f - 1| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_fraction_to_one_l1200_120006


namespace NUMINAMATH_CALUDE_valid_selections_count_l1200_120058

/-- The number of students in the group -/
def total_students : ℕ := 7

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of ways to select 4 students from 7, where at least one of A and B participates,
    and when both participate, their speeches are not adjacent -/
def valid_selections : ℕ := sorry

theorem valid_selections_count : valid_selections = 600 := by sorry

end NUMINAMATH_CALUDE_valid_selections_count_l1200_120058


namespace NUMINAMATH_CALUDE_total_discount_percentage_l1200_120041

theorem total_discount_percentage (initial_discount subsequent_discount : ℝ) : 
  initial_discount = 0.25 → 
  subsequent_discount = 0.35 → 
  1 - (1 - initial_discount) * (1 - subsequent_discount) = 0.5125 := by
sorry

end NUMINAMATH_CALUDE_total_discount_percentage_l1200_120041


namespace NUMINAMATH_CALUDE_frog_jump_distance_l1200_120025

theorem frog_jump_distance (grasshopper_distance : ℕ) (difference : ℕ) (frog_distance : ℕ) :
  grasshopper_distance = 13 →
  difference = 2 →
  grasshopper_distance = frog_distance + difference →
  frog_distance = 11 := by
sorry

end NUMINAMATH_CALUDE_frog_jump_distance_l1200_120025


namespace NUMINAMATH_CALUDE_quadratic_non_real_roots_l1200_120075

theorem quadratic_non_real_roots (b : ℝ) : 
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by sorry

end NUMINAMATH_CALUDE_quadratic_non_real_roots_l1200_120075


namespace NUMINAMATH_CALUDE_completing_square_l1200_120095

theorem completing_square (x : ℝ) : x^2 + 2*x - 3 = 0 ↔ (x + 1)^2 = 4 := by sorry

end NUMINAMATH_CALUDE_completing_square_l1200_120095


namespace NUMINAMATH_CALUDE_inequality_proof_l1200_120008

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a^2 + b^2 + c^2 + (a + b + c)^2 ≤ 4) :
  (a*b + 1)/(a + b)^2 + (b*c + 1)/(b + c)^2 + (c*a + 1)/(c + a)^2 ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1200_120008


namespace NUMINAMATH_CALUDE_cubic_minus_x_factorization_l1200_120044

theorem cubic_minus_x_factorization (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_x_factorization_l1200_120044


namespace NUMINAMATH_CALUDE_brownies_per_pan_l1200_120086

/-- Proves that the number of pieces in each pan of brownies is 16 given the problem conditions --/
theorem brownies_per_pan (total_pans : ℕ) (eaten_pans : ℚ) (ice_cream_tubs : ℕ) 
  (scoops_per_tub : ℕ) (scoops_per_guest : ℕ) (guests_without_ice_cream : ℕ) :
  total_pans = 2 →
  eaten_pans = 1 + 3/4 →
  scoops_per_tub = 8 →
  scoops_per_guest = 2 →
  ice_cream_tubs = 6 →
  guests_without_ice_cream = 4 →
  ∃ (pieces_per_pan : ℕ), pieces_per_pan = 16 ∧ 
    (ice_cream_tubs * scoops_per_tub / scoops_per_guest + guests_without_ice_cream) / eaten_pans = pieces_per_pan := by
  sorry

#check brownies_per_pan

end NUMINAMATH_CALUDE_brownies_per_pan_l1200_120086


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l1200_120088

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- First asymptote: y = x + 2 -/
  asymptote1 : ℝ → ℝ
  /-- Second asymptote: y = 4 - x -/
  asymptote2 : ℝ → ℝ
  /-- The hyperbola passes through the point (4, 4) -/
  passes_through : ℝ × ℝ
  /-- Conditions for the asymptotes -/
  h_asymptote1 : ∀ x, asymptote1 x = x + 2
  h_asymptote2 : ∀ x, asymptote2 x = 4 - x
  h_passes_through : passes_through = (4, 4)

/-- The distance between the foci of the hyperbola -/
def foci_distance (h : Hyperbola) : ℝ := 8

/-- Theorem stating that the distance between the foci of the given hyperbola is 8 -/
theorem hyperbola_foci_distance (h : Hyperbola) :
  foci_distance h = 8 := by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l1200_120088


namespace NUMINAMATH_CALUDE_four_row_arrangement_has_fourteen_triangles_l1200_120050

/-- Represents a triangular arrangement of smaller triangles. -/
structure TriangularArrangement where
  rows : Nat
  bottom_row_triangles : Nat

/-- Calculates the total number of triangles in the arrangement. -/
def total_triangles (arr : TriangularArrangement) : Nat :=
  sorry

/-- Theorem stating that a triangular arrangement with 4 rows and 4 triangles
    in the bottom row has a total of 14 triangles. -/
theorem four_row_arrangement_has_fourteen_triangles :
  ∀ (arr : TriangularArrangement),
    arr.rows = 4 →
    arr.bottom_row_triangles = 4 →
    total_triangles arr = 14 :=
  sorry

end NUMINAMATH_CALUDE_four_row_arrangement_has_fourteen_triangles_l1200_120050


namespace NUMINAMATH_CALUDE_expansion_properties_l1200_120016

open Real Nat

/-- Represents the expansion of (1 + 2∛x)^n -/
def expansion (n : ℕ) (x : ℝ) := (1 + 2 * x^(1/3))^n

/-- Coefficient of the r-th term in the expansion -/
def coefficient (n r : ℕ) : ℝ := 2^r * choose n r

/-- Condition for the coefficient relation -/
def coefficient_condition (n : ℕ) : Prop :=
  ∃ r, coefficient n r = 2 * coefficient n (r-1) ∧
       coefficient n r = 5/6 * coefficient n (r+1)

/-- Sum of all coefficients in the expansion -/
def sum_coefficients (n : ℕ) : ℝ := 3^n

/-- Sum of all binomial coefficients -/
def sum_binomial_coefficients (n : ℕ) : ℕ := 2^n

/-- Rational terms in the expansion -/
def rational_terms (n : ℕ) : List (ℝ × ℕ) :=
  [(1, 0), (560, 1), (448, 2), (2016, 3)]

theorem expansion_properties (n : ℕ) :
  coefficient_condition n →
  n = 7 ∧
  sum_coefficients n = 2187 ∧
  sum_binomial_coefficients n = 128 ∧
  rational_terms n = [(1, 0), (560, 1), (448, 2), (2016, 3)] :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l1200_120016


namespace NUMINAMATH_CALUDE_desk_lamp_profit_l1200_120062

/-- Profit function for desk lamp sales -/
def profit_function (n : ℝ) (x : ℝ) : ℝ := (x - 20) * (-10 * x + n)

/-- Theorem stating the maximum profit and corresponding selling price -/
theorem desk_lamp_profit (n : ℝ) :
  (profit_function n 25 = 120) →
  (n = 370) ∧
  (∀ x : ℝ, x > 32 → profit_function n x ≤ 160) :=
by sorry

end NUMINAMATH_CALUDE_desk_lamp_profit_l1200_120062


namespace NUMINAMATH_CALUDE_tom_search_days_l1200_120077

/-- Calculates the number of days Tom searched for an item given the daily rates and total cost -/
def search_days (initial_rate : ℕ) (initial_days : ℕ) (subsequent_rate : ℕ) (total_cost : ℕ) : ℕ :=
  let initial_cost := initial_rate * initial_days
  let remaining_cost := total_cost - initial_cost
  let additional_days := remaining_cost / subsequent_rate
  initial_days + additional_days

/-- Proves that Tom searched for 10 days given the specified rates and total cost -/
theorem tom_search_days :
  search_days 100 5 60 800 = 10 := by
  sorry

end NUMINAMATH_CALUDE_tom_search_days_l1200_120077


namespace NUMINAMATH_CALUDE_inequality_solution_set_min_mn_value_l1200_120021

def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

theorem inequality_solution_set (x : ℝ) :
  (f 1 x ≥ 4 - |x + 1|) ↔ (x ≤ -2 ∨ x ≥ 2) := by sorry

theorem min_mn_value (a m n : ℝ) :
  (∀ x, f a x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) →
  m > 0 →
  n > 0 →
  1/m + 1/(2*n) = a →
  ∀ k, m*n ≤ k → 2 ≤ k := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_min_mn_value_l1200_120021


namespace NUMINAMATH_CALUDE_irrational_cubic_roots_not_quadratic_roots_l1200_120099

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Predicate to check if a number is a root of a cubic polynomial -/
def is_root_cubic (x : ℝ) (p : CubicPolynomial) : Prop :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d = 0

/-- Predicate to check if a number is a root of a quadratic polynomial -/
def is_root_quadratic (x : ℝ) (q : QuadraticPolynomial) : Prop :=
  q.a * x^2 + q.b * x + q.c = 0

/-- Main theorem -/
theorem irrational_cubic_roots_not_quadratic_roots
  (p : CubicPolynomial)
  (h1 : ∃ x y z : ℝ, is_root_cubic x p ∧ is_root_cubic y p ∧ is_root_cubic z p)
  (h2 : ∀ x : ℝ, is_root_cubic x p → Irrational x)
  : ∀ q : QuadraticPolynomial, ∀ x : ℝ, is_root_cubic x p → ¬ is_root_quadratic x q :=
sorry

end NUMINAMATH_CALUDE_irrational_cubic_roots_not_quadratic_roots_l1200_120099


namespace NUMINAMATH_CALUDE_double_mean_value_range_l1200_120068

/-- A function is a double mean value function on an interval [a,b] if there exist
    x₁ and x₂ in (a,b) such that f'(x₁) = f'(x₂) = (f(b) - f(a)) / (b - a) -/
def IsDoubleMeanValueFunction (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b ∧
    deriv f x₁ = (f b - f a) / (b - a) ∧
    deriv f x₂ = (f b - f a) / (b - a)

/-- The main theorem: if f(x) = x³ - 6/5x² is a double mean value function on [0,t],
    then 3/5 < t < 6/5 -/
theorem double_mean_value_range (t : ℝ) :
  IsDoubleMeanValueFunction (fun x => x^3 - 6/5*x^2) 0 t →
  3/5 < t ∧ t < 6/5 := by
  sorry

end NUMINAMATH_CALUDE_double_mean_value_range_l1200_120068


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l1200_120067

theorem quadratic_roots_properties (x₁ x₂ k m : ℝ) : 
  x₁ + x₂ + x₁ * x₂ = 2 * m + k →
  (x₁ - 1) * (x₂ - 1) = m + 1 - k →
  x₁ - x₂ = 1 →
  k - m = 1 →
  (k^2 > 4 * m) ∧ ((m = 0 ∧ k = 1) ∨ (m = 2 ∧ k = 3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l1200_120067


namespace NUMINAMATH_CALUDE_abs_a_minus_sqrt_a_squared_l1200_120098

theorem abs_a_minus_sqrt_a_squared (a : ℝ) (h : a < 0) : |a - Real.sqrt (a^2)| = -2*a := by
  sorry

end NUMINAMATH_CALUDE_abs_a_minus_sqrt_a_squared_l1200_120098


namespace NUMINAMATH_CALUDE_smallest_positive_integer_form_l1200_120027

theorem smallest_positive_integer_form (m n : ℤ) :
  (∃ k : ℕ+, k = |4509 * m + 27981 * n| ∧ 
   ∀ j : ℕ+, (∃ a b : ℤ, j = |4509 * a + 27981 * b|) → k ≤ j) ↔ 
  Nat.gcd 4509 27981 = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_form_l1200_120027


namespace NUMINAMATH_CALUDE_problem_zeros_count_l1200_120092

/-- The number of zeros in the binary representation of a natural number -/
def countZeros (n : ℕ) : ℕ := sorry

/-- The expression given in the problem -/
def problemExpression : ℕ := 
  ((18 * 8192 + 8 * 128 - 12 * 16) / 6 + 4 * 64 + 3^5 - (25 * 2))

/-- Theorem stating that the number of zeros in the binary representation of the problem expression is 6 -/
theorem problem_zeros_count : countZeros problemExpression = 6 := by sorry

end NUMINAMATH_CALUDE_problem_zeros_count_l1200_120092


namespace NUMINAMATH_CALUDE_matrix_sum_proof_l1200_120009

theorem matrix_sum_proof :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, 3; -2, 1]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-1, 5; 8, -3]
  A + B = !![3, 8; 6, -2] := by
  sorry

end NUMINAMATH_CALUDE_matrix_sum_proof_l1200_120009


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1200_120005

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0) ↔ (m ≤ 4 ∧ m ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1200_120005


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1200_120064

/-- Given a geometric sequence {a_n} with common ratio q = 2 and S_3 = 7, S_6 = 63 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- geometric sequence with common ratio 2
  (a 1 * (1 - 2^3)) / (1 - 2) = 7 →  -- S_3 = 7
  (a 1 * (1 - 2^6)) / (1 - 2) = 63 :=  -- S_6 = 63
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1200_120064


namespace NUMINAMATH_CALUDE_problem_factory_daily_production_l1200_120022

/-- A factory that produces toys -/
structure ToyFactory where
  weekly_production : ℕ
  working_days : ℕ
  daily_production : ℕ
  h1 : weekly_production = working_days * daily_production

/-- The specific toy factory in the problem -/
def problem_factory : ToyFactory where
  weekly_production := 8000
  working_days := 4
  daily_production := 2000
  h1 := rfl

/-- Theorem stating that the daily production of the problem factory is 2000 toys -/
theorem problem_factory_daily_production :
  problem_factory.daily_production = 2000 := by sorry

end NUMINAMATH_CALUDE_problem_factory_daily_production_l1200_120022


namespace NUMINAMATH_CALUDE_b_amount_l1200_120079

theorem b_amount (a b : ℚ) 
  (h1 : a + b = 2530)
  (h2 : (3/5) * a = (2/7) * b) : 
  b = 1714 := by sorry

end NUMINAMATH_CALUDE_b_amount_l1200_120079


namespace NUMINAMATH_CALUDE_machine_production_l1200_120052

/-- Given the production rate of 6 machines, calculate the production of 8 machines in 4 minutes -/
theorem machine_production 
  (rate : ℕ) -- Production rate per minute for 6 machines
  (h1 : rate = 270) -- 6 machines produce 270 bottles per minute
  : (8 * 4 * (rate / 6) : ℕ) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_machine_production_l1200_120052


namespace NUMINAMATH_CALUDE_farm_area_and_planned_days_correct_l1200_120026

/-- Represents the farm field and ploughing scenario -/
structure FarmField where
  planned_daily_area : ℝ
  actual_daily_area : ℝ
  type_a_percentage : ℝ
  type_b_percentage : ℝ
  type_c_percentage : ℝ
  type_a_hours_per_hectare : ℝ
  type_b_hours_per_hectare : ℝ
  type_c_hours_per_hectare : ℝ
  extra_days_worked : ℕ
  area_left_to_plough : ℝ
  max_hours_per_day : ℝ

/-- Calculates the total area of the farm field and the initially planned work days -/
def calculate_farm_area_and_planned_days (field : FarmField) : ℝ × ℕ :=
  sorry

/-- Theorem stating the correct total area and initially planned work days -/
theorem farm_area_and_planned_days_correct (field : FarmField) 
  (h1 : field.planned_daily_area = 260)
  (h2 : field.actual_daily_area = 85)
  (h3 : field.type_a_percentage = 0.4)
  (h4 : field.type_b_percentage = 0.3)
  (h5 : field.type_c_percentage = 0.3)
  (h6 : field.type_a_hours_per_hectare = 4)
  (h7 : field.type_b_hours_per_hectare = 6)
  (h8 : field.type_c_hours_per_hectare = 3)
  (h9 : field.extra_days_worked = 2)
  (h10 : field.area_left_to_plough = 40)
  (h11 : field.max_hours_per_day = 12) :
  calculate_farm_area_and_planned_days field = (340, 2) :=
by
  sorry

end NUMINAMATH_CALUDE_farm_area_and_planned_days_correct_l1200_120026


namespace NUMINAMATH_CALUDE_perpendicular_chords_sum_l1200_120072

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define a chord passing through the focus
structure ChordThroughFocus where
  a : PointOnParabola
  b : PointOnParabola
  passes_through_focus : True  -- We assume this property without proving it

-- Define perpendicular chords
def perpendicular (c1 c2 : ChordThroughFocus) : Prop := True  -- We assume this property without proving it

-- Define the length of a chord
noncomputable def chord_length (c : ChordThroughFocus) : ℝ := sorry

-- Theorem statement
theorem perpendicular_chords_sum (ab cd : ChordThroughFocus) 
  (h_perp : perpendicular ab cd) : 
  1 / chord_length ab + 1 / chord_length cd = 1/4 := by sorry

end NUMINAMATH_CALUDE_perpendicular_chords_sum_l1200_120072


namespace NUMINAMATH_CALUDE_a_share_fraction_l1200_120046

/-- Prove that given the conditions, A's share is 2/3 of B and C's combined share -/
theorem a_share_fraction (total money : ℝ) (a_share : ℝ) (b_share : ℝ) (c_share : ℝ) : 
  total = 300 →
  a_share = 120.00000000000001 →
  b_share = (6/9) * (a_share + c_share) →
  total = a_share + b_share + c_share →
  a_share = (2/3) * (b_share + c_share) :=
by sorry


end NUMINAMATH_CALUDE_a_share_fraction_l1200_120046


namespace NUMINAMATH_CALUDE_rhombus_area_l1200_120054

-- Define the rhombus
def Rhombus (perimeter : ℝ) (diagonal1 : ℝ) : Prop :=
  perimeter > 0 ∧ diagonal1 > 0

-- Theorem statement
theorem rhombus_area 
  (perimeter : ℝ) 
  (diagonal1 : ℝ) 
  (h : Rhombus perimeter diagonal1) 
  (h_perimeter : perimeter = 80) 
  (h_diagonal : diagonal1 = 36) : 
  ∃ (area : ℝ), area = 72 * Real.sqrt 19 :=
sorry

end NUMINAMATH_CALUDE_rhombus_area_l1200_120054


namespace NUMINAMATH_CALUDE_log_product_theorem_l1200_120069

theorem log_product_theorem (c d : ℕ+) : 
  (d - c = 450) →
  (Real.log d / Real.log c = 3) →
  (c + d = 520) := by sorry

end NUMINAMATH_CALUDE_log_product_theorem_l1200_120069


namespace NUMINAMATH_CALUDE_george_room_painting_choices_l1200_120000

theorem george_room_painting_choices :
  (Nat.choose 10 3) * 5 = 600 := by sorry

end NUMINAMATH_CALUDE_george_room_painting_choices_l1200_120000


namespace NUMINAMATH_CALUDE_nina_money_problem_l1200_120012

theorem nina_money_problem (x : ℚ) :
  (5 * x = 8 * (x - 1.25)) → (5 * x = 50 / 3) := by
  sorry

end NUMINAMATH_CALUDE_nina_money_problem_l1200_120012


namespace NUMINAMATH_CALUDE_sam_distance_l1200_120001

theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) 
  (h1 : marguerite_distance = 150)
  (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) : 
  (marguerite_distance / marguerite_time) * sam_time = 200 := by
  sorry

end NUMINAMATH_CALUDE_sam_distance_l1200_120001


namespace NUMINAMATH_CALUDE_number_of_people_liking_apple_l1200_120056

/-- The number of people who like apple -/
def like_apple : ℕ := 40

/-- The number of people who like orange and mango but dislike apple -/
def like_orange_mango_not_apple : ℕ := 7

/-- The number of people who like mango and apple but dislike orange -/
def like_mango_apple_not_orange : ℕ := 10

/-- The number of people who like all three fruits -/
def like_all : ℕ := 4

/-- Theorem stating that the number of people who like apple is 40 -/
theorem number_of_people_liking_apple : 
  like_apple = 40 := by sorry

end NUMINAMATH_CALUDE_number_of_people_liking_apple_l1200_120056


namespace NUMINAMATH_CALUDE_division_problem_l1200_120093

theorem division_problem : (501 : ℝ) / (0.5 : ℝ) = 1002 := by sorry

end NUMINAMATH_CALUDE_division_problem_l1200_120093


namespace NUMINAMATH_CALUDE_other_x_intercept_l1200_120004

/-- A quadratic function with vertex (5, 10) and one x-intercept at (-1, 0) -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : a ≠ 0 → -b / (2 * a) = 5
  vertex_y : a ≠ 0 → a * 5^2 + b * 5 + c = 10
  x_intercept : a * (-1)^2 + b * (-1) + c = 0

/-- The x-coordinate of the other x-intercept is 11 -/
theorem other_x_intercept (f : QuadraticFunction) :
  ∃ x : ℝ, x ≠ -1 ∧ f.a * x^2 + f.b * x + f.c = 0 ∧ x = 11 :=
sorry

end NUMINAMATH_CALUDE_other_x_intercept_l1200_120004


namespace NUMINAMATH_CALUDE_friend_team_assignments_l1200_120040

theorem friend_team_assignments (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  k ^ n = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friend_team_assignments_l1200_120040
