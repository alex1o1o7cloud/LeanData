import Mathlib

namespace expected_boy_girl_adjacencies_l2902_290279

/-- The expected number of boy-girl adjacencies in a random arrangement of boys and girls -/
theorem expected_boy_girl_adjacencies
  (num_boys : ℕ)
  (num_girls : ℕ)
  (total : ℕ)
  (h_total : total = num_boys + num_girls)
  (h_boys : num_boys = 7)
  (h_girls : num_girls = 13) :
  (total - 1 : ℚ) * (num_boys * num_girls : ℚ) / (total * (total - 1) / 2 : ℚ) = 91 / 10 :=
sorry

end expected_boy_girl_adjacencies_l2902_290279


namespace parrots_per_cage_l2902_290241

theorem parrots_per_cage (num_cages : ℕ) (parakeets_per_cage : ℕ) (total_birds : ℕ) :
  num_cages = 8 →
  parakeets_per_cage = 7 →
  total_birds = 72 →
  ∃ (parrots_per_cage : ℕ),
    parrots_per_cage * num_cages + parakeets_per_cage * num_cages = total_birds ∧
    parrots_per_cage = 2 :=
by sorry

end parrots_per_cage_l2902_290241


namespace gravel_calculation_l2902_290237

/-- The amount of gravel bought by a construction company -/
def gravel_amount : ℝ := 14.02 - 8.11

/-- The total amount of material bought by the construction company -/
def total_material : ℝ := 14.02

/-- The amount of sand bought by the construction company -/
def sand_amount : ℝ := 8.11

theorem gravel_calculation :
  gravel_amount = 5.91 ∧
  total_material = gravel_amount + sand_amount :=
sorry

end gravel_calculation_l2902_290237


namespace geometry_biology_overlap_difference_l2902_290225

theorem geometry_biology_overlap_difference (total : ℕ) (geometry : ℕ) (biology : ℕ)
  (h1 : total = 232)
  (h2 : geometry = 144)
  (h3 : biology = 119) :
  (min geometry biology) - (geometry + biology - total) = 88 :=
by sorry

end geometry_biology_overlap_difference_l2902_290225


namespace book_pages_l2902_290272

/-- The number of pages Hallie read on the first day -/
def pages_day1 : ℕ := 63

/-- The number of pages Hallie read on the second day -/
def pages_day2 : ℕ := 2 * pages_day1

/-- The number of pages Hallie read on the third day -/
def pages_day3 : ℕ := pages_day2 + 10

/-- The number of pages Hallie read on the fourth day -/
def pages_day4 : ℕ := 29

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_day1 + pages_day2 + pages_day3 + pages_day4

theorem book_pages : total_pages = 354 := by sorry

end book_pages_l2902_290272


namespace min_value_expression_l2902_290228

theorem min_value_expression (x y : ℝ) : x^2 + y^2 - 6*x + 4*y + 18 ≥ 5 := by
  sorry

end min_value_expression_l2902_290228


namespace pierre_birthday_l2902_290277

/-- Represents a date with day and month -/
structure Date where
  day : Nat
  month : Nat

/-- Represents a person's age and birthday -/
structure Person where
  age : Nat
  birthday : Date

def nextYear (d : Date) : Date :=
  if d.month = 12 && d.day = 31 then { day := 1, month := 1 }
  else { day := d.day, month := d.month }

def yesterday (d : Date) : Date :=
  if d.day = 1 && d.month = 1 then { day := 31, month := 12 }
  else if d.day = 1 then { day := 31, month := d.month - 1 }
  else { day := d.day - 1, month := d.month }

def dayBeforeYesterday (d : Date) : Date := yesterday (yesterday d)

theorem pierre_birthday (today : Date) (pierre : Person) : 
  pierre.age = 11 → 
  (dayBeforeYesterday today).day = 31 → 
  (dayBeforeYesterday today).month = 12 →
  pierre.birthday = yesterday today →
  (nextYear today).day = 1 → 
  (nextYear today).month = 1 →
  today.day = 1 ∧ today.month = 1 := by
  sorry

#check pierre_birthday

end pierre_birthday_l2902_290277


namespace first_half_total_score_l2902_290242

/-- Represents the scores of a team in a basketball game --/
structure TeamScores where
  q1 : ℝ
  q2 : ℝ
  q3 : ℝ
  q4 : ℝ

/-- The game conditions --/
def GameConditions (alpha : TeamScores) (beta : TeamScores) : Prop :=
  -- Tied after first quarter
  alpha.q1 = beta.q1
  -- Alpha's scores form a geometric sequence
  ∧ ∃ r : ℝ, r > 1 ∧ alpha.q2 = alpha.q1 * r ∧ alpha.q3 = alpha.q2 * r ∧ alpha.q4 = alpha.q3 * r
  -- Beta's scores form an arithmetic sequence
  ∧ ∃ d : ℝ, d > 0 ∧ beta.q2 = beta.q1 + d ∧ beta.q3 = beta.q2 + d ∧ beta.q4 = beta.q3 + d
  -- Alpha won by 3 points
  ∧ alpha.q1 + alpha.q2 + alpha.q3 + alpha.q4 = beta.q1 + beta.q2 + beta.q3 + beta.q4 + 3
  -- No team scored more than 120 points
  ∧ alpha.q1 + alpha.q2 + alpha.q3 + alpha.q4 ≤ 120
  ∧ beta.q1 + beta.q2 + beta.q3 + beta.q4 ≤ 120

/-- The theorem to be proved --/
theorem first_half_total_score (alpha : TeamScores) (beta : TeamScores) 
  (h : GameConditions alpha beta) : 
  alpha.q1 + alpha.q2 + beta.q1 + beta.q2 = 35.5 := by
  sorry

end first_half_total_score_l2902_290242


namespace sin_2alpha_value_l2902_290212

theorem sin_2alpha_value (α : Real) (h : Real.sin α + 2 * Real.cos α = 0) :
  Real.sin (2 * α) = -4/5 := by
  sorry

end sin_2alpha_value_l2902_290212


namespace total_bottles_bought_l2902_290249

-- Define the variables
def bottles_per_day : ℕ := 9
def days_lasted : ℕ := 17

-- Define the theorem
theorem total_bottles_bought : 
  bottles_per_day * days_lasted = 153 := by
  sorry

end total_bottles_bought_l2902_290249


namespace problem_statement_l2902_290205

theorem problem_statement (a : ℝ) (h : a^2 + a = 0) : 4*a^2 + 4*a + 2011 = 2011 := by
  sorry

end problem_statement_l2902_290205


namespace multiplier_satisfies_equation_l2902_290213

/-- The multiplier that satisfies the equation when the number is 5.0 -/
def multiplier : ℝ := 7

/-- The given number in the problem -/
def number : ℝ := 5.0

/-- Theorem stating that the multiplier satisfies the equation -/
theorem multiplier_satisfies_equation : 
  4 * number + multiplier * number = 55 := by sorry

end multiplier_satisfies_equation_l2902_290213


namespace min_r_for_B_subset_C_l2902_290276

open Set Real

-- Define the sets A, B, and C(r)
def A : Set ℝ := {t | 0 < t ∧ t < 2 * π}

def B : Set (ℝ × ℝ) := {p | ∃ t ∈ A, p.1 = sin t ∧ p.2 = 2 * sin t * cos t}

def C (r : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ r^2 ∧ r > 0}

-- State the theorem
theorem min_r_for_B_subset_C : 
  (∀ r, B ⊆ C r → r ≥ 5/4) ∧ B ⊆ C (5/4) := by sorry

end min_r_for_B_subset_C_l2902_290276


namespace tan_ratio_from_sin_sum_diff_l2902_290293

theorem tan_ratio_from_sin_sum_diff (a b : ℝ) 
  (h1 : Real.sin (a + b) = 5/8) 
  (h2 : Real.sin (a - b) = 1/4) : 
  Real.tan a / Real.tan b = 7/3 := by
  sorry

end tan_ratio_from_sin_sum_diff_l2902_290293


namespace arithmetic_calculations_l2902_290204

theorem arithmetic_calculations :
  ((-7) * (-5) - 90 / (-15) = 41) ∧
  ((-1)^10 * 2 - (-2)^3 / 4 = 4) := by
  sorry

end arithmetic_calculations_l2902_290204


namespace beavers_still_working_l2902_290275

def initial_beavers : ℕ := 7
def swimming_beavers : ℕ := 2
def stick_collecting_beaver : ℕ := 1
def food_searching_beaver : ℕ := 1

theorem beavers_still_working : ℕ := by
  sorry

end beavers_still_working_l2902_290275


namespace circle_intersection_theorem_l2902_290273

-- Define the points A, B, and C
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, 2)
def C : ℝ × ℝ := (1, -7)

-- Define circle M passing through A, B, and C
def circle_M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 25}

-- Define the y-axis
def y_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0}

-- Define the line on which the center of circle N moves
def center_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + 6 = 0}

-- Define circle N with radius 10 and center (a, 2a + 6)
def circle_N (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - (2 * a + 6))^2 = 100}

-- Define the theorem
theorem circle_intersection_theorem :
  ∃ (P Q : ℝ × ℝ) (a : Set ℝ),
    P ∈ circle_M ∧ P ∈ y_axis ∧
    Q ∈ circle_M ∧ Q ∈ y_axis ∧
    (Q.2 - P.2)^2 = 96 ∧
    (∀ x ∈ a, ∃ y, (x, y) ∈ center_line ∧ (circle_N x ∩ circle_M).Nonempty) ∧
    a = {x : ℝ | -3 - Real.sqrt 41 ≤ x ∧ x ≤ -4 ∨ -2 ≤ x ∧ x ≤ -3 + Real.sqrt 41} :=
sorry

end circle_intersection_theorem_l2902_290273


namespace product_equality_l2902_290298

theorem product_equality : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := by
  sorry

end product_equality_l2902_290298


namespace sequence_growth_l2902_290215

theorem sequence_growth (a : ℕ → ℕ) (h1 : a 1 > a 0) 
  (h2 : ∀ n : ℕ, n ≥ 2 → a n = 3 * a (n - 1) - 2 * a (n - 2)) :
  a 100 > 2^99 := by
  sorry

end sequence_growth_l2902_290215


namespace floor_range_l2902_290256

theorem floor_range (x : ℝ) : 
  Int.floor x = -3 → -3 ≤ x ∧ x < -2 := by sorry

end floor_range_l2902_290256


namespace irrational_functional_equation_implies_constant_l2902_290278

/-- A function satisfying f(ab) = f(a+b) for all irrational a and b -/
def IrrationalFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, Irrational a → Irrational b → f (a * b) = f (a + b)

/-- Theorem: If a function satisfies the irrational functional equation, then it is constant -/
theorem irrational_functional_equation_implies_constant
  (f : ℝ → ℝ) (h : IrrationalFunctionalEquation f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c := by
  sorry

end irrational_functional_equation_implies_constant_l2902_290278


namespace kevins_weekly_revenue_l2902_290239

/-- Calculates the total weekly revenue for a fruit vendor --/
def total_weekly_revenue (total_crates price_grapes price_mangoes price_passion 
                          crates_grapes crates_mangoes : ℕ) : ℕ :=
  let crates_passion := total_crates - crates_grapes - crates_mangoes
  let revenue_grapes := crates_grapes * price_grapes
  let revenue_mangoes := crates_mangoes * price_mangoes
  let revenue_passion := crates_passion * price_passion
  revenue_grapes + revenue_mangoes + revenue_passion

/-- Theorem stating that Kevin's total weekly revenue is $1020 --/
theorem kevins_weekly_revenue : 
  total_weekly_revenue 50 15 20 25 13 20 = 1020 := by
  sorry

#eval total_weekly_revenue 50 15 20 25 13 20

end kevins_weekly_revenue_l2902_290239


namespace students_above_90_l2902_290254

/-- Represents a normal distribution of test scores -/
structure ScoreDistribution where
  mean : ℝ
  variance : ℝ
  is_normal : Bool

/-- Represents the class and score information -/
structure ClassScores where
  total_students : ℕ
  distribution : ScoreDistribution
  between_mean_and_plus_10 : ℕ

/-- Theorem stating the number of students scoring above 90 -/
theorem students_above_90 (c : ClassScores) 
  (h1 : c.total_students = 48)
  (h2 : c.distribution.mean = 80)
  (h3 : c.distribution.is_normal = true)
  (h4 : c.between_mean_and_plus_10 = 16) :
  c.total_students / 2 - c.between_mean_and_plus_10 = 8 := by
  sorry


end students_above_90_l2902_290254


namespace production_volume_proof_l2902_290271

/-- Represents the production volume equation over three years -/
def production_equation (x : ℝ) : Prop :=
  200 + 200 * (1 + x) + 200 * (1 + x)^2 = 1400

/-- 
  Proves that the given equation correctly represents the total production volume
  over three years, given an initial production of 200 units and a constant
  percentage increase x for two consecutive years, resulting in a total of 1400 units.
-/
theorem production_volume_proof (x : ℝ) : production_equation x := by
  sorry

end production_volume_proof_l2902_290271


namespace f_composition_equals_three_l2902_290268

noncomputable def f (x : ℂ) : ℂ :=
  if x.im = 0 then 1 + x else (1 - Complex.I) / Complex.abs Complex.I * x

theorem f_composition_equals_three :
  f (f (1 + Complex.I)) = 3 := by sorry

end f_composition_equals_three_l2902_290268


namespace fraction_meaningful_l2902_290270

theorem fraction_meaningful (a : ℝ) : 
  (∃ x : ℝ, x = (a + 1) / (2 * a - 1)) ↔ a ≠ 1/2 :=
sorry

end fraction_meaningful_l2902_290270


namespace jar_red_marble_difference_l2902_290258

-- Define the ratios for each jar
def jar_a_ratio : Rat := 5 / 3
def jar_b_ratio : Rat := 3 / 2

-- Define the total number of white marbles
def total_white_marbles : ℕ := 70

-- Theorem statement
theorem jar_red_marble_difference :
  ∃ (total_marbles : ℕ) (jar_a_red jar_a_white jar_b_red jar_b_white : ℕ),
    -- Both jars have equal number of marbles
    jar_a_red + jar_a_white = total_marbles ∧
    jar_b_red + jar_b_white = total_marbles ∧
    -- Ratio conditions
    jar_a_red / jar_a_white = jar_a_ratio ∧
    jar_b_red / jar_b_white = jar_b_ratio ∧
    -- Total white marbles condition
    jar_a_white + jar_b_white = total_white_marbles ∧
    -- Difference in red marbles
    jar_a_red - jar_b_red = 2 :=
by
  sorry

end jar_red_marble_difference_l2902_290258


namespace function_property_l2902_290257

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x, (x - 1) * (deriv f x) ≤ 0)
variable (h2 : ∀ x, f (x + 1) = f (-x + 1))

-- Define the theorem
theorem function_property (x₁ x₂ : ℝ) (h3 : |x₁ - 1| < |x₂ - 1|) :
  f (2 - x₁) ≥ f (2 - x₂) := by
  sorry

end function_property_l2902_290257


namespace equal_to_mac_ratio_l2902_290236

/-- Represents the survey results of computer brand preferences among college students. -/
structure SurveyResults where
  total : ℕ
  mac_preference : ℕ
  no_preference : ℕ
  windows_preference : ℕ

/-- Calculates the number of students who equally preferred both brands. -/
def equal_preference (s : SurveyResults) : ℕ :=
  s.total - (s.mac_preference + s.no_preference + s.windows_preference)

/-- Represents a ratio as a pair of natural numbers. -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem stating the ratio of students who equally preferred both brands
    to students who preferred Mac to Windows. -/
theorem equal_to_mac_ratio (s : SurveyResults)
  (h_total : s.total = 210)
  (h_mac : s.mac_preference = 60)
  (h_no_pref : s.no_preference = 90)
  (h_windows : s.windows_preference = 40) :
  ∃ (r : Ratio), r.numerator = 1 ∧ r.denominator = 3 ∧
  r.numerator * s.mac_preference = r.denominator * equal_preference s :=
sorry

end equal_to_mac_ratio_l2902_290236


namespace jan_drove_more_than_ian_l2902_290226

/-- Prove that Jan drove 174 miles more than Ian given the conditions --/
theorem jan_drove_more_than_ian (ian_time : ℝ) (ian_speed : ℝ) : 
  let han_time := ian_time + 1.5
  let han_speed := ian_speed + 6
  let jan_time := ian_time + 3
  let jan_speed := ian_speed + 8
  let ian_distance := ian_speed * ian_time
  let han_distance := han_speed * han_time
  han_distance - ian_distance = 84 →
  jan_speed * jan_time - ian_speed * ian_time = 174 :=
by sorry

end jan_drove_more_than_ian_l2902_290226


namespace pencil_count_l2902_290269

theorem pencil_count (initial : ℕ) (added : ℕ) (total : ℕ) : 
  initial = 27 → added = 45 → total = initial + added → total = 72 := by sorry

end pencil_count_l2902_290269


namespace at_least_one_is_one_l2902_290219

theorem at_least_one_is_one (x y z : ℝ) 
  (h1 : (1 / x) + (1 / y) + (1 / z) = 1) 
  (h2 : 1 / (x + y + z) = 1) : 
  x = 1 ∨ y = 1 ∨ z = 1 := by
sorry

end at_least_one_is_one_l2902_290219


namespace nine_point_centers_property_l2902_290210

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The intersection point of the diagonals -/
def diagonalIntersection (q : Quadrilateral) : Point :=
  sorry

/-- Checks if four points are collinear -/
def areCollinear (p1 p2 p3 p4 : Point) : Prop :=
  sorry

/-- Checks if four points form a parallelogram -/
def formParallelogram (p1 p2 p3 p4 : Point) : Prop :=
  sorry

/-- Computes the nine-point center of a triangle -/
def ninePointCenter (a b c : Point) : Point :=
  sorry

/-- The main theorem -/
theorem nine_point_centers_property (q : Quadrilateral) :
  let X := diagonalIntersection q
  let center1 := ninePointCenter X q.A q.B
  let center2 := ninePointCenter X q.B q.C
  let center3 := ninePointCenter X q.C q.D
  let center4 := ninePointCenter X q.D q.A
  areCollinear center1 center2 center3 center4 ∨ 
  formParallelogram center1 center2 center3 center4 := by
  sorry

end nine_point_centers_property_l2902_290210


namespace f_properties_l2902_290229

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x^2 + a*x|

-- Define the property of being monotonically increasing on [0,1]
def monotone_increasing_on_unit_interval (g : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ y ∧ y ≤ 1 → g x ≤ g y

-- Define M(a) as the maximum value of f(x) on [0,1]
noncomputable def M (a : ℝ) : ℝ :=
  ⨆ (x : ℝ) (h : x ∈ Set.Icc 0 1), f a x

-- State the theorem
theorem f_properties (a : ℝ) :
  (monotone_increasing_on_unit_interval (f a) ↔ a ≤ -2 ∨ a ≥ 0) ∧
  (∃ (a_min : ℝ), ∀ (a : ℝ), M a_min ≤ M a ∧ M a_min = 3 - 2 * Real.sqrt 2) :=
sorry

end f_properties_l2902_290229


namespace combined_prism_volume_l2902_290217

/-- The volume of a structure consisting of a triangular prism on top of a rectangular prism -/
theorem combined_prism_volume (rect_length rect_width rect_height tri_base tri_height tri_length : ℝ) :
  rect_length = 6 →
  rect_width = 4 →
  rect_height = 2 →
  tri_base = 3 →
  tri_height = 3 →
  tri_length = 4 →
  (rect_length * rect_width * rect_height) + (1/2 * tri_base * tri_height * tri_length) = 66 := by
  sorry

end combined_prism_volume_l2902_290217


namespace fixed_internet_charge_l2902_290234

/-- Represents Elvin's monthly telephone bill structure -/
structure MonthlyBill where
  callCharge : ℝ
  internetCharge : ℝ
  totalCharge : ℝ
  total_eq : totalCharge = callCharge + internetCharge

/-- Theorem stating the fixed monthly internet charge -/
theorem fixed_internet_charge 
  (jan : MonthlyBill) 
  (feb : MonthlyBill) 
  (jan_total : jan.totalCharge = 46)
  (feb_total : feb.totalCharge = 76)
  (feb_call_charge : feb.callCharge = 2 * jan.callCharge)
  : jan.internetCharge = 16 := by
  sorry

end fixed_internet_charge_l2902_290234


namespace ship_travel_ratio_l2902_290291

/-- Proves that the ratio of the distance traveled on day 2 to day 1 is 3:1 given the ship's travel conditions --/
theorem ship_travel_ratio : 
  ∀ (day1_distance day2_distance day3_distance : ℝ),
  day1_distance = 100 →
  day3_distance = day2_distance + 110 →
  day1_distance + day2_distance + day3_distance = 810 →
  day2_distance / day1_distance = 3 := by
sorry


end ship_travel_ratio_l2902_290291


namespace glasses_per_pitcher_l2902_290227

theorem glasses_per_pitcher (total_glasses : ℕ) (num_pitchers : ℕ) 
  (h1 : total_glasses = 54) 
  (h2 : num_pitchers = 9) : 
  total_glasses / num_pitchers = 6 := by
sorry

end glasses_per_pitcher_l2902_290227


namespace puzzle_solution_l2902_290265

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_four_digit_number (n : ℕ) : Prop := n ≥ 1000 ∧ n ≤ 9999

def distinct_digits (a b c d : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def construct_number (a b c d : ℕ) : ℕ := a * 1000 + b * 100 + c * 10 + d

theorem puzzle_solution :
  ∀ t h e a b g m,
    distinct_digits t h e a ∧
    distinct_digits b e t a ∧
    distinct_digits g a m m ∧
    is_four_digit_number (construct_number t h e a) ∧
    is_four_digit_number (construct_number b e t a) ∧
    is_four_digit_number (construct_number g a m m) ∧
    construct_number t h e a + construct_number b e t a = construct_number g a m m →
    t = 4 ∧ h = 9 ∧ e = 4 ∧ a = 0 ∧ b = 5 ∧ g = 1 ∧ m = 8 :=
by sorry

end puzzle_solution_l2902_290265


namespace slices_left_over_l2902_290285

/-- The number of initial pizza slices -/
def initial_slices : ℕ := 34

/-- The number of slices eaten by Dean -/
def dean_slices : ℕ := 7

/-- The number of slices eaten by Frank -/
def frank_slices : ℕ := 3

/-- The number of slices eaten by Sammy -/
def sammy_slices : ℕ := 4

/-- The number of slices eaten by Nancy -/
def nancy_slices : ℕ := 3

/-- The number of slices eaten by Olivia -/
def olivia_slices : ℕ := 3

/-- The total number of slices eaten -/
def total_eaten : ℕ := dean_slices + frank_slices + sammy_slices + nancy_slices + olivia_slices

/-- Theorem: The number of pizza slices left over is 14 -/
theorem slices_left_over : initial_slices - total_eaten = 14 := by
  sorry

end slices_left_over_l2902_290285


namespace third_to_second_package_ratio_is_half_l2902_290282

/-- Represents the delivery driver's work for a day -/
structure DeliveryDay where
  miles_first_package : ℕ
  miles_second_package : ℕ
  total_pay : ℕ
  pay_per_mile : ℕ

/-- Calculates the ratio of the distance for the third package to the second package -/
def third_to_second_package_ratio (day : DeliveryDay) : ℚ :=
  let total_miles := day.total_pay / day.pay_per_mile
  let miles_third_package := total_miles - day.miles_first_package - day.miles_second_package
  miles_third_package / day.miles_second_package

/-- Theorem stating the ratio of the third package distance to the second package distance -/
theorem third_to_second_package_ratio_is_half (day : DeliveryDay) 
    (h1 : day.miles_first_package = 10)
    (h2 : day.miles_second_package = 28)
    (h3 : day.total_pay = 104)
    (h4 : day.pay_per_mile = 2) :
    third_to_second_package_ratio day = 1/2 := by
  sorry

#eval third_to_second_package_ratio { 
  miles_first_package := 10, 
  miles_second_package := 28, 
  total_pay := 104, 
  pay_per_mile := 2 
}

end third_to_second_package_ratio_is_half_l2902_290282


namespace students_wearing_other_colors_l2902_290201

theorem students_wearing_other_colors 
  (total_students : ℕ) 
  (blue_percent : ℚ) 
  (red_percent : ℚ) 
  (green_percent : ℚ) 
  (h1 : total_students = 700)
  (h2 : blue_percent = 45/100)
  (h3 : red_percent = 23/100)
  (h4 : green_percent = 15/100) :
  ⌊(1 - (blue_percent + red_percent + green_percent)) * total_students⌋ = 119 := by
sorry

end students_wearing_other_colors_l2902_290201


namespace crown_composition_l2902_290232

theorem crown_composition (total_weight : ℝ) (gold copper tin iron : ℝ)
  (h1 : total_weight = 60)
  (h2 : gold + copper + tin + iron = total_weight)
  (h3 : gold + copper = 2/3 * total_weight)
  (h4 : gold + tin = 3/4 * total_weight)
  (h5 : gold + iron = 3/5 * total_weight) :
  gold = 30.5 ∧ copper = 9.5 ∧ tin = 14.5 ∧ iron = 5.5 := by
  sorry

end crown_composition_l2902_290232


namespace quadratic_equation_one_l2902_290294

theorem quadratic_equation_one (x : ℝ) : (x - 2)^2 = 9 ↔ x = 5 ∨ x = -1 := by sorry

end quadratic_equation_one_l2902_290294


namespace thabo_hardcover_nonfiction_count_l2902_290231

/-- Represents the number of books Thabo owns of each type -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def is_valid_collection (books : BookCollection) : Prop :=
  books.hardcover_nonfiction + books.paperback_nonfiction + books.paperback_fiction = 220 ∧
  books.paperback_nonfiction = books.hardcover_nonfiction + 20 ∧
  books.paperback_fiction = 2 * books.paperback_nonfiction

theorem thabo_hardcover_nonfiction_count :
  ∀ (books : BookCollection), is_valid_collection books → books.hardcover_nonfiction = 40 := by
  sorry

end thabo_hardcover_nonfiction_count_l2902_290231


namespace next_perfect_square_l2902_290253

theorem next_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ n : ℕ, n > x ∧ (∃ m : ℕ, n = m^2) ∧ 
  ∀ y : ℕ, y > x → (∃ l : ℕ, y = l^2) → y ≥ n :=
by
  sorry

end next_perfect_square_l2902_290253


namespace original_kittens_correct_l2902_290223

/-- The number of kittens Tim's cat originally had -/
def original_kittens : ℕ := 6

/-- The number of kittens Tim gave away -/
def kittens_given_away : ℕ := 3

/-- The number of kittens Tim received -/
def kittens_received : ℕ := 9

/-- The number of kittens Tim has now -/
def current_kittens : ℕ := 12

/-- Theorem stating that the original number of kittens is correct -/
theorem original_kittens_correct : 
  original_kittens + kittens_received - kittens_given_away = current_kittens := by
  sorry

end original_kittens_correct_l2902_290223


namespace simplify_and_rationalize_l2902_290202

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end simplify_and_rationalize_l2902_290202


namespace complex_fraction_evaluation_l2902_290251

theorem complex_fraction_evaluation (c d : ℂ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : c^2 + c*d + d^2 = 0) : 
  (c^12 + d^12) / (c + d)^12 = -2 := by
  sorry

end complex_fraction_evaluation_l2902_290251


namespace adult_tickets_correct_l2902_290208

/-- The number of adult tickets sold at the Rotary Club's Omelet Breakfast --/
def adult_tickets : ℕ :=
  let small_children_tickets : ℕ := 53
  let older_children_tickets : ℕ := 35
  let senior_tickets : ℕ := 37
  let small_children_omelet : ℚ := 1/2
  let older_children_omelet : ℕ := 1
  let adult_omelet : ℕ := 2
  let senior_omelet : ℚ := 3/2
  let extra_omelets : ℕ := 25
  let total_eggs : ℕ := 584
  let eggs_per_omelet : ℕ := 3
  26

theorem adult_tickets_correct : adult_tickets = 26 := by
  sorry

end adult_tickets_correct_l2902_290208


namespace square_fold_distance_l2902_290284

/-- Given a square ABCD with side length 4, folded along diagonal BD to form a dihedral angle of 60°,
    the distance between the midpoint of BC and point A is 2√2. -/
theorem square_fold_distance (A B C D : ℝ × ℝ) : 
  let side_length : ℝ := 4
  let dihedral_angle : ℝ := 60
  let is_square := (A.1 = 0 ∧ A.2 = 0) ∧ 
                   (B.1 = side_length ∧ B.2 = 0) ∧ 
                   (C.1 = side_length ∧ C.2 = side_length) ∧ 
                   (D.1 = 0 ∧ D.2 = side_length)
  let midpoint_BC := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let distance := Real.sqrt ((A.1 - midpoint_BC.1)^2 + (A.2 - midpoint_BC.2)^2)
  is_square → distance = 2 * Real.sqrt 2 := by
sorry

end square_fold_distance_l2902_290284


namespace modulus_of_5_minus_12i_l2902_290224

theorem modulus_of_5_minus_12i : Complex.abs (5 - 12*I) = 13 := by
  sorry

end modulus_of_5_minus_12i_l2902_290224


namespace new_person_weight_l2902_290238

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 4.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 101 :=
by sorry

end new_person_weight_l2902_290238


namespace calculator_squaring_l2902_290235

theorem calculator_squaring (initial : ℕ) (target : ℕ) : 
  (initial = 3 ∧ target = 2000) → 
  (∃ n : ℕ, initial^(2^n) > target ∧ ∀ m : ℕ, m < n → initial^(2^m) ≤ target) → 
  (∃ n : ℕ, n = 3 ∧ initial^(2^n) > target ∧ ∀ m : ℕ, m < n → initial^(2^m) ≤ target) :=
by sorry

end calculator_squaring_l2902_290235


namespace all_children_receive_candy_l2902_290295

/-- The function that determines which child receives a candy on each turn -/
def candy_distribution (n : ℕ) (x : ℕ) : ℕ :=
  (x * (x + 1) / 2) % n

/-- Proposition: All children receive candy iff the number of children is a power of 2 -/
theorem all_children_receive_candy (n : ℕ) :
  (∀ k : ℕ, k < n → ∃ x : ℕ, candy_distribution n x = k) ↔ ∃ a : ℕ, n = 2^a :=
sorry

end all_children_receive_candy_l2902_290295


namespace mila_visible_area_l2902_290259

/-- The area visible to Mila as she walks around a square -/
theorem mila_visible_area (side_length : ℝ) (visibility_radius : ℝ) : 
  side_length = 4 →
  visibility_radius = 1 →
  (side_length - 2 * visibility_radius)^2 + 
  4 * side_length * visibility_radius + 
  π * visibility_radius^2 = 28 + π := by
  sorry

end mila_visible_area_l2902_290259


namespace carrot_weight_calculation_l2902_290221

/-- The weight of carrots installed by the merchant -/
def carrot_weight : ℝ := sorry

/-- The total weight of all vegetables installed -/
def total_weight : ℝ := 36

/-- The weight of zucchini installed -/
def zucchini_weight : ℝ := 13

/-- The weight of broccoli installed -/
def broccoli_weight : ℝ := 8

/-- The weight of vegetables sold -/
def sold_weight : ℝ := 18

theorem carrot_weight_calculation :
  (carrot_weight + zucchini_weight + broccoli_weight = total_weight) ∧
  (total_weight = 2 * sold_weight) →
  carrot_weight = 15 := by sorry

end carrot_weight_calculation_l2902_290221


namespace triangle_abc_properties_l2902_290230

/-- Theorem about a specific triangle ABC --/
theorem triangle_abc_properties :
  ∀ (a b c : ℝ) (A B C : ℝ),
  A = π / 3 →  -- 60° in radians
  b = 1 →
  c = 4 →
  (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) →  -- Cosine rule
  (a = Real.sqrt 13 ∧ 
   (1 / 2) * b * c * Real.sin A = Real.sqrt 3) :=
by sorry

end triangle_abc_properties_l2902_290230


namespace line_through_points_and_equal_intercepts_l2902_290280

-- Define points
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-2, 0)
def P : ℝ × ℝ := (-1, 3)

-- Define line equations
def line_eq_1 (x y : ℝ) : Prop := 2 * x - 5 * y + 4 = 0
def line_eq_2 (x y : ℝ) : Prop := x + y = 2

-- Define a function to check if a point lies on a line
def point_on_line (p : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  line p.1 p.2

-- Define equal intercepts
def equal_intercepts (line : ℝ → ℝ → Prop) : Prop :=
  ∃ m : ℝ, line m 0 ∧ line 0 m

theorem line_through_points_and_equal_intercepts :
  (point_on_line A line_eq_1 ∧ point_on_line B line_eq_1) ∧
  (point_on_line P line_eq_2 ∧ equal_intercepts line_eq_2) :=
sorry

end line_through_points_and_equal_intercepts_l2902_290280


namespace remainder_proof_l2902_290255

theorem remainder_proof (a b : ℕ) (h : a > b) : 
  220070 % (a + b) = 220070 - (a + b) * (2 * (a - b)) :=
by
  sorry

end remainder_proof_l2902_290255


namespace quadratic_form_sum_l2902_290211

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 4 * x^2 - 8 * x + 3 = a * (x - h)^2 + k) → a + h + k = 4 := by
  sorry

end quadratic_form_sum_l2902_290211


namespace intersection_of_M_and_N_l2902_290246

def M : Set Int := {-1, 0, 1, 3, 5}
def N : Set Int := {-2, 1, 2, 3, 5}

theorem intersection_of_M_and_N : M ∩ N = {1, 3, 5} := by
  sorry

end intersection_of_M_and_N_l2902_290246


namespace negation_of_exists_positive_power_l2902_290245

theorem negation_of_exists_positive_power (x : ℝ) : 
  (¬ (∃ x < 0, 2^x > 0)) ↔ (∀ x < 0, 2^x ≤ 0) := by sorry

end negation_of_exists_positive_power_l2902_290245


namespace max_cross_section_area_l2902_290203

/-- Represents a rectangular prism in 3D space -/
structure RectangularPrism where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the area of the cross-section when a plane intersects a rectangular prism -/
def crossSectionArea (prism : RectangularPrism) (plane : Plane) : ℝ :=
  sorry

/-- The maximum area of the cross-sectional cut theorem -/
theorem max_cross_section_area :
  ∀ (prism : RectangularPrism) (plane : Plane),
    prism.width = 8 →
    prism.length = 12 →
    plane.a = 3 →
    plane.b = 5 →
    plane.c = -2 →
    plane.d = 30 →
    crossSectionArea prism plane = (1 / 2) * Real.sqrt 56016 :=
by
  sorry

end max_cross_section_area_l2902_290203


namespace arithmetic_is_F_sequence_l2902_290264

def is_F_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n ≥ 3, ∃ i j : ℕ, i ≠ j ∧ i < n ∧ j < n ∧ a n = a i + a j

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a n = 2 * n

theorem arithmetic_is_F_sequence :
  ∀ a : ℕ → ℝ, arithmetic_sequence a → is_F_sequence a :=
by sorry

end arithmetic_is_F_sequence_l2902_290264


namespace grid_transform_iff_even_l2902_290218

/-- Represents a grid operation that changes adjacent entries' signs -/
def GridOperation (n : ℕ) := Fin n → Fin n → Unit

/-- Represents the state of the grid -/
def GridState (n : ℕ) := Fin n → Fin n → Int

/-- Initial grid state with all entries 1 -/
def initialGrid (n : ℕ) : GridState n :=
  λ _ _ => 1

/-- Final grid state with all entries -1 -/
def finalGrid (n : ℕ) : GridState n :=
  λ _ _ => -1

/-- Predicate to check if a sequence of operations can transform the grid -/
def canTransform (n : ℕ) : Prop :=
  ∃ (seq : List (GridOperation n)), 
    ∃ (result : GridState n), 
      result = finalGrid n

/-- Main theorem: Grid can be transformed iff n is even -/
theorem grid_transform_iff_even (n : ℕ) (h : n ≥ 2) : 
  canTransform n ↔ Even n :=
sorry

end grid_transform_iff_even_l2902_290218


namespace function_domain_condition_l2902_290222

/-- Given a function f(x) = √(kx² - 4x + 3), prove that for f to have a domain of ℝ, 
    k must be in the range [4/3, +∞). -/
theorem function_domain_condition (k : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (k * x^2 - 4 * x + 3)) ↔ k ≥ 4/3 :=
by sorry

end function_domain_condition_l2902_290222


namespace james_puzzles_l2902_290286

/-- Calculates the number of puzzles James bought given the puzzle size, completion rate, and total time --/
theorem james_puzzles (puzzle_size : ℕ) (pieces_per_interval : ℕ) (interval_minutes : ℕ) (total_minutes : ℕ) :
  puzzle_size = 2000 →
  pieces_per_interval = 100 →
  interval_minutes = 10 →
  total_minutes = 400 →
  (total_minutes / interval_minutes) * pieces_per_interval / puzzle_size = 2 :=
by
  sorry

end james_puzzles_l2902_290286


namespace cos_alpha_plus_pi_third_l2902_290288

theorem cos_alpha_plus_pi_third (α : ℝ) (h : Real.sin (α - π/6) = 1/3) :
  Real.cos (α + π/3) = -1/3 := by
  sorry

end cos_alpha_plus_pi_third_l2902_290288


namespace ten_thousand_one_divides_eight_digit_repeated_l2902_290216

/-- Represents an 8-digit positive integer with repeated digits -/
def EightDigitRepeated : Type := 
  {n : ℕ // 10000000 ≤ n ∧ n < 100000000 ∧ ∃ a b c d : ℕ, n = a * 10000000 + b * 1000000 + c * 100000 + d * 10000 + a * 1000 + b * 100 + c * 10 + d}

/-- Theorem stating that 10001 is a factor of any EightDigitRepeated number -/
theorem ten_thousand_one_divides_eight_digit_repeated (z : EightDigitRepeated) : 
  10001 ∣ z.val := by
  sorry

end ten_thousand_one_divides_eight_digit_repeated_l2902_290216


namespace jordans_money_exceeds_alexs_by_12_5_percent_l2902_290274

/-- Proves that Jordan's money value exceeds Alex's by 12.5% given the specified conditions -/
theorem jordans_money_exceeds_alexs_by_12_5_percent 
  (exchange_rate : ℝ) 
  (alex_dollars : ℝ) 
  (jordan_pounds : ℝ) 
  (h1 : exchange_rate = 1.5)
  (h2 : alex_dollars = 600)
  (h3 : jordan_pounds = 450) :
  (jordan_pounds * exchange_rate - alex_dollars) / alex_dollars * 100 = 12.5 := by
  sorry

#check jordans_money_exceeds_alexs_by_12_5_percent

end jordans_money_exceeds_alexs_by_12_5_percent_l2902_290274


namespace unique_common_roots_l2902_290261

/-- Two cubic polynomials with two distinct common roots -/
def has_two_common_roots (p q : ℝ) : Prop :=
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧
    r₁^3 + p*r₁^2 + 8*r₁ + 10 = 0 ∧
    r₁^3 + q*r₁^2 + 17*r₁ + 15 = 0 ∧
    r₂^3 + p*r₂^2 + 8*r₂ + 10 = 0 ∧
    r₂^3 + q*r₂^2 + 17*r₂ + 15 = 0

/-- The unique solution for p and q -/
theorem unique_common_roots :
  ∃! (p q : ℝ), has_two_common_roots p q ∧ p = 19 ∧ q = 28 := by
  sorry

end unique_common_roots_l2902_290261


namespace yanna_kept_36_apples_l2902_290233

/-- The number of apples Yanna bought -/
def total_apples : ℕ := 60

/-- The number of apples Yanna gave to Zenny -/
def apples_to_zenny : ℕ := 18

/-- The number of apples Yanna gave to Andrea -/
def apples_to_andrea : ℕ := 6

/-- The number of apples Yanna kept -/
def apples_kept : ℕ := total_apples - (apples_to_zenny + apples_to_andrea)

theorem yanna_kept_36_apples : apples_kept = 36 := by
  sorry

end yanna_kept_36_apples_l2902_290233


namespace additional_boys_on_slide_l2902_290206

theorem additional_boys_on_slide (initial_boys total_boys : ℕ) 
  (h1 : initial_boys = 22)
  (h2 : total_boys = 35) :
  total_boys - initial_boys = 13 := by
  sorry

end additional_boys_on_slide_l2902_290206


namespace henry_twice_jill_age_l2902_290296

/-- Represents the number of years ago when Henry was twice Jill's age. -/
def years_ago : ℕ := 9

/-- Henry's present age -/
def henry_age : ℕ := 29

/-- Jill's present age -/
def jill_age : ℕ := 19

theorem henry_twice_jill_age :
  (henry_age + jill_age = 48) →
  (henry_age - years_ago = 2 * (jill_age - years_ago)) := by
  sorry

end henry_twice_jill_age_l2902_290296


namespace ab_pos_necessary_not_sufficient_l2902_290262

theorem ab_pos_necessary_not_sufficient (a b : ℝ) :
  (∃ a b : ℝ, (b / a + a / b > 2) ∧ (a * b > 0)) ∧
  (∃ a b : ℝ, (a * b > 0) ∧ ¬(b / a + a / b > 2)) ∧
  (∀ a b : ℝ, (b / a + a / b > 2) → (a * b > 0)) :=
by sorry

end ab_pos_necessary_not_sufficient_l2902_290262


namespace inequality_proof_l2902_290243

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : (a + b) * (b + c) * (c + a) = 1) : 
  (a^2 / (1 + Real.sqrt (b * c))) + (b^2 / (1 + Real.sqrt (c * a))) + (c^2 / (1 + Real.sqrt (a * b))) ≥ 1/2 := by
  sorry

end inequality_proof_l2902_290243


namespace min_distance_to_line_l2902_290289

/-- The minimum value of (a+1)^2 + b^2 for a point (a, b) on the line y = √3x - √3 is 3 -/
theorem min_distance_to_line : 
  ∀ a b : ℝ, 
  b = Real.sqrt 3 * a - Real.sqrt 3 → 
  (∀ x y : ℝ, y = Real.sqrt 3 * x - Real.sqrt 3 → (a + 1)^2 + b^2 ≤ (x + 1)^2 + y^2) → 
  (a + 1)^2 + b^2 = 3 :=
by sorry

end min_distance_to_line_l2902_290289


namespace johns_dad_age_l2902_290283

theorem johns_dad_age (j d : ℕ) : j + 28 = d → j + d = 76 → d = 52 := by sorry

end johns_dad_age_l2902_290283


namespace hotel_rooms_for_couples_l2902_290292

theorem hotel_rooms_for_couples :
  let single_rooms : ℕ := 14
  let bubble_bath_per_bath : ℕ := 10
  let total_bubble_bath : ℕ := 400
  let baths_per_single_room : ℕ := 1
  let baths_per_couple_room : ℕ := 2
  ∃ couple_rooms : ℕ,
    couple_rooms = 13 ∧
    total_bubble_bath = bubble_bath_per_bath * (single_rooms * baths_per_single_room + couple_rooms * baths_per_couple_room) :=
by
  sorry

end hotel_rooms_for_couples_l2902_290292


namespace addition_inequality_l2902_290299

theorem addition_inequality (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a + c > b + d := by
  sorry

end addition_inequality_l2902_290299


namespace sin_240_degrees_l2902_290260

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_degrees_l2902_290260


namespace probability_blue_after_removal_l2902_290244

/-- Probability of pulling a blue ball after removal -/
theorem probability_blue_after_removal (initial_total : ℕ) (initial_blue : ℕ) (removed_blue : ℕ) :
  initial_total = 18 →
  initial_blue = 6 →
  removed_blue = 3 →
  (initial_blue - removed_blue : ℚ) / (initial_total - removed_blue) = 1 / 5 := by
  sorry


end probability_blue_after_removal_l2902_290244


namespace system_two_solutions_l2902_290267

/-- The system of inequalities has exactly two solutions if and only if a = 7 -/
theorem system_two_solutions (a : ℝ) : 
  (∃! x y z : ℝ, x ≠ y ∧ 
    (abs z + abs (z - x) ≤ a - abs (x - 1)) ∧
    ((z - 4) * (z + 3) ≥ (4 - x) * (3 + x)) ∧
    (abs z + abs (z - y) ≤ a - abs (y - 1)) ∧
    ((z - 4) * (z + 3) ≥ (4 - y) * (3 + y)))
  ↔ a = 7 := by sorry

end system_two_solutions_l2902_290267


namespace even_number_of_solutions_l2902_290207

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  (y^2 + 6) * (x - 1) = y * (x^2 + 1) ∧
  (x^2 + 6) * (y - 1) = x * (y^2 + 1)

/-- The set of solutions to the system -/
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | system p.1 p.2}

/-- The number of solutions is finite -/
axiom finite_solutions : Set.Finite solution_set

/-- Theorem: The system has an even number of real solutions -/
theorem even_number_of_solutions : ∃ n : ℕ, n % 2 = 0 ∧ Set.ncard solution_set = n := by
  sorry

end even_number_of_solutions_l2902_290207


namespace emily_coloring_books_l2902_290252

/-- 
Given Emily's initial number of coloring books, the number she gave away,
and her current total, prove that she bought 14 coloring books.
-/
theorem emily_coloring_books 
  (initial : ℕ) 
  (given_away : ℕ) 
  (current_total : ℕ) 
  (h1 : initial = 7)
  (h2 : given_away = 2)
  (h3 : current_total = 19) :
  current_total - (initial - given_away) = 14 := by
  sorry

end emily_coloring_books_l2902_290252


namespace farm_animals_count_l2902_290263

theorem farm_animals_count : 
  ∀ (total_legs ducks dogs : ℕ),
  total_legs = 24 →
  ducks = 4 →
  total_legs = 2 * ducks + 4 * dogs →
  ducks + dogs = 8 :=
by
  sorry

end farm_animals_count_l2902_290263


namespace max_x_minus_y_l2902_290287

theorem max_x_minus_y (x y : ℝ) (h : 2 * (x^2 + y^2) = x + y + 2*x*y) : 
  ∃ (M : ℝ), M = 2 ∧ ∀ (a b : ℝ), 2 * (a^2 + b^2) = a + b + 2*a*b → a - b ≤ M :=
sorry

end max_x_minus_y_l2902_290287


namespace pentagon_largest_angle_l2902_290281

theorem pentagon_largest_angle (P Q R S T : ℝ) : 
  P = 75 →
  Q = 110 →
  R = S →
  T = 3 * R - 20 →
  P + Q + R + S + T = 540 →
  max P (max Q (max R (max S T))) = 217 :=
sorry

end pentagon_largest_angle_l2902_290281


namespace least_addition_for_divisibility_l2902_290209

theorem least_addition_for_divisibility : ∃! x : ℕ, x < 37 ∧ (1052 + x) % 37 = 0 ∧ ∀ y : ℕ, y < x → (1052 + y) % 37 ≠ 0 :=
by
  -- The proof goes here
  sorry

end least_addition_for_divisibility_l2902_290209


namespace age_difference_l2902_290240

def sachin_age : ℕ := 49

theorem age_difference (rahul_age : ℕ) 
  (h1 : sachin_age < rahul_age)
  (h2 : sachin_age * 9 = rahul_age * 7) : 
  rahul_age - sachin_age = 14 := by
sorry

end age_difference_l2902_290240


namespace f_composition_value_l2902_290248

def f (x : ℝ) : ℝ := 4 * x^3 - 6 * x + 2

theorem f_composition_value : f (f 2) = 42462 := by
  sorry

end f_composition_value_l2902_290248


namespace stating_race_result_l2902_290250

/-- Represents a runner in the race -/
inductive Runner
| Primus
| Secundus
| Tertius

/-- Represents the order of runners -/
def RunnerOrder := List Runner

/-- The number of place changes between pairs of runners -/
structure PlaceChanges where
  primus_secundus : Nat
  secundus_tertius : Nat
  primus_tertius : Nat

/-- The initial order of runners -/
def initial_order : RunnerOrder := [Runner.Primus, Runner.Secundus, Runner.Tertius]

/-- The place changes during the race -/
def race_changes : PlaceChanges := {
  primus_secundus := 9,
  secundus_tertius := 10,
  primus_tertius := 11
}

/-- The final order of runners -/
def final_order : RunnerOrder := [Runner.Secundus, Runner.Tertius, Runner.Primus]

/-- 
Theorem stating that given the initial order and place changes,
the final order is [Secundus, Tertius, Primus]
-/
theorem race_result (order : RunnerOrder) (changes : PlaceChanges) :
  order = initial_order ∧ changes = race_changes →
  final_order = [Runner.Secundus, Runner.Tertius, Runner.Primus] :=
by sorry

end stating_race_result_l2902_290250


namespace fish_tank_problem_l2902_290247

/-- Represents the number of gallons needed for each of the smaller tanks -/
def smaller_tank_gallons (total_weekly_gallons : ℕ) : ℕ :=
  (total_weekly_gallons - 2 * 8) / 2

/-- Represents the difference in gallons between larger and smaller tanks -/
def gallon_difference (total_weekly_gallons : ℕ) : ℕ :=
  8 - smaller_tank_gallons total_weekly_gallons

theorem fish_tank_problem (total_gallons : ℕ) 
  (h1 : total_gallons = 112) 
  (h2 : total_gallons % 4 = 0) : 
  gallon_difference (total_gallons / 4) = 2 := by
  sorry

end fish_tank_problem_l2902_290247


namespace M_intersect_N_equals_zero_l2902_290200

def M : Set ℝ := {x : ℝ | |x| ≤ 2}
def N : Set ℝ := {x : ℝ | x^2 - 3*x = 0}

theorem M_intersect_N_equals_zero : M ∩ N = {0} := by
  sorry

end M_intersect_N_equals_zero_l2902_290200


namespace number_of_students_l2902_290290

/-- The number of storybooks available for distribution -/
def total_books : ℕ := 60

/-- Predicate to check if there are books left over after initial distribution -/
def has_leftover (n : ℕ) : Prop := n < total_books

/-- Predicate to check if remaining books can be evenly distributed with 2 students sharing 1 book -/
def can_evenly_distribute_remainder (n : ℕ) : Prop :=
  ∃ k : ℕ, total_books - n = 2 * k

/-- The theorem stating the number of students in the class -/
theorem number_of_students :
  ∃ n : ℕ, n = 40 ∧ 
    has_leftover n ∧ 
    can_evenly_distribute_remainder n :=
sorry

end number_of_students_l2902_290290


namespace odd_expression_l2902_290266

theorem odd_expression (p q : ℕ) (hp : Odd p) (hq : Odd q) (hp_pos : p > 0) (hq_pos : q > 0) :
  Odd (2 * p^2 - q) := by
  sorry

end odd_expression_l2902_290266


namespace expression_evaluation_l2902_290220

theorem expression_evaluation (x : ℝ) : x * (x * (x * (x - 3) - 5) + 12) + 2 = x^4 - 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end expression_evaluation_l2902_290220


namespace geometric_sequence_common_ratio_l2902_290214

theorem geometric_sequence_common_ratio 
  (a b : ℝ) 
  (h1 : 2 * a = 1 + b) 
  (h2 : (a + 2)^2 = 3 * (b + 5)) 
  (h3 : a + 2 ≠ 0) 
  (h4 : b + 5 ≠ 0) : 
  (a + 2) / 3 = 2 :=
sorry

end geometric_sequence_common_ratio_l2902_290214


namespace tan_y_plus_pi_third_l2902_290297

theorem tan_y_plus_pi_third (y : ℝ) (h : Real.tan y = -1) : 
  Real.tan (y + π/3) = -1 := by
  sorry

end tan_y_plus_pi_third_l2902_290297
