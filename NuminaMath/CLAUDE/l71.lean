import Mathlib

namespace NUMINAMATH_CALUDE_sandy_correct_sums_l71_7167

theorem sandy_correct_sums : ℕ :=
  let total_sums : ℕ := 40
  let marks_per_correct : ℕ := 4
  let marks_lost_per_incorrect : ℕ := 3
  let total_marks : ℕ := 72
  let correct_sums : ℕ := 27
  let incorrect_sums : ℕ := total_sums - correct_sums

  have h1 : correct_sums + incorrect_sums = total_sums := by sorry
  have h2 : marks_per_correct * correct_sums - marks_lost_per_incorrect * incorrect_sums = total_marks := by sorry

  correct_sums

-- The proof is omitted

end NUMINAMATH_CALUDE_sandy_correct_sums_l71_7167


namespace NUMINAMATH_CALUDE_saree_price_proof_l71_7195

/-- Proves that given a product with two successive discounts of 10% and 5%, 
    if the final sale price is Rs. 513, then the original price was Rs. 600. -/
theorem saree_price_proof (original_price : ℝ) : 
  (original_price * (1 - 0.1) * (1 - 0.05) = 513) → original_price = 600 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_proof_l71_7195


namespace NUMINAMATH_CALUDE_percentage_not_sold_approx_l71_7150

def initial_stock : ℕ := 1400
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135

theorem percentage_not_sold_approx (ε : ℝ) (ε_pos : ε > 0) :
  ∃ (p : ℝ), abs (p - ((initial_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales)) / initial_stock * 100)) < ε ∧
             abs (p - 71.29) < ε := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_sold_approx_l71_7150


namespace NUMINAMATH_CALUDE_smallest_denominator_fraction_l71_7102

-- Define the fraction type
structure Fraction where
  numerator : ℕ
  denominator : ℕ
  denom_pos : denominator > 0

-- Define the property of being in the open interval
def inOpenInterval (f : Fraction) : Prop :=
  47 / 245 < f.numerator / f.denominator ∧ f.numerator / f.denominator < 34 / 177

-- Define the property of having the smallest denominator
def hasSmallestDenominator (f : Fraction) : Prop :=
  ∀ g : Fraction, inOpenInterval g → f.denominator ≤ g.denominator

-- The main theorem
theorem smallest_denominator_fraction :
  ∃ f : Fraction, f.numerator = 19 ∧ f.denominator = 99 ∧
  inOpenInterval f ∧ hasSmallestDenominator f :=
sorry

end NUMINAMATH_CALUDE_smallest_denominator_fraction_l71_7102


namespace NUMINAMATH_CALUDE_vector_simplification_1_vector_simplification_2_vector_simplification_3_l71_7173

variable {V : Type*} [AddCommGroup V]

-- Define vector between two points
def vec (A B : V) : V := B - A

-- Theorem 1
theorem vector_simplification_1 (A B C D : V) :
  vec A B + vec B C - vec A D = vec D C := by sorry

-- Theorem 2
theorem vector_simplification_2 (A B C D : V) :
  (vec A B - vec C D) - (vec A C - vec B D) = 0 := by sorry

-- Theorem 3
theorem vector_simplification_3 (A B C D O : V) :
  (vec A C + vec B O + vec O A) - (vec D C - vec D O - vec O B) = 0 := by sorry

end NUMINAMATH_CALUDE_vector_simplification_1_vector_simplification_2_vector_simplification_3_l71_7173


namespace NUMINAMATH_CALUDE_total_chicken_pieces_is_74_l71_7128

/-- The number of chicken pieces needed for all orders at Clucks Delux -/
def total_chicken_pieces : ℕ :=
  let chicken_pasta_pieces : ℕ := 2
  let barbecue_chicken_pieces : ℕ := 4
  let fried_chicken_dinner_pieces : ℕ := 8
  let grilled_chicken_salad_pieces : ℕ := 1
  
  let fried_chicken_dinner_orders : ℕ := 4
  let chicken_pasta_orders : ℕ := 8
  let barbecue_chicken_orders : ℕ := 5
  let grilled_chicken_salad_orders : ℕ := 6

  (fried_chicken_dinner_pieces * fried_chicken_dinner_orders) +
  (chicken_pasta_pieces * chicken_pasta_orders) +
  (barbecue_chicken_pieces * barbecue_chicken_orders) +
  (grilled_chicken_salad_pieces * grilled_chicken_salad_orders)

theorem total_chicken_pieces_is_74 : total_chicken_pieces = 74 := by
  sorry

end NUMINAMATH_CALUDE_total_chicken_pieces_is_74_l71_7128


namespace NUMINAMATH_CALUDE_focus_coordinates_for_specific_ellipse_l71_7100

/-- Represents an ellipse with its center and axis endpoints -/
structure Ellipse where
  center : ℝ × ℝ
  major_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ)
  minor_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ)

/-- Calculates the coordinates of the focus with greater x-coordinate for a given ellipse -/
def focus_with_greater_x (e : Ellipse) : ℝ × ℝ :=
  sorry

/-- Theorem stating that for the given ellipse, the focus with greater x-coordinate
    has coordinates (3.5 + √6/2, 0) -/
theorem focus_coordinates_for_specific_ellipse :
  let e : Ellipse := {
    center := (3.5, 0),
    major_axis_endpoints := ((0, 0), (7, 0)),
    minor_axis_endpoints := ((3.5, 2.5), (3.5, -2.5))
  }
  focus_with_greater_x e = (3.5 + Real.sqrt 6 / 2, 0) := by
  sorry

end NUMINAMATH_CALUDE_focus_coordinates_for_specific_ellipse_l71_7100


namespace NUMINAMATH_CALUDE_prob_at_least_one_3_or_5_correct_l71_7171

/-- The probability of at least one die showing either a 3 or a 5 when rolling two fair 6-sided dice -/
def prob_at_least_one_3_or_5 : ℚ :=
  5 / 9

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The set of outcomes for a single die roll -/
def die_outcomes : Finset ℕ := Finset.range num_sides

/-- The set of favorable outcomes for a single die (3 or 5) -/
def favorable_single : Finset ℕ := {3, 5}

/-- The sample space for rolling two dice -/
def sample_space : Finset (ℕ × ℕ) := die_outcomes.product die_outcomes

/-- The event where at least one die shows 3 or 5 -/
def event : Finset (ℕ × ℕ) :=
  sample_space.filter (fun p => p.1 ∈ favorable_single ∨ p.2 ∈ favorable_single)

theorem prob_at_least_one_3_or_5_correct :
  (event.card : ℚ) / sample_space.card = prob_at_least_one_3_or_5 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_3_or_5_correct_l71_7171


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l71_7120

theorem arithmetic_sequence_problem (n : ℕ) (min max sum : ℚ) (h_n : n = 150) 
  (h_min : min = 20) (h_max : max = 90) (h_sum : sum = 9000) :
  let avg := sum / n
  let d := (max - min) / (2 * (n - 1))
  let L := avg - (29 * d)
  let G := avg + (29 * d)
  G - L = 7140 / 149 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l71_7120


namespace NUMINAMATH_CALUDE_inequality_solution_l71_7199

theorem inequality_solution (x : ℝ) : 
  1 / (x^2 + 1) > 3 / x + 17 / 10 ↔ -2 < x ∧ x < 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l71_7199


namespace NUMINAMATH_CALUDE_triangle_area_l71_7191

/-- Given a triangle with perimeter 36, inradius 2.5, and sides in ratio 3:4:5, its area is 45 -/
theorem triangle_area (a b c : ℝ) (perimeter inradius : ℝ) : 
  perimeter = 36 →
  inradius = 2.5 →
  ∃ (k : ℝ), a = 3*k ∧ b = 4*k ∧ c = 5*k →
  a + b + c = perimeter →
  (a + b + c) / 2 * inradius = 45 := by
sorry


end NUMINAMATH_CALUDE_triangle_area_l71_7191


namespace NUMINAMATH_CALUDE_product_of_fractions_squared_l71_7147

theorem product_of_fractions_squared :
  (8 / 9) ^ 2 * (1 / 3) ^ 2 * (1 / 4) ^ 2 = 4 / 729 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_squared_l71_7147


namespace NUMINAMATH_CALUDE_correct_mean_after_error_fix_l71_7186

/-- Given a set of values with an incorrect mean due to a misrecorded value,
    calculate the correct mean after fixing the error. -/
theorem correct_mean_after_error_fix (n : ℕ) (incorrect_mean : ℚ) (wrong_value correct_value : ℚ) 
    (h1 : n = 30)
    (h2 : incorrect_mean = 140)
    (h3 : wrong_value = 135)
    (h4 : correct_value = 145) :
    let total_sum := n * incorrect_mean
    let difference := correct_value - wrong_value
    let corrected_sum := total_sum + difference
    corrected_sum / n = 140333 / 1000 := by
  sorry

#eval (140333 : ℚ) / 1000  -- To verify the result is indeed 140.333

end NUMINAMATH_CALUDE_correct_mean_after_error_fix_l71_7186


namespace NUMINAMATH_CALUDE_square_root_of_nine_l71_7117

-- Define the square root function
def square_root (x : ℝ) : Set ℝ := {y : ℝ | y * y = x}

-- State the theorem
theorem square_root_of_nine : square_root 9 = {3, -3} := by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l71_7117


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l71_7153

theorem solve_system_of_equations (b : ℝ) : 
  (∃ x : ℝ, 2 * x + 7 = 3 ∧ b * x - 10 = -2) → b = -4 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l71_7153


namespace NUMINAMATH_CALUDE_remaining_truck_capacity_l71_7168

theorem remaining_truck_capacity
  (max_load : ℕ)
  (bag_mass : ℕ)
  (num_bags : ℕ)
  (h1 : max_load = 900)
  (h2 : bag_mass = 8)
  (h3 : num_bags = 100) :
  max_load - (bag_mass * num_bags) = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_truck_capacity_l71_7168


namespace NUMINAMATH_CALUDE_smallest_five_digit_mod_9_5_l71_7134

theorem smallest_five_digit_mod_9_5 : ∃ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit integer
  n % 9 = 5 ∧                 -- equivalent to 5 mod 9
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000 ∧ m % 9 = 5) → n ≤ m) ∧ 
  n = 10004 := by
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_mod_9_5_l71_7134


namespace NUMINAMATH_CALUDE_circle_circumference_inscribed_rectangle_l71_7104

theorem circle_circumference_inscribed_rectangle (a b r : ℝ) (h1 : a = 9) (h2 : b = 12) 
  (h3 : r * r = (a * a + b * b) / 4) : 2 * π * r = 15 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_inscribed_rectangle_l71_7104


namespace NUMINAMATH_CALUDE_mean_score_all_students_l71_7141

/-- The mean score of all students given specific class conditions --/
theorem mean_score_all_students
  (morning_mean : ℝ)
  (afternoon_mean : ℝ)
  (class_ratio : ℚ)
  (additional_group_score : ℝ)
  (additional_group_ratio : ℚ)
  (h1 : morning_mean = 85)
  (h2 : afternoon_mean = 72)
  (h3 : class_ratio = 4/5)
  (h4 : additional_group_score = 68)
  (h5 : additional_group_ratio = 1/4) :
  ∃ (total_mean : ℝ), total_mean = 87 ∧
    total_mean = (morning_mean * class_ratio + 
                  afternoon_mean * (1 - additional_group_ratio) +
                  additional_group_score * additional_group_ratio) /
                 (class_ratio + 1) := by
  sorry

end NUMINAMATH_CALUDE_mean_score_all_students_l71_7141


namespace NUMINAMATH_CALUDE_max_water_bottles_proof_l71_7142

/-- Given a total number of water bottles and athletes, with each athlete receiving at least one water bottle,
    calculate the maximum number of water bottles one athlete could have received. -/
def max_water_bottles (total_bottles : ℕ) (total_athletes : ℕ) : ℕ :=
  total_bottles - (total_athletes - 1)

/-- Prove that given 40 water bottles distributed among 25 athletes, with each athlete receiving at least one water bottle,
    the maximum number of water bottles one athlete could have received is 16. -/
theorem max_water_bottles_proof :
  max_water_bottles 40 25 = 16 := by
  sorry

#eval max_water_bottles 40 25

end NUMINAMATH_CALUDE_max_water_bottles_proof_l71_7142


namespace NUMINAMATH_CALUDE_distance_walked_l71_7174

theorem distance_walked (x t d : ℝ) 
  (h1 : d = x * t) 
  (h2 : d = (x + 1/2) * (4/5 * t))
  (h3 : d = (x - 1/2) * (t + 5/2)) :
  d = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_walked_l71_7174


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l71_7182

/-- For all a > 0 and a ≠ 1, the function f(x) = a^(x-1) + 4 passes through (1, 5) -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 4
  f 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l71_7182


namespace NUMINAMATH_CALUDE_triangle_side_length_theorem_l71_7162

def triangle_side_length (a : ℝ) : Set ℝ :=
  if a < Real.sqrt 3 / 2 then
    ∅
  else if a = Real.sqrt 3 / 2 then
    {1/2}
  else if a < 1 then
    {(1 + Real.sqrt (4 * a^2 - 3)) / 2, (1 - Real.sqrt (4 * a^2 - 3)) / 2}
  else
    {(1 + Real.sqrt (4 * a^2 - 3)) / 2}

theorem triangle_side_length_theorem (a : ℝ) :
  let A : ℝ := 60 * π / 180
  let AB : ℝ := 1
  let BC : ℝ := a
  ∀ AC ∈ triangle_side_length a,
    AC^2 = AB^2 + BC^2 - 2 * AB * BC * Real.cos A :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_theorem_l71_7162


namespace NUMINAMATH_CALUDE_james_weekly_earnings_l71_7105

/-- Calculates the weekly earnings from car rental -/
def weekly_earnings (rate : ℝ) (hours_per_day : ℝ) (days_per_week : ℝ) : ℝ :=
  rate * hours_per_day * days_per_week

/-- Proof that James' weekly earnings from car rental are $640 -/
theorem james_weekly_earnings :
  weekly_earnings 20 8 4 = 640 := by
  sorry

end NUMINAMATH_CALUDE_james_weekly_earnings_l71_7105


namespace NUMINAMATH_CALUDE_inequality_proof_l71_7188

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a + b + c + 3 * a * b * c ≥ (a * b)^2 + (b * c)^2 + (c * a)^2 + 3) :
  (a^3 + b^3 + c^3) / 3 ≥ (a * b * c + 2021) / 2022 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l71_7188


namespace NUMINAMATH_CALUDE_eating_competition_time_l71_7183

/-- Represents the number of minutes it takes to eat everything -/
def total_time : ℕ := 48

/-- Represents the number of jars of honey eaten by Carlson -/
def carlson_honey : ℕ := 8

/-- Represents the number of jars of jam eaten by Carlson -/
def carlson_jam : ℕ := 4

/-- The time it takes Carlson to eat a jar of jam -/
def carlson_jam_time : ℕ := 2

/-- The time it takes Winnie the Pooh to eat a jar of jam -/
def pooh_jam_time : ℕ := 7

/-- The time it takes Winnie the Pooh to eat a pot of honey -/
def pooh_honey_time : ℕ := 3

/-- The time it takes Carlson to eat a pot of honey -/
def carlson_honey_time : ℕ := 5

/-- The total number of jars of jam and pots of honey -/
def total_jars : ℕ := 10

theorem eating_competition_time :
  carlson_honey * carlson_honey_time + carlson_jam * carlson_jam_time = total_time ∧
  (total_jars - carlson_honey) * pooh_honey_time + (total_jars - carlson_jam) * pooh_jam_time = total_time ∧
  carlson_honey + carlson_jam ≤ total_jars :=
by sorry

end NUMINAMATH_CALUDE_eating_competition_time_l71_7183


namespace NUMINAMATH_CALUDE_car_travel_time_l71_7190

/-- Proves that a car traveling 715 kilometers at an average speed of 65.0 km/h takes 11 hours -/
theorem car_travel_time (distance : ℝ) (speed : ℝ) (time : ℝ) :
  distance = 715 →
  speed = 65 →
  time = distance / speed →
  time = 11 :=
by sorry

end NUMINAMATH_CALUDE_car_travel_time_l71_7190


namespace NUMINAMATH_CALUDE_max_segment_length_squared_l71_7170

/-- Circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- Line defined by two points -/
structure Line where
  P : ℝ × ℝ
  Q : ℝ × ℝ

/-- Point on a circle -/
def PointOnCircle (ω : Circle) (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  let (cx, cy) := ω.O
  (x - cx)^2 + (y - cy)^2 = ω.r^2

/-- Tangent line to a circle at a point -/
def TangentLine (ω : Circle) (T : ℝ × ℝ) (l : Line) : Prop :=
  PointOnCircle ω T ∧ 
  ∃ (P : ℝ × ℝ), P ≠ T ∧ PointOnCircle ω P ∧ 
    ((P.1 - T.1) * (l.Q.1 - l.P.1) + (P.2 - T.2) * (l.Q.2 - l.P.2) = 0)

/-- Perpendicular foot from a point to a line -/
def PerpendicularFoot (A : ℝ × ℝ) (l : Line) (P : ℝ × ℝ) : Prop :=
  (P.1 - A.1) * (l.Q.1 - l.P.1) + (P.2 - A.2) * (l.Q.2 - l.P.2) = 0 ∧
  ∃ (t : ℝ), P = (l.P.1 + t * (l.Q.1 - l.P.1), l.P.2 + t * (l.Q.2 - l.P.2))

/-- The main theorem -/
theorem max_segment_length_squared 
  (ω : Circle) 
  (A B C T : ℝ × ℝ) 
  (l : Line) 
  (P : ℝ × ℝ) :
  PointOnCircle ω A ∧ 
  PointOnCircle ω B ∧
  ω.r = 12 ∧
  (A.1 - ω.O.1)^2 + (A.2 - ω.O.2)^2 = (B.1 - ω.O.1)^2 + (B.2 - ω.O.2)^2 ∧
  (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) < 0 ∧
  TangentLine ω T l ∧
  PerpendicularFoot A l P →
  ∃ (m : ℝ), m^2 = 612 ∧ 
    ∀ (X : ℝ × ℝ), PointOnCircle ω X → 
      (X.1 - B.1)^2 + (X.2 - B.2)^2 ≤ m^2 := by
  sorry

end NUMINAMATH_CALUDE_max_segment_length_squared_l71_7170


namespace NUMINAMATH_CALUDE_fraction_equality_l71_7145

theorem fraction_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b)^2 / (a^2 + 2*a*b + b^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l71_7145


namespace NUMINAMATH_CALUDE_monkey_nuts_problem_l71_7193

theorem monkey_nuts_problem (n : ℕ) (x : ℕ) : 
  n > 1 → 
  x > 1 → 
  n * x - n * (n - 1) = 35 → 
  x = 11 :=
by sorry

end NUMINAMATH_CALUDE_monkey_nuts_problem_l71_7193


namespace NUMINAMATH_CALUDE_two_cones_cost_l71_7152

/-- The cost of a single ice cream cone in cents -/
def single_cone_cost : ℕ := 99

/-- The number of ice cream cones -/
def num_cones : ℕ := 2

/-- Theorem: The cost of 2 ice cream cones is 198 cents -/
theorem two_cones_cost : single_cone_cost * num_cones = 198 := by
  sorry

end NUMINAMATH_CALUDE_two_cones_cost_l71_7152


namespace NUMINAMATH_CALUDE_sqrt_450_simplified_l71_7139

theorem sqrt_450_simplified : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplified_l71_7139


namespace NUMINAMATH_CALUDE_min_value_problem_l71_7107

theorem min_value_problem (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 9) 
  (h2 : e * f * g * h = 4) : 
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 ≥ 24 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l71_7107


namespace NUMINAMATH_CALUDE_horner_method_v3_horner_method_correct_l71_7133

def horner_polynomial (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def horner_v3 (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  v2 * x + 79

theorem horner_method_v3 :
  horner_v3 (-4) = -57 :=
by sorry

theorem horner_method_correct :
  horner_v3 (-4) = horner_polynomial (-4) :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_horner_method_correct_l71_7133


namespace NUMINAMATH_CALUDE_namjoon_lowest_height_l71_7114

/-- Heights of planks in centimeters -/
def height_A : ℝ := 2.4
def height_B : ℝ := 3.2
def height_C : ℝ := 2.8

/-- Number of planks each person stands on -/
def num_A : ℕ := 8
def num_B : ℕ := 4
def num_C : ℕ := 5

/-- Total heights for each person -/
def height_Eunji : ℝ := height_A * num_A
def height_Namjoon : ℝ := height_B * num_B
def height_Hoseok : ℝ := height_C * num_C

theorem namjoon_lowest_height :
  height_Namjoon < height_Eunji ∧ height_Namjoon < height_Hoseok :=
by sorry

end NUMINAMATH_CALUDE_namjoon_lowest_height_l71_7114


namespace NUMINAMATH_CALUDE_sqrt_meaningful_condition_l71_7175

theorem sqrt_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 4) ↔ x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_condition_l71_7175


namespace NUMINAMATH_CALUDE_square_remainder_l71_7118

theorem square_remainder (N : ℤ) : 
  (N % 5 = 3) → (N^2 % 5 = 4) := by
sorry

end NUMINAMATH_CALUDE_square_remainder_l71_7118


namespace NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l71_7148

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to distribute 5 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_five_balls_three_boxes : distribute 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l71_7148


namespace NUMINAMATH_CALUDE_walking_equations_correct_l71_7116

/-- Represents the speeds and distances of two people walking --/
structure WalkingScenario where
  distance : ℝ
  catchup_time : ℝ
  meet_time : ℝ
  speed_a : ℝ
  speed_b : ℝ

/-- The system of equations correctly represents the walking scenario --/
def correct_equations (s : WalkingScenario) : Prop :=
  (10 * s.speed_b - 10 * s.speed_a = s.distance) ∧
  (2 * s.speed_a + 2 * s.speed_b = s.distance)

/-- The given scenario satisfies the conditions --/
def satisfies_conditions (s : WalkingScenario) : Prop :=
  s.distance = 50 ∧
  s.catchup_time = 10 ∧
  s.meet_time = 2 ∧
  s.speed_a > 0 ∧
  s.speed_b > 0

theorem walking_equations_correct (s : WalkingScenario) 
  (h : satisfies_conditions s) : correct_equations s := by
  sorry


end NUMINAMATH_CALUDE_walking_equations_correct_l71_7116


namespace NUMINAMATH_CALUDE_expression_value_l71_7169

theorem expression_value (a : ℝ) (h1 : 1 ≤ a) (h2 : a < 2) :
  Real.sqrt (a + 2 * Real.sqrt (a - 1)) + Real.sqrt (a - 2 * Real.sqrt (a - 1)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l71_7169


namespace NUMINAMATH_CALUDE_product_72_difference_equals_sum_l71_7101

theorem product_72_difference_equals_sum (P Q R S : ℕ+) : 
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S →
  P * Q = 72 →
  R * S = 72 →
  P - Q = R + S →
  P = 12 := by
sorry

end NUMINAMATH_CALUDE_product_72_difference_equals_sum_l71_7101


namespace NUMINAMATH_CALUDE_gnomes_distribution_l71_7108

/-- Given a street with houses and gnomes, calculates the number of gnomes in each of the first few houses -/
def gnomes_per_house (total_houses : ℕ) (total_gnomes : ℕ) (last_house_gnomes : ℕ) : ℕ :=
  (total_gnomes - last_house_gnomes) / (total_houses - 1)

/-- Theorem stating that under given conditions, each of the first few houses has 3 gnomes -/
theorem gnomes_distribution (total_houses : ℕ) (total_gnomes : ℕ) (last_house_gnomes : ℕ)
  (h1 : total_houses = 5)
  (h2 : total_gnomes = 20)
  (h3 : last_house_gnomes = 8) :
  gnomes_per_house total_houses total_gnomes last_house_gnomes = 3 := by
  sorry

end NUMINAMATH_CALUDE_gnomes_distribution_l71_7108


namespace NUMINAMATH_CALUDE_solve_candy_store_problem_l71_7146

/-- Represents the candy store problem --/
def candy_store_problem (caramel_price toffee_price chocolate_price : ℕ)
  (initial_quantity : ℕ) (initial_money : ℕ) : Prop :=
  let chocolate_promo := initial_quantity / 3
  let toffee_to_buy := initial_quantity - chocolate_promo
  let caramel_promo := toffee_to_buy / 3
  let caramel_to_buy := initial_quantity - caramel_promo
  let total_cost := chocolate_price * initial_quantity +
                    toffee_price * toffee_to_buy +
                    caramel_price * caramel_to_buy
  initial_money - total_cost = 72

/-- Theorem stating the solution to the candy store problem --/
theorem solve_candy_store_problem :
  candy_store_problem 3 5 10 8 200 :=
sorry


end NUMINAMATH_CALUDE_solve_candy_store_problem_l71_7146


namespace NUMINAMATH_CALUDE_school_report_mistake_l71_7172

theorem school_report_mistake :
  ¬ ∃ (girls : ℕ), 
    let boys := girls + 373
    let total := girls + boys
    total = 3688 :=
by
  sorry

end NUMINAMATH_CALUDE_school_report_mistake_l71_7172


namespace NUMINAMATH_CALUDE_rationalize_cube_root_difference_l71_7198

theorem rationalize_cube_root_difference : ∃ (A B C D : ℕ),
  (((1 : ℝ) / (5^(1/3) - 3^(1/3))) * ((5^(2/3) + 5^(1/3)*3^(1/3) + 3^(2/3)) / (5^(2/3) + 5^(1/3)*3^(1/3) + 3^(2/3)))) = 
  ((A : ℝ)^(1/3) + (B : ℝ)^(1/3) + (C : ℝ)^(1/3)) / (D : ℝ) ∧
  A + B + C + D = 51 := by
sorry

end NUMINAMATH_CALUDE_rationalize_cube_root_difference_l71_7198


namespace NUMINAMATH_CALUDE_fraction_problem_l71_7158

theorem fraction_problem (n : ℤ) : 
  (n : ℚ) / (4 * n - 5 : ℚ) = 3 / 7 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l71_7158


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l71_7154

theorem quadratic_equation_roots (k : ℚ) :
  (∃ x, 5 * x^2 + k * x - 6 = 0 ∧ x = 2) →
  (∃ y, 5 * y^2 + k * y - 6 = 0 ∧ y = -3/5) ∧
  k = -7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l71_7154


namespace NUMINAMATH_CALUDE_count_numbers_with_6_or_7_proof_l71_7194

/-- The number of integers from 1 to 512 (inclusive) in base 8 that contain at least one digit 6 or 7 -/
def count_numbers_with_6_or_7 : ℕ := 296

/-- The total number of integers we're considering -/
def total_numbers : ℕ := 512

/-- The base we're working in -/
def base : ℕ := 8

/-- The number of digits available in the restricted set (0-5) -/
def restricted_digits : ℕ := 6

/-- The number of digits needed to represent the largest number in our set in base 8 -/
def num_digits : ℕ := 3

theorem count_numbers_with_6_or_7_proof :
  count_numbers_with_6_or_7 = total_numbers - restricted_digits ^ num_digits :=
by sorry

end NUMINAMATH_CALUDE_count_numbers_with_6_or_7_proof_l71_7194


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l71_7137

theorem inscribed_squares_ratio : 
  let triangle1 : ℝ × ℝ × ℝ := (5, 12, 13)
  let triangle2 : ℝ × ℝ × ℝ := (5, 12, 13)
  let a := (60 : ℝ) / 17  -- side length of square in triangle1
  let b := (65 : ℝ) / 17  -- side length of square in triangle2
  (a^2) / (b^2) = 3600 / 4225 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l71_7137


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l71_7119

theorem constant_term_binomial_expansion :
  let n : ℕ := 8
  let a : ℝ := 1
  let b : ℝ := 1/3
  (Finset.sum (Finset.range (n + 1)) (λ k => Nat.choose n k * a^k * b^(n - k) * (if k = n/2 then 1 else 0))) = 28 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l71_7119


namespace NUMINAMATH_CALUDE_square_area_l71_7131

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The line function -/
def line : ℝ := 8

theorem square_area : ∃ (x₁ x₂ : ℝ), 
  parabola x₁ = line ∧ 
  parabola x₂ = line ∧ 
  (x₂ - x₁)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l71_7131


namespace NUMINAMATH_CALUDE_building_shadow_length_l71_7135

/-- Given a flagpole and a building under similar lighting conditions, 
    this theorem proves the length of the building's shadow. -/
theorem building_shadow_length 
  (flagpole_height : ℝ) 
  (flagpole_shadow : ℝ) 
  (building_height : ℝ) 
  (h1 : flagpole_height = 18) 
  (h2 : flagpole_shadow = 45) 
  (h3 : building_height = 28) : 
  ∃ (building_shadow : ℝ), building_shadow = 70 ∧ 
  flagpole_height / flagpole_shadow = building_height / building_shadow :=
sorry

end NUMINAMATH_CALUDE_building_shadow_length_l71_7135


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l71_7112

/-- Given a rhombus with area 432 sq m and one diagonal 24 m, prove the other diagonal is 36 m -/
theorem rhombus_diagonal (area : ℝ) (diagonal2 : ℝ) (diagonal1 : ℝ) : 
  area = 432 → diagonal2 = 24 → area = (diagonal1 * diagonal2) / 2 → diagonal1 = 36 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l71_7112


namespace NUMINAMATH_CALUDE_total_hats_bought_l71_7179

theorem total_hats_bought (blue_hat_price green_hat_price total_price green_hats : ℕ)
  (h1 : blue_hat_price = 6)
  (h2 : green_hat_price = 7)
  (h3 : total_price = 540)
  (h4 : green_hats = 30) :
  ∃ (blue_hats : ℕ), blue_hats * blue_hat_price + green_hats * green_hat_price = total_price ∧
                     blue_hats + green_hats = 85 := by
  sorry

end NUMINAMATH_CALUDE_total_hats_bought_l71_7179


namespace NUMINAMATH_CALUDE_base_number_power_remainder_l71_7176

theorem base_number_power_remainder (base : ℕ) : base = 1 → base ^ 8 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_base_number_power_remainder_l71_7176


namespace NUMINAMATH_CALUDE_train_length_l71_7181

/-- The length of a train given specific conditions --/
theorem train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 280 →
  passing_time = 40 →
  (train_speed - jogger_speed) * passing_time + initial_distance = 680 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l71_7181


namespace NUMINAMATH_CALUDE_complex_statements_l71_7149

open Complex

theorem complex_statements :
  (∃ z : ℂ, z = 1 - I ∧ Complex.abs (2 / z + z^2) = Real.sqrt 2) ∧
  (∃ z : ℂ, z = 1 / I ∧ (z^5 + 1).re > 0 ∧ (z^5 + 1).im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_statements_l71_7149


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l71_7144

theorem min_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_min_sum_reciprocals_l71_7144


namespace NUMINAMATH_CALUDE_data_analytics_course_hours_l71_7113

/-- Calculates the total hours spent on a course given the course duration and weekly schedule. -/
def total_course_hours (weeks : ℕ) (three_hour_classes : ℕ) (four_hour_classes : ℕ) (homework_hours : ℕ) : ℕ :=
  weeks * (three_hour_classes * 3 + four_hour_classes * 4 + homework_hours)

/-- Proves that the total hours spent on the given course is 336. -/
theorem data_analytics_course_hours : 
  total_course_hours 24 2 1 4 = 336 := by
  sorry

end NUMINAMATH_CALUDE_data_analytics_course_hours_l71_7113


namespace NUMINAMATH_CALUDE_triangle_third_side_count_l71_7164

theorem triangle_third_side_count : 
  let side1 : ℕ := 8
  let side2 : ℕ := 12
  let valid_third_side (x : ℕ) : Prop := 
    x + side1 > side2 ∧ 
    x + side2 > side1 ∧ 
    side1 + side2 > x
  (∃! (n : ℕ), (∀ (x : ℕ), valid_third_side x ↔ x ∈ Finset.range n) ∧ n = 15) :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_count_l71_7164


namespace NUMINAMATH_CALUDE_car_both_ways_time_l71_7185

/-- Represents the time in hours for different travel scenarios -/
structure TravelTime where
  mixedTrip : ℝ  -- Time for walking one way and taking a car back
  walkingBothWays : ℝ  -- Time for walking both ways
  carBothWays : ℝ  -- Time for taking a car both ways

/-- Proves that given the conditions, the time taken if taking a car both ways is 30 minutes -/
theorem car_both_ways_time (t : TravelTime) 
  (h1 : t.mixedTrip = 1.5)
  (h2 : t.walkingBothWays = 2.5) : 
  t.carBothWays * 60 = 30 := by
  sorry

end NUMINAMATH_CALUDE_car_both_ways_time_l71_7185


namespace NUMINAMATH_CALUDE_fruit_purchase_price_l71_7165

/-- The price of an orange in cents -/
def orange_price : ℕ := 3000

/-- The price of a pear in cents -/
def pear_price : ℕ := 9000

/-- The price of a banana in cents -/
def banana_price : ℕ := pear_price - orange_price

/-- The total cost of an orange and a pear in cents -/
def orange_pear_total : ℕ := orange_price + pear_price

/-- The total cost of 50 oranges and 25 bananas in cents -/
def fifty_orange_twentyfive_banana : ℕ := 50 * orange_price + 25 * banana_price

/-- The number of items purchased -/
def total_items : ℕ := 200 + 400

/-- The discount rate as a rational number -/
def discount_rate : ℚ := 1 / 10

theorem fruit_purchase_price :
  orange_pear_total = 12000 ∧
  fifty_orange_twentyfive_banana % 700 = 0 ∧
  total_items > 300 →
  (200 * banana_price + 400 * orange_price) * (1 - discount_rate) = 2160000 := by
  sorry

end NUMINAMATH_CALUDE_fruit_purchase_price_l71_7165


namespace NUMINAMATH_CALUDE_sin_cos_sum_13_17_l71_7111

theorem sin_cos_sum_13_17 : 
  Real.sin (13 * π / 180) * Real.cos (17 * π / 180) + 
  Real.cos (13 * π / 180) * Real.sin (17 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_13_17_l71_7111


namespace NUMINAMATH_CALUDE_sum_of_first_12_mod_9_l71_7106

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_of_first_12_mod_9 : sum_of_first_n 12 % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_12_mod_9_l71_7106


namespace NUMINAMATH_CALUDE_abc_relationship_l71_7140

open Real

noncomputable def a : ℝ := (1 / Real.sqrt 2) * (cos (34 * π / 180) - sin (34 * π / 180))

noncomputable def b : ℝ := cos (50 * π / 180) * cos (128 * π / 180) + cos (40 * π / 180) * cos (38 * π / 180)

noncomputable def c : ℝ := (1 / 2) * (cos (80 * π / 180) - 2 * (cos (50 * π / 180))^2 + 1)

theorem abc_relationship : b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_abc_relationship_l71_7140


namespace NUMINAMATH_CALUDE_plant_branches_l71_7189

theorem plant_branches : ∃ (x : ℕ), x > 0 ∧ 1 + x + x * x = 57 := by
  sorry

end NUMINAMATH_CALUDE_plant_branches_l71_7189


namespace NUMINAMATH_CALUDE_rachel_math_homework_l71_7123

theorem rachel_math_homework (total_math_bio : ℕ) (bio_pages : ℕ) (h1 : total_math_bio = 11) (h2 : bio_pages = 3) :
  total_math_bio - bio_pages = 8 :=
by sorry

end NUMINAMATH_CALUDE_rachel_math_homework_l71_7123


namespace NUMINAMATH_CALUDE_fifth_power_fourth_decomposition_l71_7129

/-- 
For a natural number m ≥ 2, m^4 can be decomposed into a sum of m consecutive odd numbers.
This function returns the starting odd number for this decomposition.
-/
def startingOddNumber (m : ℕ) : ℕ := 
  if m = 2 then 7 else 2 * (((m - 1) + 2) * (m - 2) / 2) + 1

/-- 
This function returns the nth odd number in the sequence starting from a given odd number.
-/
def nthOddNumber (start : ℕ) (n : ℕ) : ℕ := start + 2 * (n - 1)

theorem fifth_power_fourth_decomposition : 
  nthOddNumber (startingOddNumber 5) 3 = 125 := by sorry

end NUMINAMATH_CALUDE_fifth_power_fourth_decomposition_l71_7129


namespace NUMINAMATH_CALUDE_inscribed_box_radius_l71_7122

/-- A rectangular box inscribed in a sphere -/
structure InscribedBox where
  s : ℝ  -- radius of the sphere
  a : ℝ  -- length of the box
  b : ℝ  -- width of the box
  c : ℝ  -- height of the box

/-- The sum of the lengths of the 12 edges of the box -/
def edge_sum (box : InscribedBox) : ℝ := 4 * (box.a + box.b + box.c)

/-- The surface area of the box -/
def surface_area (box : InscribedBox) : ℝ := 2 * (box.a * box.b + box.b * box.c + box.c * box.a)

/-- The main theorem -/
theorem inscribed_box_radius (box : InscribedBox) 
  (h1 : edge_sum box = 72)
  (h2 : surface_area box = 216) :
  box.s = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_box_radius_l71_7122


namespace NUMINAMATH_CALUDE_certain_number_equation_l71_7192

theorem certain_number_equation (x : ℝ) : 0.85 * 40 = (4/5) * x + 14 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l71_7192


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l71_7197

/-- Given the purchase of blankets with specific quantities and prices, 
    prove that the unknown rate for two blankets is 225 Rs. -/
theorem unknown_blanket_rate : 
  ∀ (unknown_rate : ℕ),
  (3 * 100 + 2 * 150 + 2 * unknown_rate) / 7 = 150 →
  unknown_rate = 225 := by
sorry

end NUMINAMATH_CALUDE_unknown_blanket_rate_l71_7197


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l71_7157

theorem quadratic_root_relation : 
  ∀ x₁ x₂ : ℝ, 
  x₁^2 - 2*x₁ - 8 = 0 → 
  x₂^2 - 2*x₂ - 8 = 0 → 
  (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l71_7157


namespace NUMINAMATH_CALUDE_simple_compound_interest_equivalence_l71_7184

theorem simple_compound_interest_equivalence (P : ℝ) : 
  (P * 0.04 * 2 = 0.5 * (4000 * ((1 + 0.10)^2 - 1))) → P = 5250 :=
by sorry

end NUMINAMATH_CALUDE_simple_compound_interest_equivalence_l71_7184


namespace NUMINAMATH_CALUDE_sum_of_cubes_l71_7155

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l71_7155


namespace NUMINAMATH_CALUDE_num_organizations_in_foundation_l71_7115

/-- The number of organizations in a public foundation --/
def num_organizations (total_raised : ℚ) (donation_percentage : ℚ) (amount_per_org : ℚ) : ℚ :=
  (total_raised * donation_percentage) / amount_per_org

/-- Theorem stating the number of organizations in the public foundation --/
theorem num_organizations_in_foundation : 
  num_organizations 2500 0.8 250 = 8 := by
  sorry

end NUMINAMATH_CALUDE_num_organizations_in_foundation_l71_7115


namespace NUMINAMATH_CALUDE_correct_propositions_l71_7177

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (parallel_lines : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem correct_propositions
  (m n : Line) (α β : Plane) 
  (h_distinct_lines : m ≠ n)
  (h_distinct_planes : α ≠ β) :
  -- Proposition 2
  (parallel_planes α β ∧ subset m α → parallel_lines m β) ∧
  -- Proposition 3
  (perp n α ∧ perp n β ∧ perp m α → perp m β) :=
by sorry

end NUMINAMATH_CALUDE_correct_propositions_l71_7177


namespace NUMINAMATH_CALUDE_not_p_and_q_implies_at_most_one_true_l71_7160

theorem not_p_and_q_implies_at_most_one_true (p q : Prop) :
  ¬(p ∧ q) → (¬p ∨ ¬q) :=
by
  sorry

end NUMINAMATH_CALUDE_not_p_and_q_implies_at_most_one_true_l71_7160


namespace NUMINAMATH_CALUDE_book_cost_in_rubles_l71_7180

/-- Represents the exchange rate between US dollars and Namibian dollars -/
def usd_to_namibian : ℚ := 10

/-- Represents the exchange rate between US dollars and Russian rubles -/
def usd_to_rubles : ℚ := 8

/-- Represents the cost of the book in Namibian dollars -/
def book_cost_namibian : ℚ := 200

/-- Theorem stating that the cost of the book in Russian rubles is 160 -/
theorem book_cost_in_rubles :
  (book_cost_namibian / usd_to_namibian) * usd_to_rubles = 160 := by
  sorry


end NUMINAMATH_CALUDE_book_cost_in_rubles_l71_7180


namespace NUMINAMATH_CALUDE_simplify_expression_l71_7127

theorem simplify_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (sum : a + b + c = 1) :
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l71_7127


namespace NUMINAMATH_CALUDE_unique_integer_value_l71_7110

theorem unique_integer_value : ∃! x : ℤ, 
  1 < x ∧ x < 9 ∧ 
  2 < x ∧ x < 15 ∧ 
  -1 < x ∧ x < 7 ∧ 
  0 < x ∧ x < 4 ∧ 
  x + 1 < 5 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_integer_value_l71_7110


namespace NUMINAMATH_CALUDE_greatest_divisible_integer_l71_7159

theorem greatest_divisible_integer (m : ℕ+) :
  (∃ (n : ℕ), n > 0 ∧ (m^2 + n) ∣ (n^2 + m)) ∧
  (∀ (k : ℕ), k > (m^4 - m^2 + m) → ¬((m^2 + k) ∣ (k^2 + m))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisible_integer_l71_7159


namespace NUMINAMATH_CALUDE_first_machine_rate_is_35_l71_7161

/-- The number of copies the first machine makes per minute -/
def first_machine_rate : ℝ := sorry

/-- The number of copies the second machine makes per minute -/
def second_machine_rate : ℝ := 75

/-- The total number of copies both machines make in 30 minutes -/
def total_copies : ℝ := 3300

/-- The time period in minutes -/
def time_period : ℝ := 30

theorem first_machine_rate_is_35 :
  first_machine_rate = 35 :=
by
  have h1 : first_machine_rate * time_period + second_machine_rate * time_period = total_copies :=
    sorry
  sorry

#check first_machine_rate_is_35

end NUMINAMATH_CALUDE_first_machine_rate_is_35_l71_7161


namespace NUMINAMATH_CALUDE_probability_of_matching_pair_l71_7187

def blue_socks : ℕ := 12
def gray_socks : ℕ := 10
def white_socks : ℕ := 8

def total_socks : ℕ := blue_socks + gray_socks + white_socks

def ways_to_pick_two : ℕ := total_socks.choose 2

def matching_blue_pairs : ℕ := blue_socks.choose 2
def matching_gray_pairs : ℕ := gray_socks.choose 2
def matching_white_pairs : ℕ := white_socks.choose 2

def total_matching_pairs : ℕ := matching_blue_pairs + matching_gray_pairs + matching_white_pairs

theorem probability_of_matching_pair :
  (total_matching_pairs : ℚ) / ways_to_pick_two = 139 / 435 := by sorry

end NUMINAMATH_CALUDE_probability_of_matching_pair_l71_7187


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l71_7163

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

/-- The theorem stating that if points A(m-1, -3) and B(2, n) are symmetric
    with respect to the origin, then m + n = 2 -/
theorem symmetric_points_sum (m n : ℝ) :
  symmetric_wrt_origin (m - 1) (-3) 2 n → m + n = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l71_7163


namespace NUMINAMATH_CALUDE_library_visitors_theorem_l71_7156

/-- Calculates the average number of visitors per day in a library for a 30-day month -/
def average_visitors (sundays : Nat) (sunday_visitors : Nat) (regular_visitors : Nat) (holiday_visitors : Nat) (holidays : Nat) : Nat :=
  let regular_days := 30 - sundays - holidays
  let total_visitors := sundays * sunday_visitors + regular_days * regular_visitors + holidays * holiday_visitors
  total_visitors / 30

/-- Theorem stating the average number of visitors for different scenarios -/
theorem library_visitors_theorem (sundays : Nat) (sunday_visitors : Nat) (regular_visitors : Nat) (holiday_visitors : Nat) (holidays : Nat) :
  (sundays = 4 ∨ sundays = 5) →
  sunday_visitors = 510 →
  regular_visitors = 240 →
  holiday_visitors = 375 →
  holidays = 2 →
  (average_visitors sundays sunday_visitors regular_visitors holiday_visitors holidays = 
    if sundays = 4 then 285 else 294) :=
by
  sorry

#eval average_visitors 4 510 240 375 2
#eval average_visitors 5 510 240 375 2

end NUMINAMATH_CALUDE_library_visitors_theorem_l71_7156


namespace NUMINAMATH_CALUDE_anna_candy_count_l71_7132

theorem anna_candy_count (initial_candies : ℕ) (received_candies : ℕ) :
  initial_candies = 5 →
  received_candies = 86 →
  initial_candies + received_candies = 91 := by
  sorry

end NUMINAMATH_CALUDE_anna_candy_count_l71_7132


namespace NUMINAMATH_CALUDE_parking_fee_calculation_l71_7196

/-- Calculates the parking fee based on the given fee structure and parking duration. -/
def parking_fee (initial_fee : ℕ) (additional_fee : ℕ) (initial_duration : ℕ) (increment : ℕ) (total_duration : ℕ) : ℕ :=
  let extra_duration := total_duration - initial_duration
  let extra_increments := (extra_duration + increment - 1) / increment
  initial_fee + additional_fee * extra_increments

/-- Theorem stating that the parking fee for 80 minutes is 1500 won given the specified fee structure. -/
theorem parking_fee_calculation : parking_fee 500 200 30 10 80 = 1500 := by
  sorry

#eval parking_fee 500 200 30 10 80

end NUMINAMATH_CALUDE_parking_fee_calculation_l71_7196


namespace NUMINAMATH_CALUDE_limit_of_sequence_a_l71_7124

def a (n : ℕ) : ℚ := (3 - n^2) / (4 + 2*n^2)

theorem limit_of_sequence_a :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - (-1/2)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_sequence_a_l71_7124


namespace NUMINAMATH_CALUDE_complement_of_M_l71_7130

-- Define the universal set U
def U : Set ℕ := {1, 2, 3}

-- Define the set M
def M : Set ℕ := {x ∈ U | x^2 - 3*x + 2 = 0}

-- State the theorem
theorem complement_of_M (x : ℕ) : x ∈ (U \ M) ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_l71_7130


namespace NUMINAMATH_CALUDE_terminating_decimal_of_19_80_l71_7109

theorem terminating_decimal_of_19_80 : ∃ (n : ℕ), (19 : ℚ) / 80 = (2375 : ℚ) / 10^n :=
sorry

end NUMINAMATH_CALUDE_terminating_decimal_of_19_80_l71_7109


namespace NUMINAMATH_CALUDE_gas_volume_calculation_l71_7121

/-- Calculate the volume of gas using the Mendeleev-Clapeyron equation -/
theorem gas_volume_calculation (m R T p M : ℝ) (h_m : m = 140) (h_R : R = 8.314) 
  (h_T : T = 305) (h_p : p = 283710) (h_M : M = 28) :
  let V := (m * R * T * 1000) / (p * M)
  ∃ ε > 0, |V - 44.7| < ε :=
sorry

end NUMINAMATH_CALUDE_gas_volume_calculation_l71_7121


namespace NUMINAMATH_CALUDE_floor_width_proof_l71_7143

/-- Proves that the width of a rectangular floor is 120 cm given specific conditions --/
theorem floor_width_proof (floor_length tile_length tile_width max_tiles : ℕ) 
  (h1 : floor_length = 180)
  (h2 : tile_length = 25)
  (h3 : tile_width = 16)
  (h4 : max_tiles = 54)
  (h5 : floor_length % tile_width = 0)
  (h6 : floor_length / tile_width * (floor_length / tile_width) ≤ max_tiles) :
  ∃ (floor_width : ℕ), floor_width = 120 ∧ 
    floor_length * floor_width = max_tiles * tile_length * tile_width :=
by sorry

end NUMINAMATH_CALUDE_floor_width_proof_l71_7143


namespace NUMINAMATH_CALUDE_bag_equals_two_balls_l71_7125

/-- Represents the weight of an object -/
structure Weight : Type :=
  (value : ℝ)

/-- Represents a balanced scale -/
structure BalancedScale : Type :=
  (left_bags : ℕ)
  (left_balls : ℕ)
  (right_bags : ℕ)
  (right_balls : ℕ)
  (bag_weight : Weight)
  (ball_weight : Weight)

/-- Predicate to check if a scale is balanced -/
def is_balanced (s : BalancedScale) : Prop :=
  s.left_bags * s.bag_weight.value + s.left_balls * s.ball_weight.value =
  s.right_bags * s.bag_weight.value + s.right_balls * s.ball_weight.value

theorem bag_equals_two_balls (s : BalancedScale) :
  s.left_bags = 5 ∧ s.left_balls = 4 ∧ s.right_bags = 2 ∧ s.right_balls = 10 ∧
  is_balanced s →
  s.bag_weight.value = 2 * s.ball_weight.value :=
sorry

end NUMINAMATH_CALUDE_bag_equals_two_balls_l71_7125


namespace NUMINAMATH_CALUDE_not_right_triangle_l71_7126

theorem not_right_triangle (a b c : ℕ) (h1 : a = 5) (h2 : b = 11) (h3 : c = 12) :
  ¬(a^2 + b^2 = c^2) := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_l71_7126


namespace NUMINAMATH_CALUDE_gold_alloy_calculation_l71_7138

theorem gold_alloy_calculation (initial_weight : ℝ) (initial_gold_percentage : ℝ) 
  (target_gold_percentage : ℝ) (added_gold : ℝ) : 
  initial_weight = 16 →
  initial_gold_percentage = 0.5 →
  target_gold_percentage = 0.8 →
  added_gold = 24 →
  (initial_weight * initial_gold_percentage + added_gold) / (initial_weight + added_gold) = target_gold_percentage :=
by
  sorry

end NUMINAMATH_CALUDE_gold_alloy_calculation_l71_7138


namespace NUMINAMATH_CALUDE_find_number_l71_7103

theorem find_number (x : ℤ) : 
  (∃ q r : ℤ, 5 * (x + 3) = 8 * q + r ∧ q = 156 ∧ r = 2) → x = 247 := by
sorry

end NUMINAMATH_CALUDE_find_number_l71_7103


namespace NUMINAMATH_CALUDE_base_seven_54321_to_decimal_l71_7151

def base_seven_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

theorem base_seven_54321_to_decimal :
  base_seven_to_decimal [1, 2, 3, 4, 5] = 13539 :=
by sorry

end NUMINAMATH_CALUDE_base_seven_54321_to_decimal_l71_7151


namespace NUMINAMATH_CALUDE_x_intercepts_count_l71_7136

theorem x_intercepts_count (x : ℝ) : 
  (∃! x, (x - 4) * (x^2 + 4*x + 8) = 0) := by
  sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l71_7136


namespace NUMINAMATH_CALUDE_min_value_a_l71_7166

theorem min_value_a (a : ℝ) (h1 : a > 1) :
  (∀ x : ℝ, x ≥ 1/3 → (1/(3*x) - 2*x + Real.log (3*x) ≤ 1/(a*(Real.exp (2*x))) + Real.log a)) →
  a ≥ 3/(2*(Real.exp 1)) := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_l71_7166


namespace NUMINAMATH_CALUDE_cards_left_calculation_l71_7178

def initial_cards : ℕ := 455
def cards_given_away : ℕ := 301

theorem cards_left_calculation : initial_cards - cards_given_away = 154 := by
  sorry

end NUMINAMATH_CALUDE_cards_left_calculation_l71_7178
