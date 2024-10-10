import Mathlib

namespace triangle_bisector_inequality_l2703_270366

/-- Given a triangle ABC with side lengths a, b, c, semiperimeter p, circumradius R,
    inradius r, and angle bisector lengths l_a, l_b, l_c, prove that
    l_a * l_b + l_b * l_c + l_c * l_a ≤ p * √(3r² + 12Rr) -/
theorem triangle_bisector_inequality
  (a b c : ℝ)
  (p : ℝ)
  (R r : ℝ)
  (l_a l_b l_c : ℝ)
  (h_p : p = (a + b + c) / 2)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_R : R > 0)
  (h_r : r > 0)
  (h_l_a : l_a > 0)
  (h_l_b : l_b > 0)
  (h_l_c : l_c > 0) :
  l_a * l_b + l_b * l_c + l_c * l_a ≤ p * Real.sqrt (3 * r^2 + 12 * R * r) :=
by sorry

end triangle_bisector_inequality_l2703_270366


namespace red_chips_probability_l2703_270352

/-- Represents the outcome of drawing chips from a hat -/
inductive DrawOutcome
| AllRed
| AllGreen

/-- Represents a hat with red and green chips -/
structure Hat :=
  (redChips : ℕ)
  (greenChips : ℕ)

/-- Represents the probability of an outcome -/
def probability (outcome : DrawOutcome) (hat : Hat) : ℚ :=
  sorry

theorem red_chips_probability (hat : Hat) :
  hat.redChips = 3 ∧ hat.greenChips = 3 →
  probability DrawOutcome.AllRed hat = 1/2 :=
by sorry

end red_chips_probability_l2703_270352


namespace equivalent_operation_l2703_270320

theorem equivalent_operation (x : ℝ) : 
  (x * (2/5)) / (3/7) = x * (14/15) := by
  sorry

end equivalent_operation_l2703_270320


namespace min_value_xy_min_value_xy_achieved_l2703_270332

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y + 6 = x*y) :
  x*y ≥ 18 := by
  sorry

theorem min_value_xy_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2*x + y + 6 = x*y ∧ x*y < 18 + ε := by
  sorry

end min_value_xy_min_value_xy_achieved_l2703_270332


namespace circle_tangent_k_range_l2703_270354

/-- Represents a circle in the 2D plane --/
structure Circle where
  k : ℝ

/-- Represents a point in the 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is outside the circle --/
def isOutside (p : Point) (c : Circle) : Prop :=
  p.x^2 + p.y^2 + 2*p.x + 2*p.y + c.k > 0

/-- Checks if two tangents can be drawn from a point to the circle --/
def hasTwoTangents (p : Point) (c : Circle) : Prop :=
  isOutside p c

/-- The main theorem --/
theorem circle_tangent_k_range (c : Circle) :
  let p : Point := ⟨1, -1⟩
  hasTwoTangents p c → -2 < c.k ∧ c.k < 2 :=
by sorry

end circle_tangent_k_range_l2703_270354


namespace curve_C_symmetry_l2703_270318

/-- The curve C in the Cartesian coordinate system -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ((p.1 - 1)^2 + p.2^2) * ((p.1 + 1)^2 + p.2^2) = 4}

/-- A point is symmetric about the x-axis -/
def symmetric_x (p : ℝ × ℝ) : Prop := (p.1, -p.2) ∈ C ↔ p ∈ C

/-- A point is symmetric about the y-axis -/
def symmetric_y (p : ℝ × ℝ) : Prop := (-p.1, p.2) ∈ C ↔ p ∈ C

/-- A point is symmetric about the origin -/
def symmetric_origin (p : ℝ × ℝ) : Prop := (-p.1, -p.2) ∈ C ↔ p ∈ C

theorem curve_C_symmetry :
  (∀ p ∈ C, symmetric_x p ∧ symmetric_y p) ∧
  (∀ p ∈ C, symmetric_origin p) := by sorry

end curve_C_symmetry_l2703_270318


namespace all_propositions_true_l2703_270325

-- Define the original proposition
def original_proposition (x y : ℝ) : Prop :=
  x^2 + y^2 = 0 → x = 0 ∧ y = 0

-- Define the converse proposition
def converse_proposition (x y : ℝ) : Prop :=
  x = 0 ∧ y = 0 → x^2 + y^2 = 0

-- Define the inverse proposition
def inverse_proposition (x y : ℝ) : Prop :=
  x^2 + y^2 ≠ 0 → x ≠ 0 ∨ y ≠ 0

-- Define the contrapositive proposition
def contrapositive_proposition (x y : ℝ) : Prop :=
  x ≠ 0 ∨ y ≠ 0 → x^2 + y^2 ≠ 0

-- Theorem stating that all propositions are true
theorem all_propositions_true :
  ∀ x y : ℝ,
    original_proposition x y ∧
    converse_proposition x y ∧
    inverse_proposition x y ∧
    contrapositive_proposition x y :=
by
  sorry


end all_propositions_true_l2703_270325


namespace monotonic_quadratic_l2703_270359

/-- The function f(x) = ax² + 2x - 3 is monotonically increasing on (-∞, 4) iff -1/4 ≤ a ≤ 0 -/
theorem monotonic_quadratic (a : ℝ) :
  (∀ x y : ℝ, x < y → x < 4 → y < 4 → a * x^2 + 2 * x - 3 < a * y^2 + 2 * y - 3) ↔
  -1/4 ≤ a ∧ a ≤ 0 := by sorry

end monotonic_quadratic_l2703_270359


namespace square_root_equation_l2703_270391

theorem square_root_equation (x : ℝ) : Real.sqrt (x - 3) = 10 → x = 103 := by
  sorry

end square_root_equation_l2703_270391


namespace ratio_equality_l2703_270331

theorem ratio_equality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_eq : y / (x - y) = (x - y) / z ∧ (x - y) / z = z / (x + y)) :
  x / y = 1 := by
  sorry

end ratio_equality_l2703_270331


namespace stone_length_proof_l2703_270348

/-- Given a hall and stones with specific dimensions, prove the length of each stone --/
theorem stone_length_proof (hall_length hall_width : ℝ) (stone_width : ℝ) (num_stones : ℕ) 
  (h1 : hall_length = 36)
  (h2 : hall_width = 15)
  (h3 : stone_width = 0.5)
  (h4 : num_stones = 1800) :
  (hall_length * hall_width * 100) / (stone_width * 10 * num_stones) = 6 := by
  sorry

end stone_length_proof_l2703_270348


namespace prime_quadratic_l2703_270327

theorem prime_quadratic (a : ℕ) : 
  Nat.Prime (a^2 - 10*a + 21) ↔ a = 2 ∨ a = 8 := by sorry

end prime_quadratic_l2703_270327


namespace arithmetic_progression_perfect_squares_l2703_270310

theorem arithmetic_progression_perfect_squares :
  ∃ (a b c : ℤ),
    b - a = c - b ∧
    ∃ (x y z : ℤ),
      a + b = x^2 ∧
      a + c = y^2 ∧
      b + c = z^2 ∧
      a = 482 ∧
      b = 3362 ∧
      c = 6242 := by
sorry

end arithmetic_progression_perfect_squares_l2703_270310


namespace product_and_ratio_implies_y_value_l2703_270371

theorem product_and_ratio_implies_y_value 
  (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x * y = 9) 
  (h4 : x / y = 36) : 
  y = 1/2 := by
sorry

end product_and_ratio_implies_y_value_l2703_270371


namespace lg_100_equals_2_l2703_270300

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_100_equals_2 : lg 100 = 2 := by
  sorry

end lg_100_equals_2_l2703_270300


namespace second_day_study_hours_l2703_270307

/-- Represents the relationship between study hours and performance score for a given day -/
structure StudyDay where
  hours : ℝ
  score : ℝ

/-- The constant product of hours and score, representing the inverse relationship -/
def inverse_constant (day : StudyDay) : ℝ := day.hours * day.score

theorem second_day_study_hours 
  (day1 : StudyDay)
  (avg_score : ℝ)
  (h1 : day1.hours = 5)
  (h2 : day1.score = 80)
  (h3 : avg_score = 85) :
  ∃ (day2 : StudyDay), 
    inverse_constant day1 = inverse_constant day2 ∧
    (day1.score + day2.score) / 2 = avg_score ∧
    day2.hours = 40 / 9 := by
  sorry

end second_day_study_hours_l2703_270307


namespace tank_fill_time_xy_l2703_270367

/-- Represents the time (in hours) to fill a tank given specific valve configurations -/
structure TankFillTime where
  all : ℝ
  xz : ℝ
  yz : ℝ

/-- Proves that given specific fill times for different valve configurations, 
    the time to fill the tank with only valves X and Y open is 2.4 hours -/
theorem tank_fill_time_xy (t : TankFillTime) 
  (h_all : t.all = 2)
  (h_xz : t.xz = 3)
  (h_yz : t.yz = 4) :
  1 / (1 / t.all - 1 / t.yz) + 1 / (1 / t.all - 1 / t.xz) = 2.4 := by
  sorry

#check tank_fill_time_xy

end tank_fill_time_xy_l2703_270367


namespace arithmetic_sequence_a12_l2703_270383

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  (∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d) ∧
  a 4 = -8 ∧
  a 8 = 2

theorem arithmetic_sequence_a12 (a : ℕ → ℚ) (h : arithmetic_sequence a) :
  a 12 = 12 := by
  sorry

end arithmetic_sequence_a12_l2703_270383


namespace sum_of_exponents_2023_l2703_270386

/-- Represents 2023 as a sum of distinct powers of 2 -/
def representation_2023 : List ℕ :=
  [10, 9, 8, 7, 6, 5, 2, 1, 0]

/-- The sum of the exponents in the representation of 2023 -/
def sum_of_exponents : ℕ :=
  representation_2023.sum

/-- Checks if the representation is valid -/
def is_valid_representation (n : ℕ) (rep : List ℕ) : Prop :=
  n = (rep.map (fun x => 2^x)).sum ∧ rep.Nodup

theorem sum_of_exponents_2023 :
  is_valid_representation 2023 representation_2023 ∧
  sum_of_exponents = 48 := by
  sorry

#eval sum_of_exponents -- Should output 48

end sum_of_exponents_2023_l2703_270386


namespace sneakers_price_l2703_270351

/-- Given a pair of sneakers with an unknown original price, if applying a $10 coupon 
followed by a 10% membership discount results in a final price of $99, 
then the original price of the sneakers was $120. -/
theorem sneakers_price (original_price : ℝ) : 
  (original_price - 10) * 0.9 = 99 → original_price = 120 := by
  sorry

end sneakers_price_l2703_270351


namespace special_triangle_area_property_l2703_270340

/-- A triangle with side length PQ = 30 and its incircle trisecting the median PS in ratio 1:2 -/
structure SpecialTriangle where
  -- Points of the triangle
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  -- Incircle center
  I : ℝ × ℝ
  -- Point where median PS intersects QR
  S : ℝ × ℝ
  -- Points where incircle touches the sides
  T : ℝ × ℝ  -- on QR
  U : ℝ × ℝ  -- on RP
  V : ℝ × ℝ  -- on PQ
  -- Properties
  pq_length : dist P Q = 30
  trisect_median : dist P T = (1/3) * dist P S ∧ dist T S = (2/3) * dist P S
  incircle_tangent : dist I T = dist I U ∧ dist I U = dist I V

/-- The area of the special triangle can be expressed as x√y where x and y are integers -/
def area_expression (t : SpecialTriangle) : ℕ × ℕ :=
  sorry

/-- Predicate to check if a number is not divisible by the square of any prime -/
def not_divisible_by_prime_square (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p^2 ∣ n)

theorem special_triangle_area_property (t : SpecialTriangle) :
  let (x, y) := area_expression t
  (x > 0 ∧ y > 0) ∧ not_divisible_by_prime_square y ∧ ∃ (k : ℕ), x + y = k :=
sorry

end special_triangle_area_property_l2703_270340


namespace martha_cakes_per_child_l2703_270384

/-- Given that Martha has 3.0 children and needs to buy 54 cakes in total,
    prove that each child will get 18 cakes. -/
theorem martha_cakes_per_child :
  let num_children : ℝ := 3.0
  let total_cakes : ℕ := 54
  (total_cakes : ℝ) / num_children = 18 :=
by sorry

end martha_cakes_per_child_l2703_270384


namespace austin_picked_24_bags_l2703_270372

/-- The number of bags of fruit Austin picked in total -/
def austin_total (dallas_apples dallas_pears austin_apples_diff austin_pears_diff : ℕ) : ℕ :=
  (dallas_apples + austin_apples_diff) + (dallas_pears - austin_pears_diff)

/-- Theorem stating that Austin picked 24 bags of fruit in total -/
theorem austin_picked_24_bags
  (dallas_apples : ℕ)
  (dallas_pears : ℕ)
  (austin_apples_diff : ℕ)
  (austin_pears_diff : ℕ)
  (h1 : dallas_apples = 14)
  (h2 : dallas_pears = 9)
  (h3 : austin_apples_diff = 6)
  (h4 : austin_pears_diff = 5) :
  austin_total dallas_apples dallas_pears austin_apples_diff austin_pears_diff = 24 :=
by
  sorry

end austin_picked_24_bags_l2703_270372


namespace texasCityGDP2009_scientific_notation_l2703_270377

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- The GDP of Texas City in 2009 in billion yuan -/
def texasCityGDP2009 : ℝ := 1545.35

theorem texasCityGDP2009_scientific_notation :
  toScientificNotation (texasCityGDP2009 * 1000000000) 3 =
    ScientificNotation.mk 1.55 11 (by norm_num) :=
  sorry

end texasCityGDP2009_scientific_notation_l2703_270377


namespace basketball_max_score_l2703_270369

def max_individual_score (n : ℕ) (total_points : ℕ) (min_points : ℕ) : ℕ :=
  total_points - (n - 1) * min_points

theorem basketball_max_score :
  max_individual_score 12 100 7 = 23 :=
by sorry

end basketball_max_score_l2703_270369


namespace volume_of_specific_open_box_l2703_270363

/-- Calculates the volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
def openBoxVolume (sheetLength sheetWidth cutSize : ℝ) : ℝ :=
  (sheetLength - 2 * cutSize) * (sheetWidth - 2 * cutSize) * cutSize

/-- Theorem stating that the volume of the specific open box is 5120 cubic meters. -/
theorem volume_of_specific_open_box :
  openBoxVolume 48 36 8 = 5120 := by
  sorry

end volume_of_specific_open_box_l2703_270363


namespace triangle_properties_main_theorem_l2703_270324

-- Define the triangle PQR
structure RightTriangle where
  PQ : ℝ
  QR : ℝ
  cosQ : ℝ

-- Define our specific triangle
def trianglePQR : RightTriangle where
  PQ := 15
  QR := 30  -- We'll prove this
  cosQ := 0.5

-- Theorem to prove QR = 30 and area = 225
theorem triangle_properties (t : RightTriangle) 
  (h1 : t.PQ = 15) 
  (h2 : t.cosQ = 0.5) : 
  t.QR = 30 ∧ (1/2 * t.PQ * t.QR) = 225 := by
  sorry

-- Main theorem combining all properties
theorem main_theorem : 
  trianglePQR.QR = 30 ∧ 
  (1/2 * trianglePQR.PQ * trianglePQR.QR) = 225 := by
  sorry

end triangle_properties_main_theorem_l2703_270324


namespace area_ratio_theorem_l2703_270378

-- Define an equilateral triangle ABC with side length s
def equilateral_triangle (A B C : ℝ × ℝ) (s : ℝ) : Prop :=
  dist A B = s ∧ dist B C = s ∧ dist C A = s

-- Define the extended points B', C', and A'
def extended_points (A B C A' B' C' : ℝ × ℝ) (s : ℝ) : Prop :=
  dist B B' = 2*s ∧ dist C C' = 3*s ∧ dist A A' = 4*s

-- Define the area of a triangle given its vertices
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_ratio_theorem (A B C A' B' C' : ℝ × ℝ) (s : ℝ) :
  equilateral_triangle A B C s →
  extended_points A B C A' B' C' s →
  triangle_area A' B' C' / triangle_area A B C = 60 := by sorry

end area_ratio_theorem_l2703_270378


namespace decagon_diagonal_intersection_probability_l2703_270336

/-- A regular decagon is a 10-sided polygon with all sides and angles equal -/
def RegularDecagon : Type := Unit

/-- The number of vertices in a regular decagon -/
def num_vertices : ℕ := 10

/-- The number of diagonals in a regular decagon -/
def num_diagonals : ℕ := 35

/-- The number of ways to choose 2 diagonals from the total number of diagonals -/
def num_diagonal_pairs : ℕ := 595

/-- The number of ways to choose 4 vertices from the decagon that form a convex quadrilateral -/
def num_convex_quadrilaterals : ℕ := 210

/-- The probability that two randomly chosen diagonals in a regular decagon
    intersect inside the decagon and form a convex quadrilateral -/
theorem decagon_diagonal_intersection_probability (d : RegularDecagon) :
  (num_convex_quadrilaterals : ℚ) / num_diagonal_pairs = 210 / 595 := by sorry

end decagon_diagonal_intersection_probability_l2703_270336


namespace ae_length_l2703_270353

/-- Triangle ABC and ADE share vertex A and angle A --/
structure NestedTriangles where
  AB : ℝ
  AC : ℝ
  AD : ℝ
  AE : ℝ
  k : ℝ
  area_proportion : AB * AC = k * AD * AE

/-- The specific nested triangles in the problem --/
def problem_triangles : NestedTriangles where
  AB := 5
  AC := 7
  AD := 2
  AE := 17.5
  k := 1
  area_proportion := by sorry

theorem ae_length (t : NestedTriangles) (h1 : t.AB = 5) (h2 : t.AC = 7) (h3 : t.AD = 2) (h4 : t.k = 1) :
  t.AE = 17.5 := by
  sorry

#check ae_length problem_triangles

end ae_length_l2703_270353


namespace circle_through_points_is_valid_circle_equation_l2703_270347

/-- Given three points in 2D space, this function returns true if they lie on the circle
    described by the equation x^2 + y^2 - 4x - 6y = 0 -/
def points_on_circle (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let f := fun (x y : ℝ) => x^2 + y^2 - 4*x - 6*y
  f p1.1 p1.2 = 0 ∧ f p2.1 p2.2 = 0 ∧ f p3.1 p3.2 = 0

/-- The theorem states that the points (0,0), (4,0), and (-1,1) lie on the circle
    described by the equation x^2 + y^2 - 4x - 6y = 0 -/
theorem circle_through_points :
  points_on_circle (0, 0) (4, 0) (-1, 1) := by
  sorry

/-- The general equation of a circle is x^2 + y^2 + Dx + Ey + F = 0 -/
def is_circle_equation (D E F : ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y = x^2 + y^2 + D*x + E*y + F

/-- This theorem states that the equation x^2 + y^2 - 4x - 6y = 0 is a valid circle equation -/
theorem is_valid_circle_equation :
  is_circle_equation (-4) (-6) 0 (fun x y => x^2 + y^2 - 4*x - 6*y) := by
  sorry

end circle_through_points_is_valid_circle_equation_l2703_270347


namespace y_intercept_of_line_l2703_270349

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := x - 2*y + x^2 = 8

/-- The y-intercept of the line -/
def y_intercept : ℝ := -4

/-- Theorem: The y-intercept of the line described by the equation x - 2y + x^2 = 8 is -4 -/
theorem y_intercept_of_line :
  line_equation 0 y_intercept := by sorry

end y_intercept_of_line_l2703_270349


namespace smallest_k_for_no_real_roots_l2703_270316

theorem smallest_k_for_no_real_roots : ∃ k : ℤ, k = 3 ∧ 
  (∀ x : ℝ, 3 * x * (k * x - 5) - 2 * x^2 + 8 ≠ 0) ∧
  (∀ m : ℤ, m < k → ∃ x : ℝ, 3 * x * (m * x - 5) - 2 * x^2 + 8 = 0) := by
  sorry

end smallest_k_for_no_real_roots_l2703_270316


namespace roxanne_sandwiches_l2703_270390

theorem roxanne_sandwiches (lemonade_price : ℚ) (sandwich_price : ℚ) 
  (lemonade_count : ℕ) (paid : ℚ) (change : ℚ) :
  lemonade_price = 2 →
  sandwich_price = 5/2 →
  lemonade_count = 2 →
  paid = 20 →
  change = 11 →
  (paid - change - lemonade_price * lemonade_count) / sandwich_price = 2 :=
by sorry

end roxanne_sandwiches_l2703_270390


namespace triangle_properties_l2703_270306

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  a / (Real.sin A) = b / (Real.sin B) →
  b / (Real.sin B) = c / (Real.sin C) →
  c * (Real.cos A) + Real.sqrt 3 * c * (Real.sin A) - b - a = 0 →
  (C = Real.pi / 3 ∧
   (c = 1 → ∀ (a' b' : ℝ), a' > 0 → b' > 0 → a' + b' > c →
     1 / 2 * a' * b' * Real.sin C ≤ Real.sqrt 3 / 4)) :=
by sorry

end triangle_properties_l2703_270306


namespace kelly_games_to_give_away_l2703_270397

theorem kelly_games_to_give_away (initial_games : ℕ) (remaining_games : ℕ) : 
  initial_games - remaining_games = 15 :=
by
  sorry

#check kelly_games_to_give_away 50 35

end kelly_games_to_give_away_l2703_270397


namespace sum_of_variables_l2703_270373

theorem sum_of_variables (x y z : ℚ) 
  (eq1 : y + z = 18 - 4*x)
  (eq2 : x + z = 16 - 4*y)
  (eq3 : x + y = 9 - 4*z) :
  3*x + 3*y + 3*z = 43/2 := by
sorry

end sum_of_variables_l2703_270373


namespace final_balance_calculation_l2703_270323

def initial_balance : ℕ := 65
def deposit : ℕ := 15
def withdrawal : ℕ := 4

theorem final_balance_calculation : 
  initial_balance + deposit - withdrawal = 76 := by sorry

end final_balance_calculation_l2703_270323


namespace cunningham_white_lambs_l2703_270355

/-- The number of white lambs owned by farmer Cunningham -/
def white_lambs (total : ℕ) (black : ℕ) : ℕ := total - black

theorem cunningham_white_lambs :
  white_lambs 6048 5855 = 193 :=
by sorry

end cunningham_white_lambs_l2703_270355


namespace pats_calculation_l2703_270387

theorem pats_calculation (x : ℝ) : (x / 8 - 20 = 12) → (x * 8 + 20 = 2068) := by
  sorry

end pats_calculation_l2703_270387


namespace trailing_zeros_count_l2703_270338

/-- The number of trailing zeros in (10¹² - 25)² is 12 -/
theorem trailing_zeros_count : ∃ n : ℕ, n > 0 ∧ (10^12 - 25)^2 = n * 10^12 ∧ n % 10 ≠ 0 := by
  sorry

end trailing_zeros_count_l2703_270338


namespace sine_cosine_inequality_l2703_270381

theorem sine_cosine_inequality (a : ℝ) :
  (a < 0) →
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a ^ 2 ≥ 1 + Real.cos x) ↔
  (a ≤ -2) := by
sorry

end sine_cosine_inequality_l2703_270381


namespace divided_square_area_l2703_270392

/-- Represents a square divided into rectangles -/
structure DividedSquare where
  side_length : ℝ
  vertical_lines : ℕ
  horizontal_lines : ℕ

/-- Calculates the total perimeter of all rectangles in a divided square -/
def total_perimeter (s : DividedSquare) : ℝ :=
  4 * s.side_length + 2 * s.side_length * (s.vertical_lines * (s.horizontal_lines + 1) + s.horizontal_lines * (s.vertical_lines + 1))

/-- The main theorem -/
theorem divided_square_area (s : DividedSquare) 
  (h1 : s.vertical_lines = 5)
  (h2 : s.horizontal_lines = 3)
  (h3 : (s.vertical_lines + 1) * (s.horizontal_lines + 1) = 24)
  (h4 : total_perimeter s = 24) :
  s.side_length ^ 2 = 1 := by
  sorry

end divided_square_area_l2703_270392


namespace triangle_problem_l2703_270305

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  3 * a * Real.cos C = 2 * c * Real.cos A →
  b = 2 * Real.sqrt 5 →
  c = 3 →
  (a = Real.sqrt 5 ∧
   Real.sin (B + π / 4) = Real.sqrt 10 / 10) := by
  sorry

end triangle_problem_l2703_270305


namespace cube_volume_from_face_perimeter_l2703_270342

theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 32) :
  let side_length := face_perimeter / 4
  let volume := side_length ^ 3
  volume = 512 := by sorry

end cube_volume_from_face_perimeter_l2703_270342


namespace restaurant_cooks_l2703_270370

theorem restaurant_cooks (initial_cooks : ℕ) (initial_waiters : ℕ) : 
  initial_cooks / initial_waiters = 3 / 11 →
  initial_cooks / (initial_waiters + 12) = 1 / 5 →
  initial_cooks = 9 := by
sorry

end restaurant_cooks_l2703_270370


namespace negation_of_forall_square_geq_one_l2703_270315

theorem negation_of_forall_square_geq_one :
  ¬(∀ x : ℝ, x ≥ 1 → x^2 ≥ 1) ↔ ∃ x : ℝ, x ≥ 1 ∧ x^2 < 1 := by sorry

end negation_of_forall_square_geq_one_l2703_270315


namespace right_triangle_area_l2703_270365

theorem right_triangle_area (leg1 leg2 : ℝ) (h1 : leg1 = 45) (h2 : leg2 = 48) :
  (1/2 : ℝ) * leg1 * leg2 = 1080 := by
  sorry

end right_triangle_area_l2703_270365


namespace inequality_proof_l2703_270380

theorem inequality_proof (x y : ℝ) (h : x^12 + y^12 ≤ 2) : x^2 + y^2 + x^2*y^2 ≤ 3 := by
  sorry

end inequality_proof_l2703_270380


namespace c_less_than_a_l2703_270312

theorem c_less_than_a (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0)
  (h1 : c / (a + b) = 2) (h2 : c / (b - a) = 3) : c < a := by
  sorry

end c_less_than_a_l2703_270312


namespace least_seven_digit_binary_l2703_270319

theorem least_seven_digit_binary : ∀ n : ℕ, 
  (n < 64 → (Nat.log2 n).succ < 7) ∧ 
  ((Nat.log2 64).succ = 7) :=
sorry

end least_seven_digit_binary_l2703_270319


namespace remaining_dimes_l2703_270330

def initial_dimes : ℕ := 5
def spent_dimes : ℕ := 2

theorem remaining_dimes : initial_dimes - spent_dimes = 3 := by
  sorry

end remaining_dimes_l2703_270330


namespace specific_can_stack_total_l2703_270361

/-- Represents a stack of cans forming an arithmetic sequence -/
structure CanStack where
  bottom_layer : ℕ
  difference : ℕ
  top_layer : ℕ

/-- Calculates the number of layers in the stack -/
def num_layers (stack : CanStack) : ℕ :=
  (stack.bottom_layer - stack.top_layer) / stack.difference + 1

/-- Calculates the total number of cans in the stack -/
def total_cans (stack : CanStack) : ℕ :=
  let n := num_layers stack
  (n * (stack.bottom_layer + stack.top_layer)) / 2

/-- Theorem stating that a specific can stack contains 172 cans -/
theorem specific_can_stack_total :
  let stack : CanStack := { bottom_layer := 35, difference := 4, top_layer := 1 }
  total_cans stack = 172 := by
  sorry

end specific_can_stack_total_l2703_270361


namespace real_roots_of_polynomial_l2703_270341

theorem real_roots_of_polynomial (x : ℝ) : 
  (x^4 - 4*x^3 + 5*x^2 - 2*x + 2 = 0) ↔ (x = 1 ∨ x = -1) :=
by sorry

end real_roots_of_polynomial_l2703_270341


namespace isosceles_triangle_from_touching_circle_l2703_270357

/-- A triangle with sides a, b, c and medians m_a, m_b, m_c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  m_a : ℝ
  m_b : ℝ
  m_c : ℝ

/-- A circle touching two sides and two medians of a triangle -/
structure TouchingCircle (T : Triangle) where
  touches_side_a : Bool
  touches_side_b : Bool
  touches_median_a : Bool
  touches_median_b : Bool

/-- 
If a circle touches two sides of a triangle and their corresponding medians,
then the triangle is isosceles.
-/
theorem isosceles_triangle_from_touching_circle (T : Triangle) 
  (C : TouchingCircle T) (h1 : C.touches_side_a) (h2 : C.touches_side_b) 
  (h3 : C.touches_median_a) (h4 : C.touches_median_b) : 
  T.a = T.b := by
  sorry


end isosceles_triangle_from_touching_circle_l2703_270357


namespace die_roll_probability_l2703_270326

/-- A fair six-sided die is rolled six times. -/
def num_rolls : ℕ := 6

/-- The probability of rolling a 5 or 6 on a fair six-sided die. -/
def prob_success : ℚ := 1/3

/-- The probability of not rolling a 5 or 6 on a fair six-sided die. -/
def prob_failure : ℚ := 1 - prob_success

/-- The number of successful outcomes we're interested in (at least 5 times). -/
def min_successes : ℕ := 5

/-- Calculates the binomial coefficient (n choose k). -/
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Calculates the probability of exactly k successes in n trials. -/
def prob_exactly (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * (1-p)^(n-k)

/-- The main theorem to prove. -/
theorem die_roll_probability : 
  prob_exactly num_rolls min_successes prob_success + 
  prob_exactly num_rolls num_rolls prob_success = 13/729 := by
  sorry

end die_roll_probability_l2703_270326


namespace trivia_game_points_per_round_l2703_270313

theorem trivia_game_points_per_round 
  (total_points : ℕ) 
  (num_rounds : ℕ) 
  (h1 : total_points = 78) 
  (h2 : num_rounds = 26) : 
  total_points / num_rounds = 3 := by
sorry

end trivia_game_points_per_round_l2703_270313


namespace f_properties_l2703_270339

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x) / Real.log 2

theorem f_properties :
  (∀ x ≠ 0, f (-x) = f x) ∧
  (∀ x y, 0 < x → x < y → f x < f y) :=
sorry

end f_properties_l2703_270339


namespace haley_marbles_l2703_270311

/-- The number of boys in Haley's class who love to play marbles -/
def num_boys : ℕ := 11

/-- The number of marbles Haley gives to each boy -/
def marbles_per_boy : ℕ := 9

/-- Theorem stating the total number of marbles Haley had -/
theorem haley_marbles : num_boys * marbles_per_boy = 99 := by
  sorry

end haley_marbles_l2703_270311


namespace peanut_seed_germination_l2703_270333

/-- The probability of at least k successes in n independent Bernoulli trials -/
def prob_at_least (n k : ℕ) (p : ℝ) : ℝ := sorry

/-- The probability of exactly k successes in n independent Bernoulli trials -/
def prob_exactly (n k : ℕ) (p : ℝ) : ℝ := sorry

theorem peanut_seed_germination :
  let n : ℕ := 4
  let k : ℕ := 2
  let p : ℝ := 4/5
  prob_at_least n k p = 608/625 := by sorry

end peanut_seed_germination_l2703_270333


namespace black_squares_in_58th_row_l2703_270379

/-- Represents a square color in the stair-step figure -/
inductive SquareColor
| White
| Black
| Red

/-- Represents a row in the stair-step figure -/
def StairRow := List SquareColor

/-- Generates a row of the stair-step figure -/
def generateRow (n : ℕ) : StairRow :=
  sorry

/-- Counts the number of black squares in a row -/
def countBlackSquares (row : StairRow) : ℕ :=
  sorry

/-- Main theorem: The number of black squares in the 58th row is 38 -/
theorem black_squares_in_58th_row :
  countBlackSquares (generateRow 58) = 38 := by
  sorry

end black_squares_in_58th_row_l2703_270379


namespace store_discount_income_increase_l2703_270393

theorem store_discount_income_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (discount_rate : ℝ) 
  (quantity_increase_rate : ℝ) 
  (h1 : discount_rate = 0.1) 
  (h2 : quantity_increase_rate = 0.2) : 
  let new_price := original_price * (1 - discount_rate)
  let new_quantity := original_quantity * (1 + quantity_increase_rate)
  let original_income := original_price * original_quantity
  let new_income := new_price * new_quantity
  (new_income - original_income) / original_income = 0.08 := by
sorry

end store_discount_income_increase_l2703_270393


namespace distance_between_points_l2703_270343

theorem distance_between_points : Real.sqrt 89 = Real.sqrt ((1 - (-4))^2 + (-3 - 5)^2) := by sorry

end distance_between_points_l2703_270343


namespace system_of_equations_solution_l2703_270329

theorem system_of_equations_solution :
  let x : ℚ := -133 / 57
  let y : ℚ := 64 / 19
  (3 * x - 4 * y = -7) ∧ (7 * x - 3 * y = 5) := by
  sorry

end system_of_equations_solution_l2703_270329


namespace second_quadrant_characterization_l2703_270388

/-- The set of points in the second quadrant of the Cartesian coordinate system -/
def second_quadrant : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0}

/-- Theorem stating that the second quadrant is equivalent to the set of points (x, y) where x < 0 and y > 0 -/
theorem second_quadrant_characterization :
  second_quadrant = {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0} := by
sorry

end second_quadrant_characterization_l2703_270388


namespace vector_perpendicular_condition_l2703_270309

theorem vector_perpendicular_condition (a b : ℝ × ℝ) (m : ℝ) : 
  ‖a‖ = Real.sqrt 3 →
  ‖b‖ = 2 →
  (a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖) = Real.cos (π / 6) →
  (a.1 - m * b.1) * a.1 + (a.2 - m * b.2) * a.2 = 0 →
  m = 1 := by
  sorry

end vector_perpendicular_condition_l2703_270309


namespace min_value_z_l2703_270376

theorem min_value_z (x y : ℝ) : x^2 + 2*y^2 + 6*x - 4*y + 22 ≥ 11 := by
  sorry

end min_value_z_l2703_270376


namespace sum_inequality_l2703_270335

theorem sum_inequality (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
  (h5 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h6 : a ≥ b ∧ a ≥ c ∧ a ≥ d)
  (h7 : d ≤ b ∧ d ≤ c)
  (h8 : a * d = b * c) :
  a + d > b + c := by
sorry

end sum_inequality_l2703_270335


namespace equation_solution_l2703_270328

theorem equation_solution : ∃! x : ℤ, 27474 + x + 1985 - 2047 = 31111 := by
  sorry

end equation_solution_l2703_270328


namespace equation_solution_l2703_270398

theorem equation_solution (x y : ℝ) : ∃ z : ℝ, 0.65 * x * y - z = 0.2 * 747.50 := by
  sorry

end equation_solution_l2703_270398


namespace library_visitors_average_l2703_270360

/-- Calculates the average number of visitors per day for a month in a library --/
def averageVisitorsPerDay (
  daysInMonth : ℕ)
  (sundayVisitors : ℕ)
  (regularDayVisitors : ℕ)
  (publicHolidays : ℕ)
  (specialEvents : ℕ) : ℚ :=
  let sundayCount := (daysInMonth + 6) / 7
  let regularDays := daysInMonth - sundayCount - publicHolidays - specialEvents
  let totalVisitors := 
    sundayCount * sundayVisitors +
    regularDays * regularDayVisitors +
    publicHolidays * (2 * regularDayVisitors) +
    specialEvents * (3 * regularDayVisitors)
  (totalVisitors : ℚ) / daysInMonth

theorem library_visitors_average :
  averageVisitorsPerDay 30 510 240 2 1 = 308 := by
  sorry

end library_visitors_average_l2703_270360


namespace li_cake_purchase_l2703_270395

theorem li_cake_purchase 
  (fruit_price : ℝ) 
  (chocolate_price : ℝ) 
  (total_spent : ℝ) 
  (average_price : ℝ)
  (h1 : fruit_price = 4.8)
  (h2 : chocolate_price = 6.6)
  (h3 : total_spent = 167.4)
  (h4 : average_price = 6.2) :
  ∃ (fruit_count chocolate_count : ℕ),
    fruit_count = 6 ∧ 
    chocolate_count = 21 ∧
    fruit_count * fruit_price + chocolate_count * chocolate_price = total_spent ∧
    (fruit_count + chocolate_count : ℝ) * average_price = total_spent :=
by sorry

end li_cake_purchase_l2703_270395


namespace centric_sequence_bound_and_extremal_points_l2703_270364

/-- The set of points (x, y) in R^2 such that x^2 + y^2 ≤ 1 -/
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 1}

/-- A sequence of points in R^2 -/
def Sequence := ℕ → ℝ × ℝ

/-- The circumcenter of a triangle formed by three points -/
noncomputable def circumcenter (p1 p2 p3 : ℝ × ℝ) : ℝ × ℝ := sorry

/-- A centric sequence satisfies the given properties -/
def IsCentric (A : Sequence) : Prop :=
  A 0 = (0, 0) ∧ A 1 = (1, 0) ∧
  ∀ n : ℕ, circumcenter (A n) (A (n+1)) (A (n+2)) ∈ C

theorem centric_sequence_bound_and_extremal_points :
  ∀ A : Sequence, IsCentric A →
    (A 2012).1^2 + (A 2012).2^2 ≤ 4048144 ∧
    (∀ x y : ℝ, x^2 + y^2 = 4048144 →
      (∃ A : Sequence, IsCentric A ∧ A 2012 = (x, y)) →
      ((x = -1006 ∧ y = 1006 * Real.sqrt 3) ∨
       (x = -1006 ∧ y = -1006 * Real.sqrt 3))) :=
by sorry

end centric_sequence_bound_and_extremal_points_l2703_270364


namespace spherical_coordinates_negated_z_l2703_270302

/-- Given a point with rectangular coordinates (x, y, z) and spherical coordinates
    (5, 3π/4, π/3), prove that the spherical coordinates of (x, y, -z) are (5, 3π/4, 2π/3) -/
theorem spherical_coordinates_negated_z 
  (x y z : ℝ) 
  (h1 : x = 5 * Real.sin (π/3) * Real.cos (3*π/4))
  (h2 : y = 5 * Real.sin (π/3) * Real.sin (3*π/4))
  (h3 : z = 5 * Real.cos (π/3)) :
  ∃ (ρ θ φ : ℝ), 
    ρ = 5 ∧ 
    θ = 3*π/4 ∧ 
    φ = 2*π/3 ∧
    x = ρ * Real.sin φ * Real.cos θ ∧
    y = ρ * Real.sin φ * Real.sin θ ∧
    -z = ρ * Real.cos φ ∧
    ρ > 0 ∧ 
    0 ≤ θ ∧ θ < 2*π ∧
    0 ≤ φ ∧ φ ≤ π := by
  sorry

end spherical_coordinates_negated_z_l2703_270302


namespace hare_tortoise_race_l2703_270374

theorem hare_tortoise_race (v : ℝ) (x : ℝ) (y : ℝ) (h_v_pos : v > 0) :
  v > 0 ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  x + y = 25 ∧ 
  x^2 + 5^2 = y^2 →
  y = 13 :=
by sorry

end hare_tortoise_race_l2703_270374


namespace complex_power_sum_l2703_270304

theorem complex_power_sum (w : ℂ) (h : w + 1 / w = 2 * Real.cos (5 * π / 180)) :
  w^1000 + 1 / w^1000 = -(Real.sqrt 5 + 1) / 2 := by
  sorry

end complex_power_sum_l2703_270304


namespace f_t_plus_one_l2703_270321

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 1

-- State the theorem
theorem f_t_plus_one (t : ℝ) : f (t + 1) = 3 * t + 2 := by
  sorry

end f_t_plus_one_l2703_270321


namespace total_employees_calculation_l2703_270301

/-- Represents the number of employees in different categories and calculates the total full-time equivalents -/
def calculate_total_employees (part_time : ℕ) (full_time : ℕ) (remote : ℕ) (temporary : ℕ) : ℕ :=
  let hours_per_fte : ℕ := 40
  let total_hours : ℕ := part_time + full_time * hours_per_fte + remote * hours_per_fte + temporary * hours_per_fte
  (total_hours + hours_per_fte / 2) / hours_per_fte

/-- Theorem stating that given the specified number of employees in each category, 
    the total number of full-time equivalent employees is 76,971 -/
theorem total_employees_calculation :
  calculate_total_employees 2041 63093 5230 8597 = 76971 := by
  sorry

end total_employees_calculation_l2703_270301


namespace books_in_bargain_bin_l2703_270368

theorem books_in_bargain_bin 
  (initial_books : ℕ) 
  (books_sold : ℕ) 
  (books_added : ℕ) 
  (h1 : initial_books ≥ books_sold) : 
  initial_books - books_sold + books_added = 
    initial_books + books_added - books_sold :=
by sorry

end books_in_bargain_bin_l2703_270368


namespace geometric_sequence_special_property_l2703_270394

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence, if a₂ · a₄ = 2a₃ - 1, then a₃ = 1 -/
theorem geometric_sequence_special_property (a : ℕ → ℝ) :
  geometric_sequence a → a 2 * a 4 = 2 * a 3 - 1 → a 3 = 1 := by
  sorry

end geometric_sequence_special_property_l2703_270394


namespace reflection_property_l2703_270399

/-- A reflection in R² --/
structure Reflection where
  /-- The reflection function --/
  reflect : ℝ × ℝ → ℝ × ℝ

/-- Given a reflection that maps (2, -3) to (-2, 9), it also maps (3, 1) to (-3, 1) --/
theorem reflection_property (r : Reflection) 
  (h1 : r.reflect (2, -3) = (-2, 9)) : 
  r.reflect (3, 1) = (-3, 1) := by
  sorry

end reflection_property_l2703_270399


namespace car_speed_problem_l2703_270382

/-- Given a car traveling for two hours with speeds x and 60 km/h, 
    prove that if the average speed is 102.5 km/h, then x must be 145 km/h. -/
theorem car_speed_problem (x : ℝ) :
  (x + 60) / 2 = 102.5 → x = 145 := by
  sorry

end car_speed_problem_l2703_270382


namespace cube_sum_equality_l2703_270356

theorem cube_sum_equality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (square_fourth_equality : a^2 + b^2 + c^2 = a^4 + b^4 + c^4) :
  a^3 + b^3 + c^3 = -3*a*b*(a+b) := by
  sorry

end cube_sum_equality_l2703_270356


namespace irrationality_of_sqrt_two_and_rationality_of_others_l2703_270385

theorem irrationality_of_sqrt_two_and_rationality_of_others : 
  (∃ (a b : ℤ), (a : ℝ) / (b : ℝ) = Real.sqrt 2) ∧ 
  (∃ (c d : ℤ), (c : ℝ) / (d : ℝ) = 3.14) ∧
  (∃ (e f : ℤ), (e : ℝ) / (f : ℝ) = -2) ∧
  (∃ (g h : ℤ), (g : ℝ) / (h : ℝ) = 1/3) ∧
  (¬∃ (i j : ℤ), (i : ℝ) / (j : ℝ) = Real.sqrt 2) :=
by sorry

end irrationality_of_sqrt_two_and_rationality_of_others_l2703_270385


namespace rhombus_perimeter_l2703_270358

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 30) (h2 : d2 = 16) :
  let s := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * s = 68 := by sorry

end rhombus_perimeter_l2703_270358


namespace descending_order_xy_xy2_x_l2703_270308

theorem descending_order_xy_xy2_x
  (x y : ℝ)
  (hx : x < 0)
  (hy : -1 < y ∧ y < 0) :
  xy > xy^2 ∧ xy^2 > x :=
by sorry

end descending_order_xy_xy2_x_l2703_270308


namespace count_acute_triangles_l2703_270317

/-- A triangle classification based on its angles -/
inductive TriangleType
  | Acute   : TriangleType
  | Right   : TriangleType
  | Obtuse  : TriangleType

/-- Represents a set of triangles -/
structure TriangleSet where
  total : Nat
  right : Nat
  obtuse : Nat

/-- Theorem: Given 7 triangles with 2 right angles and 3 obtuse angles, there are 2 acute triangles -/
theorem count_acute_triangles (ts : TriangleSet) :
  ts.total = 7 ∧ ts.right = 2 ∧ ts.obtuse = 3 →
  ts.total - ts.right - ts.obtuse = 2 := by
  sorry

#check count_acute_triangles

end count_acute_triangles_l2703_270317


namespace polygon_sides_l2703_270344

theorem polygon_sides (n : ℕ) (sum_interior_angles : ℝ) : sum_interior_angles = 1440 → n = 10 := by
  sorry

end polygon_sides_l2703_270344


namespace profit_share_ratio_l2703_270346

theorem profit_share_ratio (total_profit : ℚ) (difference : ℚ) 
  (h1 : total_profit = 1000)
  (h2 : difference = 200) :
  ∃ (x y : ℚ), x + y = total_profit ∧ x - y = difference ∧ y / total_profit = 2 / 5 :=
by sorry

end profit_share_ratio_l2703_270346


namespace pipe_ratio_l2703_270362

theorem pipe_ratio (total_length shorter_length : ℕ) 
  (h1 : total_length = 177)
  (h2 : shorter_length = 59)
  (h3 : shorter_length < total_length) :
  (total_length - shorter_length) / shorter_length = 2 := by
  sorry

end pipe_ratio_l2703_270362


namespace min_reciprocal_sum_l2703_270314

theorem min_reciprocal_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hsum : a + b + c = 2) :
  (1 / a + 1 / b + 1 / c) ≥ 9 / 2 :=
sorry

end min_reciprocal_sum_l2703_270314


namespace buccaneer_loot_sum_l2703_270334

def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (5 ^ i)) 0

theorem buccaneer_loot_sum : 
  let pearls := base5ToBase10 [1, 2, 3, 4]
  let silk := base5ToBase10 [1, 1, 1, 1]
  let spices := base5ToBase10 [1, 2, 2]
  let maps := base5ToBase10 [0, 1]
  pearls + silk + spices + maps = 808 := by sorry

end buccaneer_loot_sum_l2703_270334


namespace face_mask_selling_price_l2703_270350

theorem face_mask_selling_price
  (num_boxes : ℕ)
  (masks_per_box : ℕ)
  (total_cost : ℚ)
  (total_profit : ℚ)
  (h_num_boxes : num_boxes = 3)
  (h_masks_per_box : masks_per_box = 20)
  (h_total_cost : total_cost = 15)
  (h_total_profit : total_profit = 15) :
  (total_cost + total_profit) / (num_boxes * masks_per_box : ℚ) = 1/2 := by
sorry

end face_mask_selling_price_l2703_270350


namespace least_addition_for_divisibility_problem_solution_l2703_270345

theorem least_addition_for_divisibility (n : ℕ) (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : 
  ∃! x : ℕ, x < a * b ∧ (n + x) % a = 0 ∧ (n + x) % b = 0 ∧
  ∀ y : ℕ, y < x → ((n + y) % a ≠ 0 ∨ (n + y) % b ≠ 0) :=
by sorry

theorem problem_solution : 
  let n := 1056
  let a := 27
  let b := 31
  ∃! x : ℕ, x < a * b ∧ (n + x) % a = 0 ∧ (n + x) % b = 0 ∧
  ∀ y : ℕ, y < x → ((n + y) % a ≠ 0 ∨ (n + y) % b ≠ 0) ∧
  x = 618 :=
by sorry

end least_addition_for_divisibility_problem_solution_l2703_270345


namespace no_function_satisfies_condition_l2703_270303

theorem no_function_satisfies_condition : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2017 := by
  sorry

end no_function_satisfies_condition_l2703_270303


namespace lettuce_salads_per_plant_l2703_270396

theorem lettuce_salads_per_plant (total_salads : ℕ) (plants : ℕ) (loss_fraction : ℚ) : 
  total_salads = 12 →
  loss_fraction = 1/2 →
  plants = 8 →
  (total_salads / (1 - loss_fraction)) / plants = 3 := by
  sorry

end lettuce_salads_per_plant_l2703_270396


namespace store_holiday_customers_l2703_270375

/-- The number of customers entering a store during holiday season -/
def holiday_customers (normal_rate : ℕ) (hours : ℕ) : ℕ :=
  2 * normal_rate * hours

/-- Theorem: Given the conditions, the store will see 2800 customers in 8 hours during the holiday season -/
theorem store_holiday_customers :
  holiday_customers 175 8 = 2800 := by
  sorry

end store_holiday_customers_l2703_270375


namespace wheel_probability_l2703_270322

theorem wheel_probability (p_D p_E p_F p_G : ℚ) : 
  p_D = 1/4 → p_E = 1/3 → p_G = 1/6 → 
  p_D + p_E + p_F + p_G = 1 →
  p_F = 1/4 := by
sorry

end wheel_probability_l2703_270322


namespace no_solution_equation_l2703_270337

theorem no_solution_equation (x : ℝ) : 
  (4 * x - 1) / 6 - (5 * x - 2/3) / 10 + (9 - x/2) / 3 ≠ 101/20 := by
  sorry

end no_solution_equation_l2703_270337


namespace F_is_integer_exists_valid_s_and_t_l2703_270389

/-- Given a four-digit number, swap the thousands and tens digits, and the hundreds and units digits -/
def swap_digits (n : Nat) : Nat :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  1000 * c + 100 * d + 10 * a + b

/-- The "wholehearted number" function -/
def F (n : Nat) : Nat :=
  (n + swap_digits n) / 101

/-- Theorem: F(n) is an integer for any four-digit number n -/
theorem F_is_integer (n : Nat) (h : 1000 ≤ n ∧ n < 10000) : ∃ k : Nat, F n = k := by
  sorry

/-- Helper function to check if a number is divisible by 8 -/
def is_divisible_by_8 (n : Int) : Prop :=
  ∃ k : Int, n = 8 * k

/-- Function to generate s given a and b -/
def s (a b : Nat) : Nat :=
  3800 + 10 * a + b

/-- Function to generate t given a and b -/
def t (a b : Nat) : Nat :=
  1000 * b + 100 * a + 13

/-- Theorem: There exist values of a and b such that 3F(t) - F(s) is divisible by 8 -/
theorem exists_valid_s_and_t :
  ∃ (a b : Nat), 1 ≤ a ∧ a ≤ 5 ∧ 5 ≤ b ∧ b ≤ 9 ∧ is_divisible_by_8 (3 * (F (t a b)) - (F (s a b))) := by
  sorry

end F_is_integer_exists_valid_s_and_t_l2703_270389
