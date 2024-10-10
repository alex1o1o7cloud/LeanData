import Mathlib

namespace total_votes_l1562_156235

theorem total_votes (jerry_votes : ℕ) (vote_difference : ℕ) : 
  jerry_votes = 108375 →
  vote_difference = 20196 →
  jerry_votes + (jerry_votes - vote_difference) = 196554 :=
by
  sorry

end total_votes_l1562_156235


namespace octagon_diagonals_l1562_156215

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by
  sorry

end octagon_diagonals_l1562_156215


namespace remainder_sum_l1562_156201

theorem remainder_sum (c d : ℤ) 
  (hc : c % 60 = 47)
  (hd : d % 45 = 28) :
  (c + d) % 30 = 15 := by
  sorry

end remainder_sum_l1562_156201


namespace linear_function_decreasing_l1562_156216

def LinearFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x + b

theorem linear_function_decreasing (a b : ℝ) :
  (∀ x y, x < y → LinearFunction a b x > LinearFunction a b y) ↔ a < 0 :=
by sorry

end linear_function_decreasing_l1562_156216


namespace minutes_ratio_to_hour_l1562_156242

theorem minutes_ratio_to_hour (minutes_in_hour : ℕ) (ratio : ℚ) (result : ℕ) : 
  minutes_in_hour = 60 →
  ratio = 1/5 →
  result = minutes_in_hour * ratio →
  result = 12 := by sorry

end minutes_ratio_to_hour_l1562_156242


namespace power_of_two_equation_l1562_156205

theorem power_of_two_equation (m : ℤ) : 
  2^2000 - 2^1999 - 2^1998 + 2^1997 = m * 2^1997 → m = 3 := by
  sorry

end power_of_two_equation_l1562_156205


namespace complex_equation_solution_pure_imaginary_condition_l1562_156278

-- Problem 1
theorem complex_equation_solution (a b : ℝ) (h : (a + Complex.I) * (1 + Complex.I) = b * Complex.I) :
  a = 1 ∧ b = 2 := by sorry

-- Problem 2
theorem pure_imaginary_condition (m : ℝ) 
  (h : ∃ (k : ℝ), Complex.mk (m^2 + m - 2) (m^2 - 1) = Complex.I * k) :
  m = -2 := by sorry

end complex_equation_solution_pure_imaginary_condition_l1562_156278


namespace black_balls_count_l1562_156288

theorem black_balls_count (total_balls : ℕ) (red_balls : ℕ) (prob_red : ℚ) : 
  red_balls = 10 →
  prob_red = 2/7 →
  (red_balls : ℚ) / total_balls = prob_red →
  total_balls - red_balls = 25 := by
sorry

end black_balls_count_l1562_156288


namespace rectangle_max_area_l1562_156274

/-- A rectangle with whole number dimensions and perimeter 40 has a maximum area of 100 -/
theorem rectangle_max_area :
  ∀ l w : ℕ,
  l + w = 20 →
  ∀ l' w' : ℕ,
  l' + w' = 20 →
  l * w ≤ 100 ∧
  (∃ l'' w'' : ℕ, l'' + w'' = 20 ∧ l'' * w'' = 100) :=
by sorry

end rectangle_max_area_l1562_156274


namespace remainder_problem_l1562_156293

theorem remainder_problem (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 35 = 28 := by
  sorry

end remainder_problem_l1562_156293


namespace fraction_reciprocal_difference_l1562_156251

theorem fraction_reciprocal_difference (x : ℚ) : 
  0 < x → x < 1 → (1 / x - x = 9 / 20) → x = 4 / 5 := by
  sorry

end fraction_reciprocal_difference_l1562_156251


namespace roses_given_l1562_156249

/-- The number of students in the class -/
def total_students : ℕ := 28

/-- The number of different types of flowers -/
def flower_types : ℕ := 3

/-- The relationship between daffodils and roses -/
def rose_daffodil_ratio : ℕ := 4

/-- The relationship between tulips and roses -/
def tulip_rose_ratio : ℕ := 3

/-- The number of boys in the class -/
def num_boys : ℕ := 11

/-- The number of girls in the class -/
def num_girls : ℕ := 17

/-- The number of daffodils given -/
def num_daffodils : ℕ := 11

/-- The number of roses given -/
def num_roses : ℕ := 44

/-- The number of tulips given -/
def num_tulips : ℕ := 132

theorem roses_given :
  num_roses = 44 ∧
  total_students = num_boys + num_girls ∧
  num_roses = rose_daffodil_ratio * num_daffodils ∧
  num_tulips = tulip_rose_ratio * num_roses ∧
  num_boys * num_girls = num_daffodils + num_roses + num_tulips :=
by sorry

end roses_given_l1562_156249


namespace multiple_problem_l1562_156237

theorem multiple_problem (n : ℝ) (m : ℝ) (h1 : n = 25.0) (h2 : m * n = 3 * n - 25) : m = 2 := by
  sorry

end multiple_problem_l1562_156237


namespace minimum_dinner_cost_l1562_156283

/-- Represents an ingredient with its cost, quantity, and number of servings -/
structure Ingredient where
  name : String
  cost : ℚ
  quantity : ℚ
  servings : ℕ

/-- Calculates the minimum number of units needed to serve a given number of people -/
def minUnitsNeeded (servingsPerUnit : ℕ) (people : ℕ) : ℕ :=
  (people + servingsPerUnit - 1) / servingsPerUnit

/-- Calculates the total cost for an ingredient given the number of people to serve -/
def ingredientCost (i : Ingredient) (people : ℕ) : ℚ :=
  i.cost * (minUnitsNeeded i.servings people : ℚ)

/-- The list of ingredients for the dinner -/
def ingredients : List Ingredient := [
  ⟨"Pasta", 112/100, 500, 5⟩,
  ⟨"Meatballs", 524/100, 500, 4⟩,
  ⟨"Tomato sauce", 231/100, 400, 5⟩,
  ⟨"Tomatoes", 147/100, 400, 4⟩,
  ⟨"Lettuce", 97/100, 1, 6⟩,
  ⟨"Olives", 210/100, 1, 8⟩,
  ⟨"Cheese", 270/100, 1, 7⟩
]

/-- The number of people to serve -/
def numPeople : ℕ := 8

/-- The theorem stating the minimum total cost and cost per serving -/
theorem minimum_dinner_cost :
  let totalCost := (ingredients.map (ingredientCost · numPeople)).sum
  totalCost = 2972/100 ∧ totalCost / (numPeople : ℚ) = 3715/1000 := by
  sorry


end minimum_dinner_cost_l1562_156283


namespace another_beast_holds_all_candy_l1562_156213

/-- Represents the state of candy distribution among beasts -/
inductive CandyDistribution
  | initial (n : ℕ)  -- Initial distribution with Grogg having n candies
  | distribute (d : List ℕ)  -- List representing candy counts for each beast

/-- Represents a single step in the candy distribution process -/
def distributeStep (d : CandyDistribution) : CandyDistribution :=
  match d with
  | CandyDistribution.initial n => CandyDistribution.distribute (List.replicate n 1)
  | CandyDistribution.distribute (k :: rest) => 
      CandyDistribution.distribute (List.map (· + 1) (List.take k rest) ++ List.drop k rest)
  | _ => d

/-- Checks if all candy is held by a single beast (except Grogg) -/
def allCandyHeldBySingleBeast (d : CandyDistribution) : Bool :=
  match d with
  | CandyDistribution.distribute [n] => true
  | _ => false

/-- Main theorem: Another beast holds all candy iff n = 1 or n = 2 -/
theorem another_beast_holds_all_candy (n : ℕ) (h : n ≥ 1) :
  (∃ d : CandyDistribution, d = distributeStep (CandyDistribution.initial n) ∧ 
    allCandyHeldBySingleBeast d) ↔ n = 1 ∨ n = 2 :=
  sorry

end another_beast_holds_all_candy_l1562_156213


namespace square_root_meaningful_l1562_156230

theorem square_root_meaningful (x : ℝ) : 
  x ≥ 5 → (x = 6 ∧ x ≠ -5 ∧ x ≠ 0 ∧ x ≠ 4) :=
by sorry

end square_root_meaningful_l1562_156230


namespace max_points_without_equilateral_triangle_l1562_156275

/-- Represents a point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents the set of 10 points: vertices, centroid, and trisection points -/
def TrianglePoints (t : EquilateralTriangle) : Finset Point :=
  sorry

/-- Checks if three points form an equilateral triangle -/
def isEquilateral (p1 p2 p3 : Point) : Prop :=
  sorry

/-- The main theorem -/
theorem max_points_without_equilateral_triangle (t : EquilateralTriangle) :
  ∃ (s : Finset Point), s ⊆ TrianglePoints t ∧ s.card = 6 ∧
  ∀ (p1 p2 p3 : Point), p1 ∈ s → p2 ∈ s → p3 ∈ s → ¬(isEquilateral p1 p2 p3) ∧
  ∀ (s' : Finset Point), s' ⊆ TrianglePoints t →
    (∀ (p1 p2 p3 : Point), p1 ∈ s' → p2 ∈ s' → p3 ∈ s' → ¬(isEquilateral p1 p2 p3)) →
    s'.card ≤ 6 :=
  sorry

end max_points_without_equilateral_triangle_l1562_156275


namespace quadratic_function_properties_l1562_156257

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  b : ℝ
  c : ℝ
  f : ℝ → ℝ
  f_def : ∀ x, f x = x^2 + b*x + c
  f_sin_nonneg : ∀ α, f (Real.sin α) ≥ 0
  f_cos_nonpos : ∀ β, f (2 + Real.cos β) ≤ 0

theorem quadratic_function_properties (qf : QuadraticFunction) :
  qf.f 1 = 0 ∧ 
  qf.c ≥ 3 ∧
  (∀ α, qf.f (Real.sin α) ≤ 8 → qf.f = fun x ↦ x^2 - 4*x + 3) :=
by sorry

end quadratic_function_properties_l1562_156257


namespace knights_and_liars_l1562_156287

/-- Represents the two types of inhabitants in the country -/
inductive Inhabitant
  | Knight
  | Liar

/-- The statement made by A -/
def statement (a b : Inhabitant) : Prop :=
  a = Inhabitant.Liar ∧ b ≠ Inhabitant.Liar

/-- A function that determines if a given statement is true based on the speaker's type -/
def isTrueStatement (speaker : Inhabitant) (stmt : Prop) : Prop :=
  (speaker = Inhabitant.Knight ∧ stmt) ∨ (speaker = Inhabitant.Liar ∧ ¬stmt)

theorem knights_and_liars (a b : Inhabitant) :
  isTrueStatement a (statement a b) →
  a = Inhabitant.Liar ∧ b = Inhabitant.Liar :=
by sorry

end knights_and_liars_l1562_156287


namespace product_19_reciprocal_squares_sum_l1562_156229

theorem product_19_reciprocal_squares_sum :
  ∀ a b : ℕ+, 
  (a * b : ℕ+) = 19 → 
  (1 : ℚ) / (a * a : ℚ) + (1 : ℚ) / (b * b : ℚ) = 362 / 361 := by
  sorry

end product_19_reciprocal_squares_sum_l1562_156229


namespace cow_chicken_problem_l1562_156292

theorem cow_chicken_problem (cows chickens : ℕ) : 
  (4 * cows + 2 * chickens = 2 * (cows + chickens) + 10) → cows = 5 := by
  sorry

end cow_chicken_problem_l1562_156292


namespace vacation_days_l1562_156264

theorem vacation_days (rainy_days clear_mornings clear_afternoons : ℕ) 
  (h1 : rainy_days = 13)
  (h2 : clear_mornings = 11)
  (h3 : clear_afternoons = 12)
  (h4 : rainy_days = clear_mornings + clear_afternoons) :
  clear_mornings + clear_afternoons = 23 := by
  sorry

end vacation_days_l1562_156264


namespace f_composition_equals_one_third_l1562_156291

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 4
  else 3^x

-- State the theorem
theorem f_composition_equals_one_third :
  f (f (1/4)) = 1/3 := by
  sorry

end f_composition_equals_one_third_l1562_156291


namespace ratio_calculation_l1562_156265

theorem ratio_calculation (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (5 * A + 3 * B) / (3 * C - 2 * A) = 7 / 3 := by
  sorry

end ratio_calculation_l1562_156265


namespace equivalent_form_l1562_156262

theorem equivalent_form :
  (2 + 5) * (2^2 + 5^2) * (2^4 + 5^4) * (2^8 + 5^8) * 
  (2^16 + 5^16) * (2^32 + 5^32) * (2^64 + 5^64) = 5^128 - 2^128 := by
  sorry

end equivalent_form_l1562_156262


namespace min_bushes_for_zucchinis_l1562_156267

/-- Represents the yield of blueberry containers per bush -/
def containers_per_bush : ℕ := 10

/-- Represents the number of containers needed to trade for one zucchini -/
def containers_per_zucchini : ℕ := 3

/-- Represents the target number of zucchinis -/
def target_zucchinis : ℕ := 72

/-- 
Calculates the minimum number of bushes needed to obtain at least the target number of zucchinis.
-/
def min_bushes_needed : ℕ :=
  ((target_zucchinis * containers_per_zucchini + containers_per_bush - 1) / containers_per_bush : ℕ)

theorem min_bushes_for_zucchinis :
  min_bushes_needed = 22 ∧
  min_bushes_needed * containers_per_bush ≥ target_zucchinis * containers_per_zucchini ∧
  (min_bushes_needed - 1) * containers_per_bush < target_zucchinis * containers_per_zucchini :=
by sorry

end min_bushes_for_zucchinis_l1562_156267


namespace divisibility_problem_l1562_156268

theorem divisibility_problem (a b c d : ℤ) 
  (h : (a^4 + b^4 + c^4 + d^4) % 5 = 0) : 
  625 ∣ (a * b * c * d) := by
  sorry

end divisibility_problem_l1562_156268


namespace katies_cupcakes_l1562_156261

theorem katies_cupcakes (cupcakes cookies left_over sold : ℕ) :
  cookies = 5 →
  left_over = 8 →
  sold = 4 →
  cupcakes + cookies = left_over + sold →
  cupcakes = 7 := by
sorry

end katies_cupcakes_l1562_156261


namespace product_of_sums_l1562_156280

theorem product_of_sums (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 2) :
  (x + 2) * (y + 2) = 16 := by sorry

end product_of_sums_l1562_156280


namespace closest_integer_to_cube_root_l1562_156224

theorem closest_integer_to_cube_root : 
  ∃ (n : ℤ), n = 10 ∧ ∀ (m : ℤ), |m - (7^3 + 9^3)^(1/3)| ≥ |n - (7^3 + 9^3)^(1/3)| :=
by sorry

end closest_integer_to_cube_root_l1562_156224


namespace smallest_sum_of_reciprocals_l1562_156228

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (x : ℚ)⁻¹ + (y : ℚ)⁻¹ = 15⁻¹) :
  (x : ℕ) + y ≥ 64 :=
sorry

end smallest_sum_of_reciprocals_l1562_156228


namespace color_guard_row_length_l1562_156209

theorem color_guard_row_length 
  (num_students : ℕ) 
  (student_space : ℝ) 
  (gap_space : ℝ) 
  (h1 : num_students = 40)
  (h2 : student_space = 0.4)
  (h3 : gap_space = 0.5) : 
  (num_students : ℝ) * student_space + (num_students - 1 : ℝ) * gap_space = 35.5 :=
by sorry

end color_guard_row_length_l1562_156209


namespace calculate_speed_l1562_156294

/-- Given two people moving in opposite directions, calculate the speed of one person given the speed of the other and their total distance after a certain time. -/
theorem calculate_speed (riya_speed priya_speed : ℝ) (time : ℝ) (total_distance : ℝ) : 
  riya_speed = 24 →
  time = 0.75 →
  total_distance = 44.25 →
  priya_speed * time + riya_speed * time = total_distance →
  priya_speed = 35 := by
sorry

end calculate_speed_l1562_156294


namespace square_difference_ratio_l1562_156234

theorem square_difference_ratio : 
  (1630^2 - 1623^2) / (1640^2 - 1613^2) = 7/27 := by
  sorry

end square_difference_ratio_l1562_156234


namespace other_root_of_quadratic_l1562_156239

theorem other_root_of_quadratic (a c : ℝ) (h : a ≠ 0) :
  (∃ x, 4 * a * x^2 - 2 * a * x + c = 0 ∧ x = 0) →
  (∃ y, 4 * a * y^2 - 2 * a * y + c = 0 ∧ y = 1/2) :=
by sorry

end other_root_of_quadratic_l1562_156239


namespace circle_center_in_second_quadrant_l1562_156217

theorem circle_center_in_second_quadrant (a : ℝ) (h : a > 12) :
  let center := (-(a/2), a)
  (center.1 < 0 ∧ center.2 > 0) ∧
  (∀ x y : ℝ, x^2 + y^2 + a*x - 2*a*y + a^2 + 3*a = 0 ↔ 
    (x - center.1)^2 + (y - center.2)^2 = (a^2/4 - 3*a)) :=
by sorry

end circle_center_in_second_quadrant_l1562_156217


namespace equation_solution_range_l1562_156211

theorem equation_solution_range (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ k / (2 * x - 4) - 1 = x / (x - 2)) → 
  (k > -4 ∧ k ≠ 4) :=
by sorry

end equation_solution_range_l1562_156211


namespace even_function_property_l1562_156248

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem even_function_property (f : ℝ → ℝ) 
  (h_even : EvenFunction f) 
  (h_positive : ∀ x > 0, f x = 10^x) : 
  ∀ x < 0, f x = (1/10)^x := by
sorry

end even_function_property_l1562_156248


namespace unique_n_for_integer_Sn_l1562_156240

theorem unique_n_for_integer_Sn : ∃! (n : ℕ+), ∃ (m : ℕ), 
  n.val > 0 ∧ m^2 = 17^2 + n.val^2 ∧ 
  ∀ (k : ℕ+), k ≠ n → ¬∃ (l : ℕ), l^2 = 17^2 + k.val^2 := by
  sorry

end unique_n_for_integer_Sn_l1562_156240


namespace sales_third_month_l1562_156225

def sales_problem (m1 m2 m4 m5 m6 avg : ℚ) : ℚ :=
  let total_sales := avg * 6
  let known_sales := m1 + m2 + m4 + m5 + m6
  total_sales - known_sales

theorem sales_third_month
  (m1 m2 m4 m5 m6 avg : ℚ)
  (h_avg : avg = 6600)
  (h_m1 : m1 = 6435)
  (h_m2 : m2 = 6927)
  (h_m4 : m4 = 7230)
  (h_m5 : m5 = 6562)
  (h_m6 : m6 = 5591) :
  sales_problem m1 m2 m4 m5 m6 avg = 14085 := by
  sorry

#eval sales_problem 6435 6927 7230 6562 5591 6600

end sales_third_month_l1562_156225


namespace investment_total_calculation_l1562_156276

/-- Represents an investment split between two interest rates -/
structure Investment where
  total : ℝ
  rate1 : ℝ
  rate2 : ℝ
  amount1 : ℝ

/-- Calculates the total interest earned from an investment -/
def totalInterest (inv : Investment) : ℝ :=
  inv.rate1 * inv.amount1 + inv.rate2 * (inv.total - inv.amount1)

theorem investment_total_calculation (inv : Investment) 
  (h1 : inv.rate1 = 0.07)
  (h2 : inv.rate2 = 0.09)
  (h3 : inv.amount1 = 5500)
  (h4 : totalInterest inv = 970) :
  inv.total = 12000 := by
sorry

end investment_total_calculation_l1562_156276


namespace average_age_combined_l1562_156289

theorem average_age_combined (num_students : Nat) (num_parents : Nat) 
  (avg_age_students : ℝ) (avg_age_parents : ℝ) :
  num_students = 45 →
  num_parents = 60 →
  avg_age_students = 12 →
  avg_age_parents = 40 →
  (num_students * avg_age_students + num_parents * avg_age_parents) / (num_students + num_parents : ℝ) = 28 := by
  sorry

end average_age_combined_l1562_156289


namespace coordinates_of_q_l1562_156263

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A triangle in 2D space -/
structure Triangle where
  p : Point2D
  q : Point2D
  r : Point2D

/-- Predicate for a right-angled triangle at Q -/
def isRightAngledAtQ (t : Triangle) : Prop :=
  -- Definition of right angle at Q (placeholder)
  True

/-- Predicate for a horizontal line segment -/
def isHorizontal (p1 p2 : Point2D) : Prop :=
  p1.y = p2.y

/-- Predicate for a vertical line segment -/
def isVertical (p1 p2 : Point2D) : Prop :=
  p1.x = p2.x

theorem coordinates_of_q (t : Triangle) :
  isRightAngledAtQ t →
  isHorizontal t.p t.q →
  isVertical t.q t.r →
  t.p = Point2D.mk 1 1 →
  t.r = Point2D.mk 5 3 →
  t.q = Point2D.mk 5 1 := by
  sorry

end coordinates_of_q_l1562_156263


namespace PQ_length_is_correct_l1562_156259

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the lengths of the sides
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ :=
  (7, 8, 9)

-- Define the altitude AH
def altitude (t : Triangle) : ℝ × ℝ → ℝ × ℝ := sorry

-- Define the angle bisectors BD and CE
def angle_bisector_BD (t : Triangle) : ℝ × ℝ → ℝ × ℝ := sorry
def angle_bisector_CE (t : Triangle) : ℝ × ℝ → ℝ × ℝ := sorry

-- Define the intersection points P and Q
def P (t : Triangle) : ℝ × ℝ := sorry
def Q (t : Triangle) : ℝ × ℝ := sorry

-- Define the length of PQ
def PQ_length (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem PQ_length_is_correct (t : Triangle) :
  PQ_length t = (8 / 15) * Real.sqrt 5 := by sorry

end PQ_length_is_correct_l1562_156259


namespace largest_712_triple_l1562_156233

/-- Converts a number from base 7 to base 10 --/
def base7ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 12 --/
def decimalToBase12 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 7-12 triple --/
def is712Triple (n : ℕ) : Prop :=
  decimalToBase12 n = 3 * base7ToDecimal n

/-- The largest 7-12 triple --/
def largestTriple : ℕ := 450

theorem largest_712_triple :
  is712Triple largestTriple ∧
  ∀ n : ℕ, n > largestTriple → ¬is712Triple n := by sorry

end largest_712_triple_l1562_156233


namespace y_plus_z_value_l1562_156271

theorem y_plus_z_value (x y z : ℕ) (hx : x = 4) (hy : y = 3 * x) (hz : z = 2 * y) : 
  y + z = 36 := by
  sorry

end y_plus_z_value_l1562_156271


namespace sqrt_calculation_l1562_156206

theorem sqrt_calculation : 
  Real.sqrt 3 * Real.sqrt 12 - 2 * Real.sqrt 6 / Real.sqrt 3 + Real.sqrt 32 + (Real.sqrt 2)^2 = 8 + 2 * Real.sqrt 2 := by
  sorry

end sqrt_calculation_l1562_156206


namespace point_coordinates_l1562_156223

/-- Given a point P with coordinates (2m+4, m-1), prove that P has coordinates (-6, -6) 
    under the condition that it lies on the y-axis or its distance from the y-axis is 6, 
    and it lies in the third quadrant and is equidistant from both coordinate axes. -/
theorem point_coordinates (m : ℝ) : 
  (((2*m + 4 = 0) ∨ (|2*m + 4| = 6)) ∧ 
   (2*m + 4 < 0) ∧ (m - 1 < 0) ∧ 
   (|2*m + 4| = |m - 1|)) → 
  (2*m + 4 = -6 ∧ m - 1 = -6) := by
  sorry

end point_coordinates_l1562_156223


namespace quadratic_condition_l1562_156295

def is_quadratic (m : ℝ) : Prop :=
  (|m| = 2) ∧ (m - 2 ≠ 0)

theorem quadratic_condition (m : ℝ) :
  is_quadratic m ↔ m = -2 := by sorry

end quadratic_condition_l1562_156295


namespace smallest_b_value_l1562_156236

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 8) 
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 16) :
  ∀ c : ℕ+, c.val < b.val → 
    ¬(∃ d : ℕ+, d.val - c.val = 8 ∧ 
      Nat.gcd ((d.val^3 + c.val^3) / (d.val + c.val)) (d.val * c.val) = 16) :=
by sorry

end smallest_b_value_l1562_156236


namespace equation_solution_l1562_156243

theorem equation_solution (x y : ℝ) (hx : x ≠ 0) (hxy : x + y ≠ 0) :
  (x + y) / x = 2 * y / (x + y) + 1 ↔ x = y ∨ x = -3 * y :=
by sorry

end equation_solution_l1562_156243


namespace point_on_line_l1562_156290

theorem point_on_line (m n k : ℝ) : 
  (m = 2 * n + 5) ∧ (m + 4 = 2 * (n + k) + 5) → k = 2 := by
  sorry

end point_on_line_l1562_156290


namespace square_of_binomial_b_value_l1562_156231

/-- If 9x^2 + 27x + b is the square of a binomial, then b = 81/4 -/
theorem square_of_binomial_b_value (b : ℝ) : 
  (∃ (c : ℝ), ∀ (x : ℝ), 9*x^2 + 27*x + b = (3*x + c)^2) → 
  b = 81/4 := by
sorry

end square_of_binomial_b_value_l1562_156231


namespace parallelogram_intersection_theorem_l1562_156260

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram ABCD with point F -/
structure Parallelogram :=
  (A B C D F : Point)
  (isParallelogram : sorry) -- Condition that ABCD is a parallelogram
  (F_on_AD_extension : sorry) -- Condition that F is on the extension of AD

/-- Represents the intersection points E and G -/
structure Intersections (p : Parallelogram) :=
  (E : Point)
  (G : Point)
  (E_on_AC_BF : sorry) -- Condition that E is on both AC and BF
  (G_on_DC_BF : sorry) -- Condition that G is on both DC and BF

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- The main theorem -/
theorem parallelogram_intersection_theorem (p : Parallelogram) (i : Intersections p) :
  distance i.E p.F = 40 → distance i.G p.F = 18 → distance p.B i.E = 20 * Real.sqrt 2 := by
  sorry


end parallelogram_intersection_theorem_l1562_156260


namespace sum_in_base5_l1562_156250

/-- Represents a number in base 5 --/
def Base5 : Type := ℕ

/-- Converts a base 5 number to its decimal representation --/
def to_decimal (n : Base5) : ℕ := sorry

/-- Converts a decimal number to its base 5 representation --/
def to_base5 (n : ℕ) : Base5 := sorry

/-- Addition operation for base 5 numbers --/
def base5_add (a b : Base5) : Base5 := to_base5 (to_decimal a + to_decimal b)

theorem sum_in_base5 :
  let a : Base5 := to_base5 231
  let b : Base5 := to_base5 414
  let c : Base5 := to_base5 123
  let result : Base5 := to_base5 1323
  base5_add (base5_add a b) c = result := by sorry

end sum_in_base5_l1562_156250


namespace larry_basketball_shots_l1562_156258

theorem larry_basketball_shots 
  (initial_shots : ℕ) 
  (initial_success_rate : ℚ) 
  (additional_shots : ℕ) 
  (new_success_rate : ℚ) 
  (h1 : initial_shots = 30)
  (h2 : initial_success_rate = 3/5)
  (h3 : additional_shots = 10)
  (h4 : new_success_rate = 13/20) :
  (new_success_rate * (initial_shots + additional_shots) - initial_success_rate * initial_shots : ℚ) = 8 := by
sorry

end larry_basketball_shots_l1562_156258


namespace power_of_power_at_three_l1562_156285

theorem power_of_power_at_three : (3^(3^2))^(3^3) = 3^243 := by
  sorry

end power_of_power_at_three_l1562_156285


namespace no_integer_roots_l1562_156269

theorem no_integer_roots (a b : ℤ) : ¬ ∃ x : ℤ, 2 * a * b * x^4 - a^2 * x^2 - b^2 - 1 = 0 := by
  sorry

end no_integer_roots_l1562_156269


namespace rhombus_perimeter_l1562_156208

/-- Given a rhombus whose diagonal lengths are the roots of x^2 - 14x + 48 = 0, its perimeter is 20 -/
theorem rhombus_perimeter (x₁ x₂ : ℝ) : 
  x₁^2 - 14*x₁ + 48 = 0 → 
  x₂^2 - 14*x₂ + 48 = 0 → 
  x₁ ≠ x₂ →
  let s := Real.sqrt ((x₁^2 + x₂^2) / 4)
  4 * s = 20 := by
  sorry

end rhombus_perimeter_l1562_156208


namespace opposite_reciprocal_problem_l1562_156212

theorem opposite_reciprocal_problem (a b c d m : ℤ) : 
  (a = -b) →  -- a and b are opposite numbers
  (c * d = 1) →  -- c and d are reciprocals
  (m = -1) →  -- m is the largest negative integer
  c * d - a - b + m ^ 2022 = 2 := by
sorry

end opposite_reciprocal_problem_l1562_156212


namespace prob_even_sum_first_15_primes_l1562_156299

/-- The number of prime numbers we're considering -/
def n : ℕ := 15

/-- The number of prime numbers we're selecting -/
def k : ℕ := 5

/-- The number of odd primes among the first n primes -/
def odd_primes : ℕ := n - 1

/-- The probability of selecting k primes from n primes such that their sum is even -/
def prob_even_sum (n k odd_primes : ℕ) : ℚ :=
  (Nat.choose odd_primes k + Nat.choose odd_primes (k - 3)) / Nat.choose n k

theorem prob_even_sum_first_15_primes : 
  prob_even_sum n k odd_primes = 2093 / 3003 :=
sorry

end prob_even_sum_first_15_primes_l1562_156299


namespace abs_S_eq_1024_l1562_156238

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the expression S
def S : ℂ := (1 + i)^18 - (1 - i)^18

-- Theorem statement
theorem abs_S_eq_1024 : Complex.abs S = 1024 := by
  sorry

end abs_S_eq_1024_l1562_156238


namespace lucas_income_l1562_156279

/-- Represents the tax structure and Lucas's income --/
structure TaxSystem where
  p : ℝ  -- Base tax rate as a decimal
  income : ℝ  -- Lucas's annual income
  taxPaid : ℝ  -- Total tax paid by Lucas

/-- The tax system satisfies the given conditions --/
def validTaxSystem (ts : TaxSystem) : Prop :=
  ts.taxPaid = (0.01 * ts.p * 35000 + 0.01 * (ts.p + 4) * (ts.income - 35000))
  ∧ ts.taxPaid = 0.01 * (ts.p + 0.5) * ts.income
  ∧ ts.income ≥ 35000

/-- Theorem stating that Lucas's income is $40000 --/
theorem lucas_income (ts : TaxSystem) (h : validTaxSystem ts) : ts.income = 40000 := by
  sorry

end lucas_income_l1562_156279


namespace fifa_world_cup_players_l1562_156207

/-- The number of teams in the 17th FIFA World Cup -/
def num_teams : ℕ := 35

/-- The number of players in each team -/
def players_per_team : ℕ := 23

/-- The total number of players in the 17th FIFA World Cup -/
def total_players : ℕ := num_teams * players_per_team

theorem fifa_world_cup_players :
  total_players = 805 := by sorry

end fifa_world_cup_players_l1562_156207


namespace circle_tangent_to_line_l1562_156220

/-- A circle is tangent to a line if and only if the distance from the center of the circle to the line is equal to the radius of the circle. -/
theorem circle_tangent_to_line (b : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + 2*x + y^2 - 4*y + 3 = 0}
  let line := {(x, y) : ℝ × ℝ | x + y + b = 0}
  let center := (-1, 2)
  let radius := Real.sqrt 2
  (∀ p ∈ circle, p ∈ line → (∀ q ∈ circle, q = p ∨ q ∉ line)) → 
  b = 1 :=
by sorry

end circle_tangent_to_line_l1562_156220


namespace solution_system1_solution_system2_l1562_156210

-- Define the first system of equations
def system1 (x y : ℝ) : Prop :=
  2 * x + 3 * y = 8 ∧ x = y - 1

-- Define the second system of equations
def system2 (x y : ℝ) : Prop :=
  2 * x - y = -1 ∧ x + 3 * y = 17

-- Theorem for the first system
theorem solution_system1 : ∃ x y : ℝ, system1 x y ∧ x = 1 ∧ y = 2 := by
  sorry

-- Theorem for the second system
theorem solution_system2 : ∃ x y : ℝ, system2 x y ∧ x = 2 ∧ y = 5 := by
  sorry

end solution_system1_solution_system2_l1562_156210


namespace jungkook_points_l1562_156282

/-- Calculates the total points earned by Jungkook in a math test. -/
theorem jungkook_points (total_problems : ℕ) (correct_two_point : ℕ) (correct_one_point : ℕ) 
  (h1 : total_problems = 15)
  (h2 : correct_two_point = 8)
  (h3 : correct_one_point = 2) :
  correct_two_point * 2 + correct_one_point = 18 := by
  sorry

#check jungkook_points

end jungkook_points_l1562_156282


namespace complex_to_exponential_form_l1562_156281

theorem complex_to_exponential_form (z : ℂ) :
  z = 2 - I →
  Real.arctan (1 / 2) = Real.arctan (Complex.abs z / Complex.im z) :=
by sorry

end complex_to_exponential_form_l1562_156281


namespace solution_exists_l1562_156266

theorem solution_exists (x : ℝ) : 3 ∈ ({x + 2, x^2 + 2*x} : Set ℝ) → x = -3 := by
  sorry

end solution_exists_l1562_156266


namespace borrowed_amount_correct_l1562_156221

/-- The amount of money borrowed, in Rupees -/
def borrowed_amount : ℝ := 5000

/-- The interest rate for borrowing, as a decimal -/
def borrow_rate : ℝ := 0.04

/-- The interest rate for lending, as a decimal -/
def lend_rate : ℝ := 0.07

/-- The duration of the loan in years -/
def duration : ℝ := 2

/-- The yearly gain from the transaction, in Rupees -/
def yearly_gain : ℝ := 150

/-- Theorem stating that the borrowed amount is correct given the conditions -/
theorem borrowed_amount_correct :
  borrowed_amount * borrow_rate * duration = 
  borrowed_amount * lend_rate * duration - yearly_gain * duration := by
  sorry

#check borrowed_amount_correct

end borrowed_amount_correct_l1562_156221


namespace initial_courses_is_three_l1562_156247

/-- Represents the wall construction problem -/
def WallProblem (initial_courses : ℕ) : Prop :=
  let bricks_per_course : ℕ := 400
  let added_courses : ℕ := 2
  let removed_bricks : ℕ := 200
  let total_bricks : ℕ := 1800
  (initial_courses * bricks_per_course) + (added_courses * bricks_per_course) - removed_bricks = total_bricks

/-- Theorem stating that the initial number of courses is 3 -/
theorem initial_courses_is_three : WallProblem 3 := by
  sorry

end initial_courses_is_three_l1562_156247


namespace x_over_y_value_l1562_156218

theorem x_over_y_value (x y : ℝ) 
  (h1 : 3 < (2*x - y)/(x + 2*y)) 
  (h2 : (2*x - y)/(x + 2*y) < 7) 
  (h3 : ∃ (n : ℤ), x/y = n) : 
  x/y = -4 := by sorry

end x_over_y_value_l1562_156218


namespace deriv_sin_plus_cos_at_pi_fourth_l1562_156227

/-- The derivative of sin(x) + cos(x) at π/4 is 0 -/
theorem deriv_sin_plus_cos_at_pi_fourth (f : ℝ → ℝ) (h : f = λ x => Real.sin x + Real.cos x) :
  deriv f (π / 4) = 0 := by
  sorry

end deriv_sin_plus_cos_at_pi_fourth_l1562_156227


namespace binomial_five_one_l1562_156244

theorem binomial_five_one : (5 : ℕ).choose 1 = 5 := by sorry

end binomial_five_one_l1562_156244


namespace largest_product_of_three_primes_digit_sum_l1562_156253

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem largest_product_of_three_primes_digit_sum :
  ∃ (n d e : ℕ),
    is_prime d ∧ d < 20 ∧
    is_prime e ∧ e < 20 ∧
    is_prime (e^2 + 10*d) ∧
    n = d * e * (e^2 + 10*d) ∧
    (∀ (n' d' e' : ℕ),
      is_prime d' ∧ d' < 20 ∧
      is_prime e' ∧ e' < 20 ∧
      is_prime (e'^2 + 10*d') ∧
      n' = d' * e' * (e'^2 + 10*d') →
      n' ≤ n) ∧
    sum_of_digits n = 16 :=
by sorry

end largest_product_of_three_primes_digit_sum_l1562_156253


namespace circle_equation_l1562_156204

theorem circle_equation (x y : ℝ) :
  (x^2 + 8*x + y^2 + 4*y - 36 = 0) ↔
  ((x + 4)^2 + (y + 2)^2 = 4^2) :=
sorry

end circle_equation_l1562_156204


namespace empty_proper_subset_singleton_zero_l1562_156214

theorem empty_proper_subset_singleton_zero :
  ∅ ⊂ ({0} : Set ℕ) :=
sorry

end empty_proper_subset_singleton_zero_l1562_156214


namespace tomato_production_l1562_156270

/-- The number of tomatoes produced by the first plant -/
def plant1_tomatoes : ℕ := 24

/-- The number of tomatoes produced by the second plant -/
def plant2_tomatoes : ℕ := plant1_tomatoes / 2 + 5

/-- The number of tomatoes produced by the third plant -/
def plant3_tomatoes : ℕ := plant2_tomatoes + 2

/-- The total number of tomatoes produced by all three plants -/
def total_tomatoes : ℕ := plant1_tomatoes + plant2_tomatoes + plant3_tomatoes

theorem tomato_production : total_tomatoes = 60 := by
  sorry

end tomato_production_l1562_156270


namespace operation_probability_l1562_156232

/-- An operation that randomly changes a positive integer to a smaller nonnegative integer -/
def operation (n : ℕ+) : ℕ := sorry

/-- The probability of choosing any specific smaller number during the operation -/
def transition_prob (n k : ℕ) : ℝ := sorry

/-- The probability of encountering specific numbers during the operation process -/
def encounter_prob (start : ℕ+) (targets : List ℕ) : ℝ := sorry

theorem operation_probability :
  encounter_prob 2019 [10, 100, 1000] = 1 / 2019000000 := by sorry

end operation_probability_l1562_156232


namespace no_double_application_plus_one_l1562_156254

theorem no_double_application_plus_one : 
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 1 := by
sorry

end no_double_application_plus_one_l1562_156254


namespace min_value_of_z_l1562_156277

/-- The objective function to be minimized -/
def z (x y : ℝ) : ℝ := y - 2 * x

/-- The feasible region defined by the given constraints -/
def feasible_region (x y : ℝ) : Prop :=
  3 * x + y - 6 ≥ 0 ∧ x - y - 2 ≤ 0 ∧ y - 3 ≤ 0

/-- Theorem stating that the minimum value of z in the feasible region is -7 -/
theorem min_value_of_z :
  ∃ (x y : ℝ), feasible_region x y ∧
  ∀ (x' y' : ℝ), feasible_region x' y' → z x' y' ≥ z x y ∧
  z x y = -7 :=
sorry

end min_value_of_z_l1562_156277


namespace max_books_on_shelf_l1562_156245

theorem max_books_on_shelf (n : ℕ) (s₁ s₂ S : ℕ) : 
  (S + s₁ ≥ (n - 2) / 2) →
  (S + s₂ < (n - 2) / 3) →
  (n ≤ 12) :=
sorry

end max_books_on_shelf_l1562_156245


namespace exists_removable_piece_l1562_156284

/-- Represents a chessboard configuration -/
structure ChessboardConfig where
  pieces : Finset (Fin 8 × Fin 8)
  piece_count : pieces.card = 15
  row_coverage : ∀ r : Fin 8, ∃ c : Fin 8, (r, c) ∈ pieces
  col_coverage : ∀ c : Fin 8, ∃ r : Fin 8, (r, c) ∈ pieces

/-- Theorem stating that there always exists a removable piece -/
theorem exists_removable_piece (config : ChessboardConfig) :
  ∃ p ∈ config.pieces, 
    let remaining := config.pieces.erase p
    (∀ r : Fin 8, ∃ c : Fin 8, (r, c) ∈ remaining) ∧
    (∀ c : Fin 8, ∃ r : Fin 8, (r, c) ∈ remaining) :=
  sorry

end exists_removable_piece_l1562_156284


namespace even_function_implies_a_zero_l1562_156252

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - |x + a|

-- State the theorem
theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by
  sorry

end even_function_implies_a_zero_l1562_156252


namespace a_equals_3_sufficient_not_necessary_l1562_156256

/-- Two lines are parallel if their slopes are equal and they are not identical. -/
def are_parallel (m1 a1 b1 c1 m2 a2 b2 c2 : ℝ) : Prop :=
  m1 = m2 ∧ (a1, b1, c1) ≠ (a2, b2, c2)

/-- The condition for the given lines to be parallel. -/
def parallel_condition (a : ℝ) : Prop :=
  are_parallel (-a/2) a 2 1 (-3/(a-1)) 3 (a-1) (-2)

theorem a_equals_3_sufficient_not_necessary :
  (∃ a, a ≠ 3 ∧ parallel_condition a) ∧
  (parallel_condition 3) :=
sorry

end a_equals_3_sufficient_not_necessary_l1562_156256


namespace equation_solutions_l1562_156297

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 - 81 = 0 ↔ x = 9/2 ∨ x = -9/2) ∧
  (∀ x : ℝ, (x - 1)^3 = -8 ↔ x = -1) := by
  sorry

end equation_solutions_l1562_156297


namespace k_h_negative_three_equals_sixteen_l1562_156246

-- Define the function h
def h (x : ℝ) : ℝ := 4 * x^2 - 8

-- Define a variable k as a function from ℝ to ℝ
variable (k : ℝ → ℝ)

-- State the theorem
theorem k_h_negative_three_equals_sixteen 
  (h_def : ∀ x, h x = 4 * x^2 - 8)
  (k_h_three : k (h 3) = 16) :
  k (h (-3)) = 16 := by
  sorry

end k_h_negative_three_equals_sixteen_l1562_156246


namespace percentage_of_women_in_survey_l1562_156202

theorem percentage_of_women_in_survey (mothers_full_time : Real) 
  (fathers_full_time : Real) (total_not_full_time : Real) :
  mothers_full_time = 5/6 →
  fathers_full_time = 3/4 →
  total_not_full_time = 1/5 →
  ∃ (w : Real), w = 3/5 ∧ 
    w * (1 - mothers_full_time) + (1 - w) * (1 - fathers_full_time) = total_not_full_time :=
by sorry

end percentage_of_women_in_survey_l1562_156202


namespace sufficient_not_necessary_condition_l1562_156241

theorem sufficient_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, x^2 - m*x + 1 ≠ 0) → (|m| < 1) ∧
  ¬(∀ m : ℝ, (∀ x : ℝ, x^2 - m*x + 1 ≠ 0) → (|m| < 1)) :=
by sorry

end sufficient_not_necessary_condition_l1562_156241


namespace temperature_decrease_l1562_156255

theorem temperature_decrease (current_temp : ℝ) (decrease_factor : ℝ) :
  current_temp = 84 →
  decrease_factor = 3/4 →
  current_temp - (decrease_factor * current_temp) = 21 := by
sorry

end temperature_decrease_l1562_156255


namespace expression_equivalence_l1562_156200

theorem expression_equivalence (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 2 * y) :
  (x - 2 / x) * (y + 2 / y) = (1 / 2) * (x^2 - 2*x + 8 - 16 / x) := by
  sorry

end expression_equivalence_l1562_156200


namespace select_volunteers_l1562_156203

theorem select_volunteers (boys girls volunteers : ℕ) 
  (h1 : boys = 6)
  (h2 : girls = 2)
  (h3 : volunteers = 3) :
  (Nat.choose (boys + girls) volunteers) - (Nat.choose boys volunteers) = 36 := by
  sorry

end select_volunteers_l1562_156203


namespace unique_quadruple_l1562_156296

theorem unique_quadruple :
  ∃! (a b c d : ℝ), 
    0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
    a + b + c + d = 2 ∧
    a^2 + b^2 + c^2 + d^2 = 3 ∧
    (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 18 :=
by sorry

end unique_quadruple_l1562_156296


namespace pizza_toppings_l1562_156298

theorem pizza_toppings (total_slices ham_slices olive_slices : ℕ) 
  (h_total : total_slices = 16)
  (h_ham : ham_slices = 8)
  (h_olive : olive_slices = 12)
  (h_at_least_one : ∀ slice, slice ≤ total_slices → (slice ≤ ham_slices ∨ slice ≤ olive_slices)) :
  ∃ both : ℕ, both = ham_slices + olive_slices - total_slices :=
by sorry

end pizza_toppings_l1562_156298


namespace condition_relationship_l1562_156273

theorem condition_relationship (x : ℝ) :
  (∀ x, x > 1 → 1/x < 1) ∧
  (∃ x, 1/x < 1 ∧ x ≤ 1) :=
by sorry

end condition_relationship_l1562_156273


namespace eugene_model_house_l1562_156226

/-- The number of toothpicks required for each card -/
def toothpicks_per_card : ℕ := 75

/-- The total number of cards in a deck -/
def cards_in_deck : ℕ := 52

/-- The number of cards Eugene didn't use -/
def unused_cards : ℕ := 16

/-- The number of toothpicks in each box -/
def toothpicks_per_box : ℕ := 450

/-- Calculate the number of boxes of toothpicks Eugene used -/
def boxes_used : ℕ :=
  let cards_used := cards_in_deck - unused_cards
  let total_toothpicks := cards_used * toothpicks_per_card
  (total_toothpicks + toothpicks_per_box - 1) / toothpicks_per_box

theorem eugene_model_house :
  boxes_used = 6 := by
  sorry

end eugene_model_house_l1562_156226


namespace brooke_earnings_l1562_156219

/-- Represents Brooke's milk and butter business -/
structure MilkBusiness where
  milk_price : ℝ
  butter_cost : ℝ
  milk_to_butter : ℝ
  butter_price : ℝ
  num_cows : ℕ
  milk_per_cow : ℝ
  num_customers : ℕ
  min_demand : ℝ
  max_demand : ℝ

/-- Calculates the total milk produced -/
def total_milk (b : MilkBusiness) : ℝ :=
  b.num_cows * b.milk_per_cow

/-- Calculates Brooke's earnings -/
def earnings (b : MilkBusiness) : ℝ :=
  total_milk b * b.milk_price

/-- Theorem stating that Brooke's earnings are $144 -/
theorem brooke_earnings :
  ∀ b : MilkBusiness,
    b.milk_price = 3 ∧
    b.butter_cost = 0.5 ∧
    b.milk_to_butter = 2 ∧
    b.butter_price = 1.5 ∧
    b.num_cows = 12 ∧
    b.milk_per_cow = 4 ∧
    b.num_customers = 6 ∧
    b.min_demand = 4 ∧
    b.max_demand = 8 →
    earnings b = 144 :=
by
  sorry

end brooke_earnings_l1562_156219


namespace smallest_n_for_candy_l1562_156286

theorem smallest_n_for_candy (n : ℕ+) : 
  (∃ m : ℕ+, 16 ∣ m ∧ 18 ∣ m ∧ 20 ∣ m ∧ m = 30 * n) →
  (∀ k : ℕ+, k < n → ¬∃ m : ℕ+, 16 ∣ m ∧ 18 ∣ m ∧ 20 ∣ m ∧ m = 30 * k) →
  n = 24 := by
sorry

end smallest_n_for_candy_l1562_156286


namespace complex_argument_and_reality_l1562_156222

noncomputable def arg (z : ℂ) : ℝ := Real.arctan (z.im / z.re)

theorem complex_argument_and_reality (θ : ℝ) (a : ℝ) :
  0 < θ ∧ θ < 2 * Real.pi →
  let z : ℂ := 1 - Real.cos θ + Complex.I * Real.sin θ
  let u : ℂ := a^2 + Complex.I * a
  (z * u).re = 0 →
  (
    (0 < θ ∧ θ < Real.pi → arg u = θ / 2) ∧
    (Real.pi < θ ∧ θ < 2 * Real.pi → arg u = Real.pi + θ / 2)
  ) ∧
  ∀ ω : ℂ, ω = z^2 + u^2 + 2 * z * u → ω.im ≠ 0 :=
by sorry

end complex_argument_and_reality_l1562_156222


namespace unique_solution_natural_numbers_l1562_156272

theorem unique_solution_natural_numbers : 
  ∃! (a b : ℕ), a^b + a + b = b^a ∧ a = 5 ∧ b = 2 := by sorry

end unique_solution_natural_numbers_l1562_156272
