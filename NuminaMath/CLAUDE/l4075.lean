import Mathlib

namespace polygon_sides_l4075_407507

theorem polygon_sides (n : ℕ) (missing_angle : ℝ) : 
  (n ≥ 3) →
  (missing_angle < 170) →
  ((n - 2) * 180 - missing_angle = 2970) →
  n = 19 :=
by sorry

end polygon_sides_l4075_407507


namespace sandy_change_l4075_407555

def football_price : ℚ := 9.14
def baseball_price : ℚ := 6.81
def payment : ℚ := 20

theorem sandy_change : payment - (football_price + baseball_price) = 4.05 := by
  sorry

end sandy_change_l4075_407555


namespace smallest_whole_number_above_sum_l4075_407549

def sum_fractions : ℚ :=
  3 + 1/3 + 4 + 1/4 + 5 + 1/6 + 6 + 1/8 + 7 + 1/9

theorem smallest_whole_number_above_sum : 
  ∃ n : ℕ, n = 26 ∧ (∀ m : ℕ, m < n → (m : ℚ) ≤ sum_fractions) ∧ sum_fractions < (n : ℚ) :=
sorry

end smallest_whole_number_above_sum_l4075_407549


namespace quadratic_coefficient_l4075_407539

/-- A quadratic function with vertex (3, 2) passing through (-2, -43) has a = -1.8 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x, (a * x^2 + b * x + c) = a * (x - 3)^2 + 2) → 
  (a * (-2)^2 + b * (-2) + c = -43) → 
  a = -1.8 := by
sorry

end quadratic_coefficient_l4075_407539


namespace f_zero_eq_two_l4075_407500

/-- The function f(x) with parameter a -/
def f (a x : ℝ) : ℝ := (a^2 - 1) * (x^2 - 1) + (a - 1) * (x - 1)

/-- Theorem: If f represents a straight line, then f(0) = 2 -/
theorem f_zero_eq_two (a : ℝ) (h : ∀ x y : ℝ, f a x - f a y = (f a 1 - f a 0) * (x - y)) : 
  f a 0 = 2 := by
  sorry

end f_zero_eq_two_l4075_407500


namespace binomial_coefficient_ratio_l4075_407543

theorem binomial_coefficient_ratio (n : ℕ) : 
  (2^3 * (n.choose 3) = 4 * 2^2 * (n.choose 2)) → n = 8 := by
  sorry

end binomial_coefficient_ratio_l4075_407543


namespace dorothy_doughnut_profit_l4075_407533

/-- Dorothy's doughnut business problem -/
theorem dorothy_doughnut_profit :
  let ingredient_cost : ℤ := 53
  let rent_utilities : ℤ := 27
  let num_doughnuts : ℕ := 25
  let price_per_doughnut : ℤ := 3
  let total_expenses : ℤ := ingredient_cost + rent_utilities
  let revenue : ℤ := num_doughnuts * price_per_doughnut
  let profit : ℤ := revenue - total_expenses
  profit = -5 := by
sorry


end dorothy_doughnut_profit_l4075_407533


namespace angle_D_value_l4075_407553

theorem angle_D_value (A B C D : ℝ) 
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 40)
  (h4 : B + C = 130) :
  D = 40 := by sorry

end angle_D_value_l4075_407553


namespace negation_of_forall_quadratic_inequality_l4075_407566

theorem negation_of_forall_quadratic_inequality :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) := by sorry

end negation_of_forall_quadratic_inequality_l4075_407566


namespace michaels_bills_l4075_407515

def total_amount : ℕ := 280
def bill_denomination : ℕ := 20

theorem michaels_bills : 
  total_amount / bill_denomination = 14 :=
by sorry

end michaels_bills_l4075_407515


namespace seventh_root_unity_product_l4075_407524

theorem seventh_root_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 := by
  sorry

end seventh_root_unity_product_l4075_407524


namespace parallelogram_area_example_l4075_407599

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogramArea (u v : ℝ × ℝ) : ℝ :=
  |u.1 * v.2 - u.2 * v.1|

theorem parallelogram_area_example : 
  let u : ℝ × ℝ := (4, 7)
  let z : ℝ × ℝ := (-6, 3)
  parallelogramArea u z = 54 := by
sorry

end parallelogram_area_example_l4075_407599


namespace water_transfer_problem_l4075_407501

/-- Represents a rectangular pool with given dimensions -/
structure Pool where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Represents a valve with a specific flow rate -/
structure Valve where
  flow_rate : ℝ

/-- The main theorem to prove -/
theorem water_transfer_problem 
  (pool_A pool_B : Pool)
  (valve_1 valve_2 : Valve)
  (h1 : pool_A.length = 3 ∧ pool_A.width = 2 ∧ pool_A.depth = 1.2)
  (h2 : pool_B.length = 3 ∧ pool_B.width = 2 ∧ pool_B.depth = 1.2)
  (h3 : valve_1.flow_rate * 18 = pool_A.length * pool_A.width * pool_A.depth)
  (h4 : valve_2.flow_rate * 24 = pool_A.length * pool_A.width * pool_A.depth)
  (h5 : 0.4 * pool_A.length * pool_A.width = (valve_1.flow_rate - valve_2.flow_rate) * t)
  (h6 : t > 0) :
  valve_2.flow_rate * t = 7.2 := by sorry


end water_transfer_problem_l4075_407501


namespace rhombus_side_length_l4075_407556

/-- A rhombus with an inscribed circle of radius 2, where the diagonal divides the rhombus into two equilateral triangles, has a side length of 8√3/3. -/
theorem rhombus_side_length (r : ℝ) (s : ℝ) :
  r = 2 →  -- The radius of the inscribed circle is 2
  s > 0 →  -- The side length is positive
  s^2 = (s/2)^2 + 16 →  -- From the diagonal relationship
  s * 4 = (s * s) / 2 →  -- Area equality
  s = 8 * Real.sqrt 3 / 3 :=
by sorry

end rhombus_side_length_l4075_407556


namespace special_function_is_one_l4075_407536

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, a > 0 ∧ b > 0 →
    (f (a^2 + b^2) = f a * f b) ∧
    (f (a^2) = (f a)^2)

/-- The main theorem stating that any function satisfying the conditions is constant 1 -/
theorem special_function_is_one (f : ℕ → ℕ) (h : SpecialFunction f) :
  ∀ n : ℕ, n > 0 → f n = 1 := by
  sorry

end special_function_is_one_l4075_407536


namespace solve_auction_problem_l4075_407568

def auction_problem (starting_price harry_initial_increase second_bidder_multiplier third_bidder_addition harry_final_increase : ℕ) : Prop :=
  let harry_first_bid := starting_price + harry_initial_increase
  let second_bid := harry_first_bid * second_bidder_multiplier
  let third_bid := second_bid + (harry_first_bid * third_bidder_addition)
  let harry_final_bid := third_bid + harry_final_increase
  harry_final_bid = 4000

theorem solve_auction_problem :
  auction_problem 300 200 2 3 1500 := by sorry

end solve_auction_problem_l4075_407568


namespace baseball_cards_distribution_l4075_407518

theorem baseball_cards_distribution (n : ℕ) (h : n > 0) :
  ∃ (cards_per_friend : ℕ), 
    cards_per_friend * n = 12 ∧ 
    cards_per_friend = 12 / n :=
sorry

end baseball_cards_distribution_l4075_407518


namespace negation_of_proposition_l4075_407514

theorem negation_of_proposition (p : ℝ → Prop) :
  (∀ n ∈ Set.Icc 1 2, n^2 < 3*n + 4) ↔ ¬(∃ n ∈ Set.Icc 1 2, n^2 ≥ 3*n + 4) :=
by sorry

end negation_of_proposition_l4075_407514


namespace division_problem_l4075_407585

theorem division_problem (n : ℕ) : 
  (n / 6 = 8) → (n % 6 ≤ 5) → (n % 6 = 5) → (n = 53) :=
by
  sorry

end division_problem_l4075_407585


namespace poster_system_area_l4075_407560

/-- Represents a rectangular poster --/
structure Poster where
  length : ℝ
  width : ℝ

/-- Calculates the area of a poster --/
def poster_area (p : Poster) : ℝ := p.length * p.width

/-- Represents the system of overlapping posters --/
structure PosterSystem where
  posters : List Poster
  num_intersections : ℕ

/-- Theorem: The total area covered by the poster system is 96 square feet --/
theorem poster_system_area (ps : PosterSystem) : 
  ps.posters.length = 4 ∧ 
  (∀ p ∈ ps.posters, p.length = 15 ∧ p.width = 2) ∧
  ps.num_intersections = 3 →
  (ps.posters.map poster_area).sum - ps.num_intersections * 8 = 96 := by
  sorry

#check poster_system_area

end poster_system_area_l4075_407560


namespace max_sum_reciprocal_cubes_l4075_407570

/-- The roots of a cubic polynomial satisfying a specific condition -/
structure CubicRoots where
  r₁ : ℝ
  r₂ : ℝ
  r₃ : ℝ
  sum_eq_sum_squares : r₁ + r₂ + r₃ = r₁^2 + r₂^2 + r₃^2

/-- The coefficients of a cubic polynomial -/
structure CubicCoeffs where
  s : ℝ
  p : ℝ
  q : ℝ

/-- The theorem stating the maximum value of the sum of reciprocal cubes of roots -/
theorem max_sum_reciprocal_cubes (roots : CubicRoots) (coeffs : CubicCoeffs) 
  (vieta₁ : roots.r₁ + roots.r₂ + roots.r₃ = coeffs.s)
  (vieta₂ : roots.r₁ * roots.r₂ + roots.r₂ * roots.r₃ + roots.r₃ * roots.r₁ = coeffs.p)
  (vieta₃ : roots.r₁ * roots.r₂ * roots.r₃ = coeffs.q) :
  ∃ (max : ℝ), max = 3 ∧ ∀ (roots' : CubicRoots),
    (1 / roots'.r₁^3 + 1 / roots'.r₂^3 + 1 / roots'.r₃^3) ≤ max :=
sorry

end max_sum_reciprocal_cubes_l4075_407570


namespace box_width_l4075_407528

/-- The width of a rectangular box with given dimensions and filling rate -/
theorem box_width (fill_rate : ℝ) (length depth : ℝ) (fill_time : ℝ) :
  fill_rate = 4 →
  length = 7 →
  depth = 2 →
  fill_time = 21 →
  (fill_rate * fill_time) / (length * depth) = 6 :=
by sorry

end box_width_l4075_407528


namespace sum_of_constants_l4075_407554

def polynomial (x : ℝ) : ℝ := x^3 - 7*x^2 + 14*x - 8

def t (k : ℕ) : ℝ := sorry

theorem sum_of_constants (x y z : ℝ) : 
  (∀ k ≥ 2, t (k+1) = x * t k + y * t (k-1) + z * t (k-2)) →
  t 0 = 3 →
  t 1 = 7 →
  t 2 = 15 →
  x + y + z = 3 := by sorry

end sum_of_constants_l4075_407554


namespace integers_between_neg_one_third_and_two_l4075_407559

theorem integers_between_neg_one_third_and_two :
  ∀ x : ℤ, -1/3 < (x : ℚ) ∧ (x : ℚ) < 2 → x = 0 ∨ x = 1 := by
sorry

end integers_between_neg_one_third_and_two_l4075_407559


namespace isosceles_trapezoid_shorter_base_l4075_407521

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  /-- Length of the longer base -/
  longerBase : ℝ
  /-- Length of the shorter base -/
  shorterBase : ℝ
  /-- Length of the line joining the midpoints of the diagonals -/
  midpointLine : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : True
  /-- The midpoint line length is half the difference of the bases -/
  midpointProperty : midpointLine = (longerBase - shorterBase) / 2

/-- Theorem: In an isosceles trapezoid where the line joining the midpoints of the diagonals
    has length 4 and the longer base is 100, the shorter base has length 92 -/
theorem isosceles_trapezoid_shorter_base
  (t : IsoscelesTrapezoid)
  (h1 : t.longerBase = 100)
  (h2 : t.midpointLine = 4) :
  t.shorterBase = 92 := by
  sorry

end isosceles_trapezoid_shorter_base_l4075_407521


namespace CQRP_is_parallelogram_l4075_407586

-- Define the triangle ABC
structure Triangle (α β : ℝ) :=
  (A B C : ℂ)
  (angle_condition : α > 45 ∧ β > 45)

-- Define the construction of points R, P, and Q
def construct_R (t : Triangle α β) : ℂ :=
  t.B + (t.A - t.B) * Complex.I

def construct_P (t : Triangle α β) : ℂ :=
  t.C + (t.B - t.C) * (-Complex.I)

def construct_Q (t : Triangle α β) : ℂ :=
  t.C + (t.A - t.C) * Complex.I

-- State the theorem
theorem CQRP_is_parallelogram (α β : ℝ) (t : Triangle α β) :
  let R := construct_R t
  let P := construct_P t
  let Q := construct_Q t
  (R + P) / 2 = (t.C + Q) / 2 := by sorry

end CQRP_is_parallelogram_l4075_407586


namespace cos_ninety_degrees_l4075_407561

theorem cos_ninety_degrees : Real.cos (π / 2) = 0 := by
  sorry

end cos_ninety_degrees_l4075_407561


namespace blossom_room_area_l4075_407571

/-- Converts feet and inches to centimeters -/
def to_cm (feet : ℕ) (inches : ℕ) : ℝ :=
  (feet : ℝ) * 30.48 + (inches : ℝ) * 2.54

/-- Calculates the area of a room in square centimeters -/
def room_area (length_feet : ℕ) (length_inches : ℕ) (width_feet : ℕ) (width_inches : ℕ) : ℝ :=
  (to_cm length_feet length_inches) * (to_cm width_feet width_inches)

theorem blossom_room_area :
  room_area 14 8 10 5 = 141935.4 := by
  sorry

end blossom_room_area_l4075_407571


namespace complement_union_theorem_l4075_407520

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 4} := by sorry

end complement_union_theorem_l4075_407520


namespace test_results_l4075_407578

theorem test_results (total_students : ℕ) (correct_q1 : ℕ) (correct_q2 : ℕ) (not_taken : ℕ)
  (h1 : total_students = 40)
  (h2 : correct_q1 = 30)
  (h3 : correct_q2 = 29)
  (h4 : not_taken = 10)
  : (total_students - not_taken) = correct_q1 ∧ correct_q1 - 1 = correct_q2 := by
  sorry

end test_results_l4075_407578


namespace plane_equation_l4075_407513

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if two planes are parallel -/
def parallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ p1.a = k * p2.a ∧ p1.b = k * p2.b ∧ p1.c = k * p2.c

/-- Check if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- Check if coefficients are integers and their GCD is 1 -/
def validCoefficients (plane : Plane) : Prop :=
  Int.gcd (Int.natAbs (Int.floor plane.a)) (Int.gcd (Int.natAbs (Int.floor plane.b)) (Int.gcd (Int.natAbs (Int.floor plane.c)) (Int.natAbs (Int.floor plane.d)))) = 1

theorem plane_equation : ∃ (result : Plane),
  result.a = 2 ∧ result.b = -1 ∧ result.c = 3 ∧ result.d = -14 ∧
  pointOnPlane ⟨2, -1, 3⟩ result ∧
  parallel result ⟨4, -2, 6, -5⟩ ∧
  result.a > 0 ∧
  validCoefficients result :=
sorry

end plane_equation_l4075_407513


namespace percentage_problem_l4075_407579

theorem percentage_problem : ∃ X : ℝ, 
  (X / 100 * 100 = (0.6 * 80 + 22)) ∧ 
  (X = 70) := by sorry

end percentage_problem_l4075_407579


namespace basketball_points_l4075_407569

theorem basketball_points (T : ℕ) : 
  T + (T + 6) + (2 * T + 4) = 26 → T = 4 := by sorry

end basketball_points_l4075_407569


namespace width_to_length_ratio_l4075_407572

/-- A rectangle represents a rectangular hall --/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Properties of the rectangular hall --/
def RectangleProperties (r : Rectangle) : Prop :=
  r.width > 0 ∧ 
  r.length > 0 ∧ 
  r.width * r.length = 450 ∧ 
  r.length - r.width = 15

/-- Theorem stating the ratio of width to length --/
theorem width_to_length_ratio (r : Rectangle) 
  (h : RectangleProperties r) : r.width / r.length = 1 / 2 := by
  sorry

end width_to_length_ratio_l4075_407572


namespace empty_solution_set_iff_k_ge_one_l4075_407548

-- Define the inequality function
def f (k x : ℝ) : ℝ := k * x^2 - 2 * abs (x - 1) + 3 * k

-- Define the property of having an empty solution set
def has_empty_solution_set (k : ℝ) : Prop :=
  ∀ x, f k x ≥ 0

-- State the theorem
theorem empty_solution_set_iff_k_ge_one :
  ∀ k, has_empty_solution_set k ↔ k ≥ 1 := by sorry

end empty_solution_set_iff_k_ge_one_l4075_407548


namespace circle_radius_l4075_407525

/-- Given a circle centered at (0,k) with k > 4, which is tangent to the lines y=x, y=-x, and y=4,
    the radius of the circle is 4(1+√2). -/
theorem circle_radius (k : ℝ) (h1 : k > 4) : ∃ r : ℝ,
  (∀ x y : ℝ, (x = y ∨ x = -y ∨ y = 4) → (x^2 + (y - k)^2 = r^2)) ∧
  r = 4*(1 + Real.sqrt 2) := by
  sorry

end circle_radius_l4075_407525


namespace peach_difference_is_eight_l4075_407550

/-- The number of green peaches in the basket -/
def green_peaches : ℕ := 14

/-- The number of yellow peaches in the basket -/
def yellow_peaches : ℕ := 6

/-- The number of red peaches in the basket -/
def red_peaches : ℕ := 2

/-- The difference between the number of green peaches and yellow peaches -/
def peach_difference : ℕ := green_peaches - yellow_peaches

theorem peach_difference_is_eight : peach_difference = 8 := by
  sorry

end peach_difference_is_eight_l4075_407550


namespace pen_price_payment_l4075_407565

/-- Given the price of a pen and the number of pens bought, determine if the price and total payment are constants or variables -/
theorem pen_price_payment (x : ℕ) (y : ℝ) : 
  (∀ n : ℕ, 3 * n = 3 * n) ∧ (∃ m : ℕ, y ≠ 3 * m) := by
  sorry

end pen_price_payment_l4075_407565


namespace log_equation_solution_l4075_407552

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log (x^3) / Real.log 9 = 9 →
  x = 3^(18/5) := by
sorry

end log_equation_solution_l4075_407552


namespace rectangle_area_rectangle_area_is_180_l4075_407529

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_is_180 :
  rectangle_area 2025 10 = 180 := by
  sorry

end rectangle_area_rectangle_area_is_180_l4075_407529


namespace triangle_inequality_bound_l4075_407506

theorem triangle_inequality_bound (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + c^2) / (b + c)^2 ≤ (1 : ℝ) / 2 ∧
  ∀ ε > 0, ∃ a' b' c' : ℝ, 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b' ∧
    (a'^2 + c'^2) / (b' + c')^2 > (1 : ℝ) / 2 - ε :=
sorry

end triangle_inequality_bound_l4075_407506


namespace city_death_rate_l4075_407577

/-- Represents the population dynamics of a city --/
structure CityPopulation where
  birth_rate : ℕ  -- Birth rate per two seconds
  net_increase : ℕ  -- Net population increase per day

/-- Calculates the death rate per two seconds given city population data --/
def death_rate (city : CityPopulation) : ℕ :=
  let seconds_per_day : ℕ := 24 * 60 * 60
  let birth_rate_per_second : ℕ := city.birth_rate / 2
  let net_increase_per_second : ℕ := city.net_increase / seconds_per_day
  2 * (birth_rate_per_second - net_increase_per_second)

/-- Theorem stating that for the given city data, the death rate is 6 people every two seconds --/
theorem city_death_rate :
  let city : CityPopulation := { birth_rate := 8, net_increase := 86400 }
  death_rate city = 6 := by
  sorry

end city_death_rate_l4075_407577


namespace quadrilateral_dc_length_l4075_407508

theorem quadrilateral_dc_length
  (AB : ℝ) (sinA sinC : ℝ)
  (h1 : AB = 30)
  (h2 : sinA = 1/2)
  (h3 : sinC = 2/5)
  : ∃ (DC : ℝ), DC = 5 * Real.sqrt 47.25 :=
by
  sorry

end quadrilateral_dc_length_l4075_407508


namespace ratio_change_proof_l4075_407563

theorem ratio_change_proof (x y a : ℚ) : 
  y = 40 →
  x / y = 3 / 4 →
  (x + a) / (y + a) = 4 / 5 →
  a = 10 := by
sorry

end ratio_change_proof_l4075_407563


namespace common_chord_circle_center_on_line_smallest_circle_l4075_407504

-- Define the two circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0

-- Define points A and B as the intersection of C₁ and C₂
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (0, 2)

-- Theorem for the common chord
theorem common_chord : 
  ∀ x y : ℝ, C₁ x y ∧ C₂ x y → x - 2*y + 4 = 0 :=
by sorry

-- Theorem for the circle with center on y = -x
theorem circle_center_on_line : 
  ∃ h k : ℝ, h = -k ∧ 
  (A.1 - h)^2 + (A.2 - k)^2 = (B.1 - h)^2 + (B.2 - k)^2 ∧
  ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = 10 ↔ 
  ((x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2)) :=
by sorry

-- Theorem for the smallest circle
theorem smallest_circle : 
  ∀ x y : ℝ, (x + 2)^2 + (y - 1)^2 = 5 ↔ 
  ((x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2)) :=
by sorry

end common_chord_circle_center_on_line_smallest_circle_l4075_407504


namespace parallel_line_m_value_l4075_407581

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Get the line passing through two points -/
def line_through_points (p1 p2 : Point) : Line :=
  { a := p2.y - p1.y
    b := p1.x - p2.x
    c := p2.x * p1.y - p1.x * p2.y }

/-- The main theorem -/
theorem parallel_line_m_value :
  ∀ m : ℝ,
  let A : Point := ⟨-2, m⟩
  let B : Point := ⟨m, 4⟩
  let L1 : Line := line_through_points A B
  let L2 : Line := ⟨2, 1, -1⟩
  are_parallel L1 L2 → m = -8 := by
  sorry

end parallel_line_m_value_l4075_407581


namespace min_value_inequality_l4075_407562

theorem min_value_inequality (a : ℝ) : 
  (∀ x y : ℝ, |x| + |y| ≤ 1 → |2*x - 3*y + 3/2| + |y - 1| + |2*y - x - 3| ≤ a) ↔ 
  23/2 ≤ a :=
sorry

end min_value_inequality_l4075_407562


namespace trader_profit_equation_l4075_407503

/-- The trader's profit after a week of sales -/
def trader_profit : ℝ := 960

/-- The amount of donations received -/
def donations : ℝ := 310

/-- The trader's goal amount -/
def goal : ℝ := 610

/-- The amount above the goal -/
def above_goal : ℝ := 180

theorem trader_profit_equation :
  trader_profit / 2 + donations = goal + above_goal :=
by sorry

end trader_profit_equation_l4075_407503


namespace jerry_throwing_points_l4075_407587

/-- Represents the point system in Mrs. Carlton's class -/
structure PointSystem where
  interrupt_points : ℕ
  insult_points : ℕ
  office_threshold : ℕ

/-- Represents Jerry's behavior -/
structure JerryBehavior where
  interrupts : ℕ
  insults : ℕ
  throws : ℕ

/-- Calculates the points Jerry has accumulated so far -/
def accumulated_points (ps : PointSystem) (jb : JerryBehavior) : ℕ :=
  ps.interrupt_points * jb.interrupts + ps.insult_points * jb.insults

/-- Theorem stating that Jerry gets 25 points for throwing things -/
theorem jerry_throwing_points (ps : PointSystem) (jb : JerryBehavior) :
    ps.interrupt_points = 5 →
    ps.insult_points = 10 →
    ps.office_threshold = 100 →
    jb.interrupts = 2 →
    jb.insults = 4 →
    jb.throws = 2 →
    (ps.office_threshold - accumulated_points ps jb) / jb.throws = 25 := by
  sorry

end jerry_throwing_points_l4075_407587


namespace sqrt_equation_solution_l4075_407535

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt x + Real.sqrt (x + 6) = 12 → x = 529 / 16 := by
  sorry

end sqrt_equation_solution_l4075_407535


namespace count_of_satisfying_integers_l4075_407544

/-- The number of integers satisfying the equation -/
def solution_count : ℕ := 40200

/-- The equation to be satisfied -/
def satisfies_equation (n : ℤ) : Prop :=
  1 + ⌊(200 * n) / 201⌋ = ⌈(198 * n) / 200⌉

theorem count_of_satisfying_integers :
  (∃! (s : Finset ℤ), s.card = solution_count ∧ ∀ n, n ∈ s ↔ satisfies_equation n) :=
sorry

end count_of_satisfying_integers_l4075_407544


namespace min_points_for_obtuse_triangle_l4075_407580

/-- A color representing red, yellow, or blue -/
inductive Color
  | Red
  | Yellow
  | Blue

/-- A point on the circumference of a circle -/
structure CirclePoint where
  angle : Real
  color : Color

/-- A function that colors every point on the circle's circumference -/
def colorCircle : Real → Color := sorry

/-- Predicate to check if all three colors are present on the circle -/
def allColorsPresent (colorCircle : Real → Color) : Prop := sorry

/-- Predicate to check if three points form an obtuse triangle -/
def isObtuseTriangle (p1 p2 p3 : CirclePoint) : Prop := sorry

/-- The minimum number of points that guarantees an obtuse triangle of the same color -/
def minPointsForObtuseTriangle : Nat := sorry

/-- Theorem stating the minimum number of points required -/
theorem min_points_for_obtuse_triangle :
  ∀ (colorCircle : Real → Color),
    allColorsPresent colorCircle →
    (∀ (points : Finset CirclePoint),
      points.card ≥ minPointsForObtuseTriangle →
      ∃ (p1 p2 p3 : CirclePoint),
        p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
        p1.color = p2.color ∧ p2.color = p3.color ∧
        isObtuseTriangle p1 p2 p3) ∧
    minPointsForObtuseTriangle = 13 :=
by sorry

end min_points_for_obtuse_triangle_l4075_407580


namespace max_value_abcd_l4075_407541

theorem max_value_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a * b * c * d * (a + b + c + d)) / ((a + b)^2 * (c + d)^2) ≤ (1 : ℝ) / 4 := by
  sorry

end max_value_abcd_l4075_407541


namespace parallel_lines_circle_solution_l4075_407526

/-- A circle intersected by three equally spaced parallel lines -/
structure ParallelLinesCircle where
  /-- The radius of the circle -/
  r : ℝ
  /-- The distance between adjacent parallel lines -/
  d : ℝ
  /-- The lengths of the three chords formed by the intersection -/
  chord1 : ℝ
  chord2 : ℝ
  chord3 : ℝ
  /-- The chords are formed by equally spaced parallel lines -/
  parallel_lines : chord1 = chord3
  /-- The given chord lengths -/
  chord_lengths : chord1 = 40 ∧ chord2 = 36 ∧ chord3 = 40

/-- The theorem stating the distance between lines and radius of the circle -/
theorem parallel_lines_circle_solution (c : ParallelLinesCircle) :
  c.d = Real.sqrt 1188 ∧ c.r = Real.sqrt 357 :=
by sorry

end parallel_lines_circle_solution_l4075_407526


namespace wall_painting_fraction_l4075_407505

theorem wall_painting_fraction (total_time minutes : ℕ) (fraction : ℚ) : 
  total_time = 60 → 
  minutes = 12 → 
  fraction = minutes / total_time → 
  fraction = 1 / 5 := by
sorry

end wall_painting_fraction_l4075_407505


namespace track_length_l4075_407534

/-- Represents a circular track with two runners -/
structure CircularTrack where
  length : ℝ
  runner1_speed : ℝ
  runner2_speed : ℝ

/-- Theorem stating the length of the track given the conditions -/
theorem track_length (track : CircularTrack) 
  (h1 : track.runner1_speed > 0)
  (h2 : track.runner2_speed > 0)
  (h3 : track.length / 2 = 100)
  (h4 : 200 = track.runner2_speed * (track.length / (track.runner1_speed + track.runner2_speed)))
  : track.length = 200 := by
  sorry

#check track_length

end track_length_l4075_407534


namespace baker_cakes_left_l4075_407564

/-- Given a baker who made a total of 217 cakes and sold 145 of them,
    prove that the number of cakes left is 72. -/
theorem baker_cakes_left (total : ℕ) (sold : ℕ) (h1 : total = 217) (h2 : sold = 145) :
  total - sold = 72 := by
  sorry

end baker_cakes_left_l4075_407564


namespace arithmetic_sequence_property_l4075_407582

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence property
  a 3 + a 5 = 2 →                                       -- given condition
  a 4 = 1 :=                                            -- conclusion to prove
by sorry

end arithmetic_sequence_property_l4075_407582


namespace birthday_square_l4075_407540

theorem birthday_square (x y : ℕ+) (h1 : 40000 + 1000 * x + 100 * y + 29 < 100000) : 
  ∃ (T : ℕ), T = 2379 ∧ T^2 = 40000 + 1000 * x + 100 * y + 29 := by
  sorry

end birthday_square_l4075_407540


namespace profit_share_difference_l4075_407567

/-- Represents the initial capital and interest rate for each partner --/
structure Partner where
  capital : ℕ
  rate : ℚ

/-- Calculates the interest earned by a partner --/
def interest (p : Partner) : ℚ := p.capital * p.rate

/-- Calculates the profit share of a partner --/
def profitShare (p : Partner) (totalProfit : ℕ) : ℚ :=
  p.capital + interest p

theorem profit_share_difference
  (a b c : Partner)
  (ha : a.capital = 8000 ∧ a.rate = 5/100)
  (hb : b.capital = 10000 ∧ b.rate = 6/100)
  (hc : c.capital = 12000 ∧ c.rate = 7/100)
  (totalProfit : ℕ)
  (hProfit : profitShare b totalProfit = 13600) :
  profitShare c totalProfit - profitShare a totalProfit = 4440 := by
  sorry

end profit_share_difference_l4075_407567


namespace min_notebooks_correct_l4075_407516

/-- The minimum number of notebooks needed to get a discount -/
def min_notebooks : ℕ := 18

/-- The cost of a single pen in yuan -/
def pen_cost : ℕ := 10

/-- The cost of a single notebook in yuan -/
def notebook_cost : ℕ := 4

/-- The number of pens Xiao Wei plans to buy -/
def num_pens : ℕ := 3

/-- The minimum spending amount to get a discount in yuan -/
def discount_threshold : ℕ := 100

/-- Theorem stating that min_notebooks is the minimum number of notebooks
    needed to get the discount -/
theorem min_notebooks_correct : 
  (num_pens * pen_cost + min_notebooks * notebook_cost ≥ discount_threshold) ∧ 
  (∀ n : ℕ, n < min_notebooks → num_pens * pen_cost + n * notebook_cost < discount_threshold) :=
sorry

end min_notebooks_correct_l4075_407516


namespace derivative_of_f_l4075_407551

-- Define the function f(x) = (3x+4)(2x+6)
def f (x : ℝ) : ℝ := (3*x + 4) * (2*x + 6)

-- State the theorem
theorem derivative_of_f (x : ℝ) : 
  deriv f x = 12*x + 26 := by sorry

end derivative_of_f_l4075_407551


namespace f_zero_at_one_f_zero_at_five_f_value_at_three_l4075_407575

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2 * x^2 + 12 * x - 10

/-- The function f has a zero at x = 1 -/
theorem f_zero_at_one : f 1 = 0 := by sorry

/-- The function f has a zero at x = 5 -/
theorem f_zero_at_five : f 5 = 0 := by sorry

/-- The function f takes the value 8 when x = 3 -/
theorem f_value_at_three : f 3 = 8 := by sorry

end f_zero_at_one_f_zero_at_five_f_value_at_three_l4075_407575


namespace beef_pounds_calculation_l4075_407519

theorem beef_pounds_calculation (total_cost : ℝ) (chicken_cost : ℝ) (oil_cost : ℝ) (beef_cost_per_pound : ℝ) :
  total_cost = 16 ∧ 
  chicken_cost = 3 ∧ 
  oil_cost = 1 ∧ 
  beef_cost_per_pound = 4 →
  (total_cost - chicken_cost - oil_cost) / beef_cost_per_pound = 3 := by
  sorry

end beef_pounds_calculation_l4075_407519


namespace geometric_sequence_first_term_l4075_407558

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (· + 1)

/-- Theorem: In a geometric sequence where the fourth term is 6! and the seventh term is 7!, the first term is 720/7. -/
theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geometric : IsGeometricSequence a)
  (h_fourth : a 4 = factorial 6)
  (h_seventh : a 7 = factorial 7) :
  a 1 = 720 / 7 := by
sorry

end geometric_sequence_first_term_l4075_407558


namespace solve_laptop_battery_problem_l4075_407589

def laptop_battery_problem (standby_capacity : ℝ) (gaming_capacity : ℝ) 
  (standby_used : ℝ) (gaming_used : ℝ) : Prop :=
  standby_capacity = 10 ∧ 
  gaming_capacity = 2 ∧ 
  standby_used = 4 ∧ 
  gaming_used = 1 ∧ 
  (1 - (standby_used / standby_capacity + gaming_used / gaming_capacity)) * standby_capacity = 1

theorem solve_laptop_battery_problem :
  ∀ standby_capacity gaming_capacity standby_used gaming_used,
  laptop_battery_problem standby_capacity gaming_capacity standby_used gaming_used := by
  sorry

end solve_laptop_battery_problem_l4075_407589


namespace factorization_sum_l4075_407557

theorem factorization_sum (a b : ℤ) :
  (∀ x, 25 * x^2 - 155 * x - 150 = (5 * x + a) * (5 * x + b)) →
  a + 2 * b = 27 := by
sorry

end factorization_sum_l4075_407557


namespace reinforcement_size_l4075_407594

/-- Calculates the size of reinforcement given initial garrison size, initial provision duration,
    days before reinforcement, and remaining provision duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_provisions : ℕ) 
                            (days_before_reinforcement : ℕ) (remaining_provisions : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_provisions
  let used_provisions := initial_garrison * days_before_reinforcement
  let remaining_total := total_provisions - used_provisions
  (remaining_total / remaining_provisions) - initial_garrison

theorem reinforcement_size :
  let initial_garrison : ℕ := 2000
  let initial_provisions : ℕ := 54
  let days_before_reinforcement : ℕ := 21
  let remaining_provisions : ℕ := 20
  calculate_reinforcement initial_garrison initial_provisions days_before_reinforcement remaining_provisions = 1300 := by
  sorry

end reinforcement_size_l4075_407594


namespace total_pages_theorem_l4075_407523

/-- The number of pages Jairus read -/
def jairus_pages : ℕ := 20

/-- The number of pages Arniel read -/
def arniel_pages : ℕ := 2 * jairus_pages + 2

/-- The total number of pages read by Jairus and Arniel -/
def total_pages : ℕ := jairus_pages + arniel_pages

theorem total_pages_theorem : total_pages = 62 := by
  sorry

end total_pages_theorem_l4075_407523


namespace technician_round_trip_completion_l4075_407597

theorem technician_round_trip_completion (D : ℝ) (h : D > 0) : 
  let total_distance : ℝ := 2 * D
  let completed_distance : ℝ := D + 0.2 * D
  (completed_distance / total_distance) * 100 = 60 := by
sorry

end technician_round_trip_completion_l4075_407597


namespace hyperbola_theorem_l4075_407546

/-- The standard form of a hyperbola with center at the origin --/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Check if a point (x, y) is on the hyperbola --/
def Hyperbola.contains (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- Check if two hyperbolas have the same asymptotes --/
def same_asymptotes (h1 h2 : Hyperbola) : Prop :=
  h1.a^2 / h1.b^2 = h2.a^2 / h2.b^2

theorem hyperbola_theorem (h1 h2 : Hyperbola) :
  h1.a^2 = 3 ∧ h1.b^2 = 12 ∧
  h2.a^2 = 1 ∧ h2.b^2 = 4 →
  same_asymptotes h1 h2 ∧ h1.contains 2 2 := by
  sorry

end hyperbola_theorem_l4075_407546


namespace positive_f_one_l4075_407530

def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem positive_f_one (f : ℝ → ℝ) 
    (h_mono : MonoIncreasing f) (h_odd : OddFunction f) : 
    f 1 > 0 := by
  sorry

end positive_f_one_l4075_407530


namespace ratio_x_to_y_l4075_407531

theorem ratio_x_to_y (x y : ℝ) (h : (8*x - 5*y) / (11*x - 3*y) = 4/7) : 
  x/y = 23/12 := by sorry

end ratio_x_to_y_l4075_407531


namespace locus_of_T_and_min_distance_l4075_407574

-- Define the circle A
def circle_A (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 12

-- Define point B
def point_B : ℝ × ℝ := (1, 0)

-- Define the locus Γ
def Γ (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 = 1

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem locus_of_T_and_min_distance :
  ∃ (P : ℝ × ℝ) (T : ℝ × ℝ),
    (circle_A P.1 P.2) ∧
    (∃ (M N : ℝ × ℝ) (H : ℝ × ℝ),
      (Γ M.1 M.2) ∧ (Γ N.1 N.2) ∧
      (H = ((M.1 + N.1) / 2, (M.2 + N.2) / 2)) ∧
      (unit_circle H.1 H.2)) →
    ((∀ (x y : ℝ), (Γ x y) ↔ (x^2 / 3 + y^2 / 2 = 1)) ∧
     (∃ (d : ℝ),
       d = 2 * Real.sqrt 6 / 5 ∧
       ∀ (M N : ℝ × ℝ),
         (Γ M.1 M.2) → (Γ N.1 N.2) →
         (unit_circle ((M.1 + N.1) / 2) ((M.2 + N.2) / 2)) →
         d ≤ Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2))) :=
by sorry


end locus_of_T_and_min_distance_l4075_407574


namespace japanese_students_fraction_l4075_407510

theorem japanese_students_fraction (J : ℕ) : 
  let S := 2 * J
  let seniors_japanese := (3 * S) / 8
  let juniors_japanese := J / 4
  let total_students := S + J
  let total_japanese := seniors_japanese + juniors_japanese
  (total_japanese : ℚ) / total_students = 1 / 3 := by
sorry

end japanese_students_fraction_l4075_407510


namespace base_equality_l4075_407596

/-- Converts a base 6 number to its decimal equivalent -/
def base6ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number in base b to its decimal equivalent -/
def baseBToDecimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- The unique positive integer b that satisfies 34₆ = 121ᵦ is 3 -/
theorem base_equality : ∃! (b : ℕ), b > 0 ∧ base6ToDecimal 34 = baseBToDecimal 121 b ∧ b = 3 := by sorry

end base_equality_l4075_407596


namespace petya_wins_l4075_407595

/-- Represents a position on the board -/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents the game state -/
structure GameState :=
  (board : Fin 101 → Fin 101 → Bool)
  (lastMoveLength : Nat)

/-- Represents a move in the game -/
inductive Move
  | Initial : Position → Move
  | Strip : Position → Nat → Move

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (move : Move) : Bool :=
  match move with
  | Move.Initial _ => state.lastMoveLength = 0
  | Move.Strip _ n => n = state.lastMoveLength ∨ n = state.lastMoveLength + 1

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if a player has a winning strategy from a given state -/
def hasWinningStrategy (state : GameState) (isFirstPlayer : Bool) : Prop :=
  sorry

/-- The main theorem stating that the first player (Petya) has a winning strategy -/
theorem petya_wins :
  ∃ (initialMove : Move),
    isValidMove { board := λ _ _ => false, lastMoveLength := 0 } initialMove ∧
    hasWinningStrategy (applyMove { board := λ _ _ => false, lastMoveLength := 0 } initialMove) true :=
  sorry

end petya_wins_l4075_407595


namespace second_largest_divided_by_smallest_remainder_l4075_407545

theorem second_largest_divided_by_smallest_remainder : ∃ (a b c d : ℕ),
  (a = 10 ∧ b = 11 ∧ c = 12 ∧ d = 13) →
  (a < b ∧ b < c ∧ c < d) →
  c % a = 2 := by
sorry

end second_largest_divided_by_smallest_remainder_l4075_407545


namespace jellybean_problem_l4075_407573

theorem jellybean_problem :
  ∃ n : ℕ, n ≥ 200 ∧ n % 17 = 15 ∧ ∀ m : ℕ, m ≥ 200 ∧ m % 17 = 15 → m ≥ n :=
by
  -- The proof goes here
  sorry

end jellybean_problem_l4075_407573


namespace min_ratio_of_intersections_l4075_407502

theorem min_ratio_of_intersections (a : ℝ) (ha : a > 0) :
  let f (x : ℝ) := |Real.log x / Real.log 4|
  let x_A := (4 : ℝ) ^ (-a)
  let x_B := (4 : ℝ) ^ a
  let x_C := (4 : ℝ) ^ (-18 / (2 * a + 1))
  let x_D := (4 : ℝ) ^ (18 / (2 * a + 1))
  let m := |x_A - x_C|
  let n := |x_B - x_D|
  ∃ (a_min : ℝ), ∀ (a : ℝ), a > 0 → n / m ≥ 2^11 ∧ n / m = 2^11 ↔ a = a_min :=
sorry

end min_ratio_of_intersections_l4075_407502


namespace arithmetic_sequence_b_l4075_407588

def arithmetic_sequence (a₁ a₂ a₃ : ℝ) : Prop :=
  ∃ d : ℝ, a₂ = a₁ + d ∧ a₃ = a₂ + d

theorem arithmetic_sequence_b (b : ℝ) 
  (h₁ : arithmetic_sequence 120 b (1/5))
  (h₂ : b > 0) : 
  b = 60.1 := by sorry

end arithmetic_sequence_b_l4075_407588


namespace calculate_expression_l4075_407598

theorem calculate_expression : |-3| - 2 * Real.tan (π / 4) + (-1) ^ 2023 - (Real.sqrt 3 - Real.pi) ^ 0 = -1 := by
  sorry

end calculate_expression_l4075_407598


namespace neg_p_sufficient_not_necessary_for_neg_q_l4075_407538

-- Define the conditions
def p (x : ℝ) : Prop := abs x > 1
def q (x : ℝ) : Prop := x < -2

-- State the theorem
theorem neg_p_sufficient_not_necessary_for_neg_q :
  (∀ x : ℝ, ¬(p x) → ¬(q x)) ∧ (∃ x : ℝ, ¬(q x) ∧ p x) := by
  sorry

end neg_p_sufficient_not_necessary_for_neg_q_l4075_407538


namespace arithmetic_geometric_mean_product_l4075_407584

theorem arithmetic_geometric_mean_product : 
  ∀ a b : ℝ, 
  (a = (1 + 2) / 2) → 
  (b^2 = (-1) * (-16)) → 
  (a * b = 6 ∨ a * b = -6) := by
sorry

end arithmetic_geometric_mean_product_l4075_407584


namespace fraction_unchanged_l4075_407517

theorem fraction_unchanged (x y : ℝ) : (x + y) / (x - 2*y) = ((-x) + (-y)) / ((-x) - 2*(-y)) :=
by sorry

end fraction_unchanged_l4075_407517


namespace money_distribution_l4075_407542

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (AC : A + C = 200)
  (BC : B + C = 360) :
  C = 60 := by
sorry

end money_distribution_l4075_407542


namespace special_triangle_properties_l4075_407512

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.a = t.b * Real.cos t.C + (Real.sqrt 3 / 3) * t.c * Real.sin t.B ∧
  t.a = 2 ∧
  (1 / 2) * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 3

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.B = π / 3 ∧ t.b = 2 * Real.sqrt 3 := by
  sorry


end special_triangle_properties_l4075_407512


namespace modulo_graph_intercepts_sum_l4075_407537

theorem modulo_graph_intercepts_sum (m : Nat) (x₀ y₀ : Nat) : m = 7 →
  0 ≤ x₀ → x₀ < m →
  0 ≤ y₀ → y₀ < m →
  (2 * x₀) % m = 1 % m →
  (3 * y₀ + 1) % m = 0 →
  x₀ + y₀ = 6 := by
sorry

end modulo_graph_intercepts_sum_l4075_407537


namespace solve_equation_l4075_407576

-- Define the * operation
def star (a b : ℝ) : ℝ := 3 * a - b

-- State the theorem
theorem solve_equation : ∃ x : ℝ, star 2 (star 5 x) = 1 ∧ x = 10 := by
  sorry

end solve_equation_l4075_407576


namespace simple_interest_time_period_l4075_407590

theorem simple_interest_time_period 
  (P : ℝ) -- Principal sum
  (R : ℝ) -- Rate of interest per annum
  (T : ℝ) -- Time period in years
  (h1 : R = 4) -- Given rate of interest is 4%
  (h2 : P / 5 = (P * R * T) / 100) -- Simple interest is one-fifth of principal and follows the formula
  : T = 5 := by
sorry

end simple_interest_time_period_l4075_407590


namespace integer_solutions_exist_l4075_407593

theorem integer_solutions_exist : ∃ (k x : ℤ), (k - 5) * x + 6 = 1 - 5 * x ∧
  ((k = 1 ∧ x = -5) ∨ (k = -1 ∧ x = 5) ∨ (k = 5 ∧ x = -1) ∨ (k = -5 ∧ x = 1)) :=
by sorry

end integer_solutions_exist_l4075_407593


namespace temperature_conversion_l4075_407522

theorem temperature_conversion (C F : ℝ) : 
  C = 35 → C = (4/7) * (F - 40) → F = 101.25 := by
  sorry

end temperature_conversion_l4075_407522


namespace tangent_secant_theorem_l4075_407509

/-- Represents a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Check if a point is outside a circle -/
def is_outside (p : Point) (c : Circle) : Prop := sorry

/-- Check if a segment is tangent to a circle -/
def is_tangent (p q : Point) (c : Circle) : Prop := sorry

/-- Check if a segment is a secant of a circle -/
def is_secant (p q r : Point) (c : Circle) : Prop := sorry

theorem tangent_secant_theorem (C : Circle) (Q U M N : Point) :
  is_outside Q C →
  is_tangent Q U C →
  is_secant Q M N C →
  distance Q M < distance Q N →
  distance Q M = 4 →
  distance Q U = distance M N - distance Q M →
  distance Q N = 16 := by sorry

end tangent_secant_theorem_l4075_407509


namespace inequality_proof_l4075_407527

theorem inequality_proof (u v w : ℝ) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h : u + v + w + Real.sqrt (u * v * w) = 4) :
  Real.sqrt (v * w / u) + Real.sqrt (u * w / v) + Real.sqrt (u * v / w) ≥ u + v + w := by
  sorry

end inequality_proof_l4075_407527


namespace tank_problem_l4075_407532

theorem tank_problem (tank1_capacity tank2_capacity tank3_capacity : ℚ)
  (tank1_fill_ratio tank2_fill_ratio : ℚ) (total_water : ℚ) :
  tank1_capacity = 7000 →
  tank2_capacity = 5000 →
  tank3_capacity = 3000 →
  tank1_fill_ratio = 3/4 →
  tank2_fill_ratio = 4/5 →
  total_water = 10850 →
  (total_water - (tank1_capacity * tank1_fill_ratio + tank2_capacity * tank2_fill_ratio)) / tank3_capacity = 8/15 := by
  sorry

end tank_problem_l4075_407532


namespace quadratic_root_property_l4075_407591

theorem quadratic_root_property : ∀ a b : ℝ, 
  (a^2 - 3*a + 1 = 0) → (b^2 - 3*b + 1 = 0) → (a + b - a*b = 2) := by sorry

end quadratic_root_property_l4075_407591


namespace rabbits_per_cat_l4075_407511

theorem rabbits_per_cat (total_animals : ℕ) (num_cats : ℕ) (hares_per_rabbit : ℕ) :
  total_animals = 37 →
  num_cats = 4 →
  hares_per_rabbit = 3 →
  ∃ (rabbits_per_cat : ℕ),
    total_animals = 1 + num_cats + (num_cats * rabbits_per_cat) + (num_cats * rabbits_per_cat * hares_per_rabbit) ∧
    rabbits_per_cat = 2 := by
  sorry

end rabbits_per_cat_l4075_407511


namespace expression_evaluation_l4075_407547

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (hsum : x + y ≠ 0) (hsum_sq : x^2 + y^2 ≠ 0) :
  (x^2 + y^2)⁻¹ * ((x + y)⁻¹ + (x / y)⁻¹) = (1 + y) / ((x^2 + y^2) * (x + y)) := by
  sorry

end expression_evaluation_l4075_407547


namespace charity_fundraising_l4075_407583

theorem charity_fundraising (total : ℕ) (people : ℕ) (raised : ℕ) (target : ℕ) :
  total = 2100 →
  people = 8 →
  raised = 150 →
  target = 279 →
  (total - raised) / (people - 1) = target := by
  sorry

end charity_fundraising_l4075_407583


namespace book_sales_properties_l4075_407592

/-- Represents the daily sales and profit functions for a book selling business -/
structure BookSales where
  cost : ℝ              -- Cost price per book
  min_price : ℝ         -- Minimum selling price
  max_profit_rate : ℝ   -- Maximum profit rate
  base_sales : ℝ        -- Base sales at minimum price
  sales_decrease : ℝ    -- Sales decrease per unit price increase

variable (bs : BookSales)

/-- Daily sales as a function of price -/
def daily_sales (x : ℝ) : ℝ := bs.base_sales - bs.sales_decrease * (x - bs.min_price)

/-- Daily profit as a function of price -/
def daily_profit (x : ℝ) : ℝ := (x - bs.cost) * (daily_sales bs x)

/-- Theorem stating the properties of the book selling business -/
theorem book_sales_properties (bs : BookSales) 
  (h_cost : bs.cost = 40)
  (h_min_price : bs.min_price = 45)
  (h_max_profit_rate : bs.max_profit_rate = 0.5)
  (h_base_sales : bs.base_sales = 310)
  (h_sales_decrease : bs.sales_decrease = 10) :
  -- 1. Daily sales function
  (∀ x, daily_sales bs x = -10 * x + 760) ∧
  -- 2. Selling price range
  (∀ x, bs.min_price ≤ x ∧ x ≤ bs.cost * (1 + bs.max_profit_rate)) ∧
  -- 3. Profit-maximizing price
  (∃ x_max, ∀ x, daily_profit bs x ≤ daily_profit bs x_max ∧ x_max = 58) ∧
  -- 4. Maximum daily profit
  (∃ max_profit, max_profit = daily_profit bs 58 ∧ max_profit = 3240) ∧
  -- 5. Price for $2600 profit
  (∃ x_2600, daily_profit bs x_2600 = 2600 ∧ x_2600 = 50) := by
  sorry

end book_sales_properties_l4075_407592
