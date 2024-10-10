import Mathlib

namespace problem_statement_l4041_404184

theorem problem_statement (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (a^2)^(4*b) = a^(2*b) * x^(3*b) → x = a^2 :=
by sorry

end problem_statement_l4041_404184


namespace reduced_speed_percentage_l4041_404108

theorem reduced_speed_percentage (usual_time : ℝ) (additional_time : ℝ) : 
  usual_time = 24 → additional_time = 24 → 
  (usual_time / (usual_time + additional_time)) * 100 = 50 := by
  sorry

end reduced_speed_percentage_l4041_404108


namespace red_cards_after_turning_l4041_404160

def is_divisible (n m : ℕ) : Prop := ∃ k, n = m * k

def count_red_cards (n : ℕ) : ℕ :=
  let initial_red := n
  let turned_by_2 := n / 2
  let odd_turned_by_3 := (n / 3 + 1) / 2
  let even_turned_by_3 := n / 6
  initial_red - turned_by_2 - odd_turned_by_3 + even_turned_by_3

theorem red_cards_after_turning (n : ℕ) (h : n = 100) : count_red_cards n = 49 := by
  sorry

end red_cards_after_turning_l4041_404160


namespace simplify_and_rationalize_l4041_404135

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 5) * (Real.sqrt 5 / Real.sqrt 7) * (Real.sqrt 8 / Real.sqrt 11) =
  2 * Real.sqrt 462 / 77 := by
  sorry

end simplify_and_rationalize_l4041_404135


namespace expression_evaluation_l4041_404124

theorem expression_evaluation (a b c : ℕ) (ha : a = 3) (hb : b = 2) (hc : c = 4) :
  ((a^b)^a - (b^a)^b) * c = 2660 := by
  sorry

end expression_evaluation_l4041_404124


namespace weight_of_CaI2_l4041_404173

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of calcium atoms in CaI2 -/
def num_Ca_atoms : ℕ := 1

/-- The number of iodine atoms in CaI2 -/
def num_I_atoms : ℕ := 2

/-- The number of moles of CaI2 -/
def num_moles : ℝ := 3

/-- The molecular weight of CaI2 in g/mol -/
def molecular_weight_CaI2 : ℝ := atomic_weight_Ca * num_Ca_atoms + atomic_weight_I * num_I_atoms

/-- The total weight of CaI2 in grams -/
def weight_CaI2 : ℝ := molecular_weight_CaI2 * num_moles

theorem weight_of_CaI2 : weight_CaI2 = 881.64 := by sorry

end weight_of_CaI2_l4041_404173


namespace problem_1_problem_2_l4041_404197

-- Problem 1
theorem problem_1 (x : ℝ) : (-2*x)^2 + 3*x*x = 7*x^2 := by sorry

-- Problem 2
theorem problem_2 (m a b : ℝ) : m*a^2 - m*b^2 = m*(a - b)*(a + b) := by sorry

end problem_1_problem_2_l4041_404197


namespace gravel_path_width_is_quarter_length_l4041_404116

/-- Represents a rectangular garden with a rose garden and gravel path. -/
structure RectangularGarden where
  length : ℝ
  width : ℝ
  roseGardenArea : ℝ
  gravelPathWidth : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  roseGarden_half : roseGardenArea = (length * width) / 2
  gravelPath_constant : gravelPathWidth > 0

/-- Theorem stating that the gravel path width is one-fourth of the garden length. -/
theorem gravel_path_width_is_quarter_length (garden : RectangularGarden) :
  garden.gravelPathWidth = garden.length / 4 := by
  sorry

end gravel_path_width_is_quarter_length_l4041_404116


namespace largest_prime_factor_of_factorial_sum_l4041_404167

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

theorem largest_prime_factor_of_factorial_sum :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (factorial 6 + factorial 7) ∧
  ∀ (q : ℕ), Nat.Prime q → q ∣ (factorial 6 + factorial 7) → q ≤ p :=
sorry

end largest_prime_factor_of_factorial_sum_l4041_404167


namespace magnitude_of_z_l4041_404111

/-- The complex number z defined as 1 + 2i + i^3 -/
def z : ℂ := 1 + 2 * Complex.I + Complex.I ^ 3

/-- Theorem stating that the magnitude of z is √2 -/
theorem magnitude_of_z : Complex.abs z = Real.sqrt 2 := by sorry

end magnitude_of_z_l4041_404111


namespace arithmetic_sequence_sum_l4041_404168

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 1 + a 2011 = 10) →
  (a 1 * a 2011 = 16) →
  a 2 + a 1006 + a 2010 = 15 := by
  sorry

end arithmetic_sequence_sum_l4041_404168


namespace meat_purchase_cost_l4041_404190

/-- Represents the cost and quantity of a type of meat -/
structure Meat where
  name : String
  cost : ℝ
  quantity : ℝ

/-- Calculates the total cost of a purchase given meat prices and quantities -/
def totalCost (meats : List Meat) : ℝ :=
  meats.map (fun m => m.cost * m.quantity) |>.sum

/-- Theorem stating the total cost of the meat purchase -/
theorem meat_purchase_cost :
  let pork_cost : ℝ := 6
  let chicken_cost : ℝ := pork_cost - 2
  let beef_cost : ℝ := chicken_cost + 4
  let lamb_cost : ℝ := pork_cost + 3
  let meats : List Meat := [
    { name := "Chicken", cost := chicken_cost, quantity := 3.5 },
    { name := "Pork", cost := pork_cost, quantity := 1.2 },
    { name := "Beef", cost := beef_cost, quantity := 2.3 },
    { name := "Lamb", cost := lamb_cost, quantity := 0.8 }
  ]
  totalCost meats = 46.8 := by
  sorry

end meat_purchase_cost_l4041_404190


namespace health_run_distance_to_finish_l4041_404196

/-- The distance between a runner and the finish line in a health run event -/
def distance_to_finish (total_distance : ℝ) (speed : ℝ) (time : ℝ) : ℝ :=
  total_distance - speed * time

/-- Theorem: In a 7.5 km health run, after running for 10 minutes at speed x km/min, 
    the distance to the finish line is 7.5 - 10x km -/
theorem health_run_distance_to_finish (x : ℝ) : 
  distance_to_finish 7.5 x 10 = 7.5 - 10 * x := by
  sorry

end health_run_distance_to_finish_l4041_404196


namespace only_lottery_is_random_l4041_404180

-- Define the events
inductive Event
| BasketballFall
| LotteryWin
| BirthdayMatch
| DrawBlackBall

-- Define the properties of events
def isCertain (e : Event) : Prop :=
  match e with
  | Event.BasketballFall => true
  | _ => false

def isImpossible (e : Event) : Prop :=
  match e with
  | Event.DrawBlackBall => true
  | _ => false

def isRandom (e : Event) : Prop :=
  ¬(isCertain e) ∧ ¬(isImpossible e)

-- Define the given conditions
axiom gravity_exists : isCertain Event.BasketballFall
axiom pigeonhole_principle : isCertain Event.BirthdayMatch
axiom bag_contents : isImpossible Event.DrawBlackBall

-- State the theorem
theorem only_lottery_is_random :
  ∀ e : Event, isRandom e ↔ e = Event.LotteryWin :=
sorry

end only_lottery_is_random_l4041_404180


namespace bg_length_is_two_l4041_404183

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 0 ∧ B.2 = Real.sqrt 12 ∧ C.1 = 2 ∧ C.2 = 0

-- Define the square BDEC
def Square (B D E C : ℝ × ℝ) : Prop :=
  (D.1 - B.1)^2 + (D.2 - B.2)^2 = (E.1 - D.1)^2 + (E.2 - D.2)^2 ∧
  (E.1 - D.1)^2 + (E.2 - D.2)^2 = (C.1 - E.1)^2 + (C.2 - E.2)^2 ∧
  (C.1 - E.1)^2 + (C.2 - E.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

-- Define the center of the square
def CenterOfSquare (F B C : ℝ × ℝ) : Prop :=
  F.1 = (B.1 + C.1) / 2 ∧ F.2 = (B.2 + C.2) / 2

-- Define the intersection point G
def Intersection (A F B C G : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, G.1 = t * (F.1 - A.1) + A.1 ∧ G.2 = t * (F.2 - A.2) + A.2 ∧
  G.1 = B.1 + (C.1 - B.1) * ((G.2 - B.2) / (C.2 - B.2))

-- Main theorem
theorem bg_length_is_two 
  (A B C D E F G : ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : Square B D E C) 
  (h3 : CenterOfSquare F B C) 
  (h4 : Intersection A F B C G) : 
  (G.1 - B.1)^2 + (G.2 - B.2)^2 = 4 := by
  sorry

end bg_length_is_two_l4041_404183


namespace problem_solution_l4041_404123

theorem problem_solution (a b c : ℝ) 
  (h1 : ∀ x, (x - a) * (x - b) / (x - c) ≤ 0 ↔ x < -2 ∨ |x - 30| ≤ 2)
  (h2 : a < b) : 
  a + 2*b + 3*c = 86 := by
sorry

end problem_solution_l4041_404123


namespace problem_solution_l4041_404105

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x^a

-- Define an odd function
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the specific function f(x) = lg(x + √(x^2 + 1))
noncomputable def f (x : ℝ) : ℝ := lg (x + Real.sqrt (x^2 + 1))

theorem problem_solution :
  (isPowerFunction (λ _ : ℝ => 1)) ∧
  (∀ g : ℝ → ℝ, isOddFunction g → g 0 = 0) ∧
  (isOddFunction f) ∧
  (∃ a : ℝ, a < 0 ∧ (a^2)^(3/2) ≠ a^3) ∧
  (∃! x : ℝ, (λ _ : ℝ => 1) x = 0 → False) :=
by sorry

end problem_solution_l4041_404105


namespace parabola_sum_property_l4041_404121

/-- Represents a quadratic function of the form ax^2 + bx + c --/
structure Quadratic where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The function resulting from reflecting a quadratic about the y-axis --/
def reflect (q : Quadratic) : Quadratic :=
  { a := q.a, b := -q.b, c := q.c }

/-- Vertical translation of a quadratic function --/
def translate (q : Quadratic) (d : ℝ) : Quadratic :=
  { a := q.a, b := q.b, c := q.c + d }

/-- The sum of two quadratic functions --/
def add (q1 q2 : Quadratic) : Quadratic :=
  { a := q1.a + q2.a, b := q1.b + q2.b, c := q1.c + q2.c }

theorem parabola_sum_property (q : Quadratic) :
  let f := translate q 4
  let g := translate (reflect q) (-4)
  (add f g).b = 0 := by sorry

end parabola_sum_property_l4041_404121


namespace max_k_value_l4041_404159

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + a*x + 2 * log x

noncomputable def g (k : ℤ) (x : ℝ) : ℝ := (1/2) * x^2 + k*x + (2-x) * log x - k

theorem max_k_value :
  ∃ (k_max : ℤ),
    (∀ (k : ℤ), (∀ (x : ℝ), x > 1 → g k x < f 1 x) → k ≤ k_max) ∧
    (∀ (x : ℝ), x > 1 → g k_max x < f 1 x) ∧
    k_max = 3 :=
sorry

end max_k_value_l4041_404159


namespace min_value_expression_l4041_404191

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxy : x * y = 1) :
  (x + 2 * y) * (2 * x + z) * (y + 2 * z) ≥ 48 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ = 1 ∧
    (x₀ + 2 * y₀) * (2 * x₀ + z₀) * (y₀ + 2 * z₀) = 48 :=
by sorry

end min_value_expression_l4041_404191


namespace smallest_ends_in_9_divisible_by_13_l4041_404164

/-- A positive integer that ends in 9 -/
def EndsIn9 (n : ℕ) : Prop := n % 10 = 9 ∧ n > 0

/-- The smallest positive integer that ends in 9 and is divisible by 13 -/
def SmallestEndsIn9DivisibleBy13 : ℕ := 99

theorem smallest_ends_in_9_divisible_by_13 :
  EndsIn9 SmallestEndsIn9DivisibleBy13 ∧
  SmallestEndsIn9DivisibleBy13 % 13 = 0 ∧
  ∀ n : ℕ, EndsIn9 n ∧ n % 13 = 0 → n ≥ SmallestEndsIn9DivisibleBy13 := by
  sorry

end smallest_ends_in_9_divisible_by_13_l4041_404164


namespace smallest_number_divisible_by_all_l4041_404132

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 5) % 12 = 0 ∧
  (n - 5) % 16 = 0 ∧
  (n - 5) % 18 = 0 ∧
  (n - 5) % 21 = 0 ∧
  (n - 5) % 28 = 0

theorem smallest_number_divisible_by_all :
  ∀ m : ℕ, m < 1013 → ¬(is_divisible_by_all m) ∧ is_divisible_by_all 1013 :=
by sorry

end smallest_number_divisible_by_all_l4041_404132


namespace minimum_number_of_boys_l4041_404185

theorem minimum_number_of_boys (k : ℕ) (n m : ℕ) : 
  (k > 0) →  -- total number of apples is positive
  (n > 0) →  -- there is at least one boy who collected 10 apples
  (m > 0) →  -- there is at least one boy who collected 10% of apples
  (100 * n + m * k = 10 * k) →  -- equation representing total apples collected
  (n + m ≥ 6) →  -- total number of boys is at least 6
  ∀ (n' m' : ℕ), (n' > 0) → (m' > 0) → 
    (∃ (k' : ℕ), k' > 0 ∧ 100 * n' + m' * k' = 10 * k') → 
    (n' + m' ≥ 6) :=
by
  sorry

#check minimum_number_of_boys

end minimum_number_of_boys_l4041_404185


namespace equilateral_triangle_intersection_l4041_404141

/-- Given a right triangular prism with base edges a, b, and c,
    when intersected by a plane to form an equilateral triangle with side length d,
    prove that d satisfies the equation: 3d^4 - 100d^2 + 576 = 0 -/
theorem equilateral_triangle_intersection (a b c d : ℝ) : 
  a = 3 → b = 4 → c = 5 → 
  3 * d^4 - 100 * d^2 + 576 = 0 := by
  sorry

end equilateral_triangle_intersection_l4041_404141


namespace middle_number_proof_l4041_404186

theorem middle_number_proof (x y z : ℕ) (h1 : x < y) (h2 : y < z) 
  (h3 : x + y = 16) (h4 : x + z = 21) (h5 : y + z = 23) : y = 9 := by
  sorry

end middle_number_proof_l4041_404186


namespace trapezoid_areas_l4041_404171

/-- Represents a trapezoid with given dimensions and a parallel line through the intersection of diagonals -/
structure Trapezoid :=
  (ad : ℝ) -- Length of base AD
  (bc : ℝ) -- Length of base BC
  (ab : ℝ) -- Length of side AB
  (cd : ℝ) -- Length of side CD

/-- Calculates the areas of the two resulting trapezoids formed by a line parallel to the bases through the diagonal intersection point -/
def calculate_areas (t : Trapezoid) : ℝ × ℝ := sorry

/-- Theorem stating the areas of the resulting trapezoids for the given dimensions -/
theorem trapezoid_areas (t : Trapezoid) 
  (h1 : t.ad = 84) (h2 : t.bc = 42) (h3 : t.ab = 39) (h4 : t.cd = 45) : 
  calculate_areas t = (588, 1680) := by sorry

end trapezoid_areas_l4041_404171


namespace ploughing_time_l4041_404162

/-- Given two workers R and S who can plough a field together in 10 hours,
    and R alone can plough the field in 15 hours,
    prove that S alone would take 30 hours to plough the same field. -/
theorem ploughing_time (r s : ℝ) : 
  (r + s = 1 / 10) →  -- R and S together take 10 hours
  (r = 1 / 15) →      -- R alone takes 15 hours
  (s = 1 / 30) :=     -- S alone takes 30 hours
by sorry

end ploughing_time_l4041_404162


namespace exponent_multiplication_l4041_404100

theorem exponent_multiplication (a : ℝ) (m n : ℕ) : a ^ m * a ^ n = a ^ (m + n) := by
  sorry

end exponent_multiplication_l4041_404100


namespace cos_x_plus_2y_equals_one_l4041_404153

-- Define the variables and conditions
variable (x y a : ℝ)
variable (h1 : x * y ∈ Set.Icc (-π/4) (π/4))
variable (h2 : x^3 + Real.sin x - 2*a = 0)
variable (h3 : 4*y^3 + (1/2) * Real.sin (2*y) - a = 0)

-- State the theorem
theorem cos_x_plus_2y_equals_one : Real.cos (x + 2*y) = 1 := by
  sorry

end cos_x_plus_2y_equals_one_l4041_404153


namespace quadratic_through_origin_l4041_404158

/-- A quadratic function passing through the origin -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ :=
  (m - 2) * x^2 - 4 * x + m^2 + 2 * m - 8

/-- The theorem stating that if the quadratic function passes through the origin, then m = -4 -/
theorem quadratic_through_origin (m : ℝ) :
  (∀ x, quadratic_function m x = 0 → x = 0) →
  m = -4 := by
  sorry

end quadratic_through_origin_l4041_404158


namespace new_person_weight_l4041_404117

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 4.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 101 :=
by
  sorry

end new_person_weight_l4041_404117


namespace fixed_point_power_function_l4041_404154

-- Define the conditions
variable (a : ℝ) (ha : a > 0) (hna : a ≠ 1)
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (hf1 : f (Real.sqrt 2) = 2)
variable (hf2 : ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α)

-- State the theorem
theorem fixed_point_power_function : f 3 = 9 := by
  sorry

end fixed_point_power_function_l4041_404154


namespace count_integers_satisfying_inequality_l4041_404165

/-- The number of positive integers satisfying the inequality -/
def count_satisfying_integers : ℕ := 8

/-- The inequality function -/
def inequality (n : ℤ) : Prop :=
  (n + 7) * (n - 4) * (n - 10) < 0

theorem count_integers_satisfying_inequality :
  (∃ (S : Finset ℤ), S.card = count_satisfying_integers ∧
    (∀ n ∈ S, n > 0 ∧ inequality n) ∧
    (∀ n : ℤ, n > 0 → inequality n → n ∈ S)) :=
sorry

end count_integers_satisfying_inequality_l4041_404165


namespace intersection_length_l4041_404177

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 5
def circle2 (x y m : ℝ) : Prop := (x - m)^2 + y^2 = 20

-- Define the intersection points
def intersectionPoints (m : ℝ) : Prop := ∃ (A B : ℝ × ℝ), 
  circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 A.1 A.2 m ∧ circle2 B.1 B.2 m

-- Define the perpendicular tangents condition
def perpendicularTangents (m : ℝ) (A : ℝ × ℝ) : Prop :=
  circle1 A.1 A.2 ∧ circle2 A.1 A.2 m ∧
  (A.1 * m = 5) -- This condition represents perpendicular tangents

-- Theorem statement
theorem intersection_length (m : ℝ) :
  intersectionPoints m →
  (∃ (A : ℝ × ℝ), perpendicularTangents m A) →
  (∃ (A B : ℝ × ℝ), circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧ 
                     circle2 A.1 A.2 m ∧ circle2 B.1 B.2 m ∧
                     ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16) :=
by sorry

end intersection_length_l4041_404177


namespace division_remainder_proof_l4041_404175

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) 
  (h1 : dividend = 725)
  (h2 : divisor = 36)
  (h3 : quotient = 20) :
  dividend = divisor * quotient + 5 := by
sorry

end division_remainder_proof_l4041_404175


namespace minimum_travel_cost_l4041_404106

-- Define the cities and distances
def X : City := sorry
def Y : City := sorry
def Z : City := sorry

-- Define the distances
def distance_XY : ℝ := 5000
def distance_XZ : ℝ := 4000

-- Define the cost functions
def bus_cost (distance : ℝ) : ℝ := 0.2 * distance
def plane_cost (distance : ℝ) : ℝ := 150 + 0.15 * distance

-- Define the theorem
theorem minimum_travel_cost :
  ∃ (cost : ℝ),
    cost = plane_cost distance_XY + 
           plane_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
           plane_cost distance_XZ ∧
    cost = 2250 ∧
    ∀ (alternative_cost : ℝ),
      (alternative_cost = bus_cost distance_XY + 
                          bus_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          bus_cost distance_XZ ∨
       alternative_cost = plane_cost distance_XY + 
                          bus_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          bus_cost distance_XZ ∨
       alternative_cost = bus_cost distance_XY + 
                          plane_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          bus_cost distance_XZ ∨
       alternative_cost = bus_cost distance_XY + 
                          bus_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          plane_cost distance_XZ ∨
       alternative_cost = plane_cost distance_XY + 
                          plane_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          bus_cost distance_XZ ∨
       alternative_cost = plane_cost distance_XY + 
                          bus_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          plane_cost distance_XZ ∨
       alternative_cost = bus_cost distance_XY + 
                          plane_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          plane_cost distance_XZ) →
      cost ≤ alternative_cost :=
by sorry

end minimum_travel_cost_l4041_404106


namespace clock_equivalent_square_l4041_404143

theorem clock_equivalent_square : ∃ (n : ℕ), n > 9 ∧ n ≤ 13 ∧ (n ^ 2 - n) % 12 = 0 ∧ ∀ (m : ℕ), m > 9 ∧ m < n → (m ^ 2 - m) % 12 ≠ 0 := by
  sorry

end clock_equivalent_square_l4041_404143


namespace remainder_problem_l4041_404178

theorem remainder_problem (N : ℤ) : 
  (∃ k : ℤ, N = 45 * k + 31) → (∃ m : ℤ, N = 15 * m + 1) :=
by sorry

end remainder_problem_l4041_404178


namespace overlapping_part_length_l4041_404125

/-- Given three wooden planks of equal length and a total fence length,
    calculate the length of one overlapping part. -/
theorem overlapping_part_length
  (plank_length : ℝ)
  (num_planks : ℕ)
  (fence_length : ℝ)
  (h1 : plank_length = 217)
  (h2 : num_planks = 3)
  (h3 : fence_length = 627)
  (h4 : num_planks > 1) :
  let overlap_length := (num_planks * plank_length - fence_length) / (num_planks - 1)
  overlap_length = 12 := by
sorry

end overlapping_part_length_l4041_404125


namespace pizza_slice_volume_l4041_404163

/-- The volume of a pizza slice -/
theorem pizza_slice_volume (thickness : ℝ) (diameter : ℝ) (num_pieces : ℕ) :
  thickness = 1/2 →
  diameter = 18 →
  num_pieces = 16 →
  (π * (diameter/2)^2 * thickness) / num_pieces = 2.53125 * π := by
  sorry

end pizza_slice_volume_l4041_404163


namespace successive_discounts_equivalence_l4041_404179

/-- Proves that two successive discounts are equivalent to a single discount --/
theorem successive_discounts_equivalence (original_price : ℝ) 
  (first_discount second_discount : ℝ) :
  original_price = 800 ∧ 
  first_discount = 0.15 ∧ 
  second_discount = 0.10 →
  let price_after_first := original_price * (1 - first_discount)
  let final_price := price_after_first * (1 - second_discount)
  let equivalent_discount := (original_price - final_price) / original_price
  equivalent_discount = 0.235 := by
  sorry

end successive_discounts_equivalence_l4041_404179


namespace largest_covered_range_l4041_404169

def is_monic_quadratic (p : ℤ → ℤ) : Prop :=
  ∃ a b : ℤ, ∀ x, p x = x^2 + a*x + b

def covers_range (p₁ p₂ p₃ : ℤ → ℤ) (n : ℕ) : Prop :=
  ∀ i ∈ Finset.range n, ∃ j ∈ [1, 2, 3], ∃ m : ℤ, 
    (if j = 1 then p₁ else if j = 2 then p₂ else p₃) m = i + 1

theorem largest_covered_range : 
  (∃ p₁ p₂ p₃ : ℤ → ℤ, 
    is_monic_quadratic p₁ ∧ 
    is_monic_quadratic p₂ ∧ 
    is_monic_quadratic p₃ ∧ 
    covers_range p₁ p₂ p₃ 9) ∧ 
  (∀ n > 9, ¬∃ p₁ p₂ p₃ : ℤ → ℤ, 
    is_monic_quadratic p₁ ∧ 
    is_monic_quadratic p₂ ∧ 
    is_monic_quadratic p₃ ∧ 
    covers_range p₁ p₂ p₃ n) :=
by sorry

end largest_covered_range_l4041_404169


namespace grain_remaining_after_crash_l4041_404134

/-- The amount of grain remaining onboard after a ship crash -/
def remaining_grain (original : ℕ) (spilled : ℕ) : ℕ :=
  original - spilled

/-- Theorem stating the amount of grain remaining onboard after the specific crash -/
theorem grain_remaining_after_crash : 
  remaining_grain 50870 49952 = 918 := by
  sorry

end grain_remaining_after_crash_l4041_404134


namespace last_four_digits_of_2_to_1965_l4041_404110

theorem last_four_digits_of_2_to_1965 : 2^1965 % 10000 = 3125 := by
  sorry

end last_four_digits_of_2_to_1965_l4041_404110


namespace distance_relationships_l4041_404148

structure Distance where
  r : ℝ
  a : ℝ
  b : ℝ

def perpendicular_to_x12 (d : Distance) : Prop := sorry
def parallel_to_H (d : Distance) : Prop := sorry
def parallel_to_P1 (d : Distance) : Prop := sorry
def perpendicular_to_H (d : Distance) : Prop := sorry
def perpendicular_to_P1 (d : Distance) : Prop := sorry
def parallel_to_x12 (d : Distance) : Prop := sorry

theorem distance_relationships (d : Distance) :
  (∃ α β : ℝ, d.a = d.r * Real.cos α ∧ d.b = d.r * Real.cos β) ∧
  (perpendicular_to_x12 d → d.a^2 + d.b^2 = d.r^2) ∧
  (parallel_to_H d → d.a = d.b) ∧
  (parallel_to_P1 d → d.a = d.r ∧ ∃ β : ℝ, d.b = d.a * Real.cos β) ∧
  (perpendicular_to_H d → d.a = d.b ∧ d.a = d.r * Real.sqrt 2 / 2) ∧
  (perpendicular_to_P1 d → d.a = 0 ∧ d.b = d.r) ∧
  (parallel_to_x12 d → d.a = d.b ∧ d.a = d.r) :=
by sorry

end distance_relationships_l4041_404148


namespace two_points_l4041_404144

/-- The number of integer points satisfying the given equation and conditions -/
def num_points : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    let x := p.1
    let y := p.2
    x > 0 ∧ y > 0 ∧ x > y ∧ y * (x - 1) = 2 * x + 2018
  ) (Finset.product (Finset.range 10000) (Finset.range 10000))).card

/-- Theorem stating that there are exactly two points satisfying the conditions -/
theorem two_points : num_points = 2 := by
  sorry

end two_points_l4041_404144


namespace geometric_sequence_third_term_l4041_404150

/-- Given a geometric sequence {aₙ} where a₁ + a₂ = 3 and a₂ + a₃ = 6, prove that a₃ = 4 -/
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)  -- a is a sequence of real numbers
  (h_geom : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1))  -- a is a geometric sequence
  (h_sum1 : a 1 + a 2 = 3)  -- first condition
  (h_sum2 : a 2 + a 3 = 6)  -- second condition
  : a 3 = 4 := by
  sorry

end geometric_sequence_third_term_l4041_404150


namespace vector_magnitude_l4041_404103

/-- Given vectors a and b, if c is parallel to a and perpendicular to b + c, 
    then the magnitude of c is 3√2. -/
theorem vector_magnitude (a b c : ℝ × ℝ) : 
  a = (-1, 1) → 
  b = (-2, 4) → 
  (∃ k : ℝ, c = k • a) →  -- parallel condition
  (a.1 * (b.1 + c.1) + a.2 * (b.2 + c.2) = 0) →  -- perpendicular condition
  ‖c‖ = 3 * Real.sqrt 2 := by
  sorry

end vector_magnitude_l4041_404103


namespace initial_charge_calculation_l4041_404151

/-- A taxi company's pricing model -/
structure TaxiPricing where
  initial_charge : ℝ  -- Charge for the first 1/5 mile
  additional_charge : ℝ  -- Charge for each additional 1/5 mile
  total_charge : ℝ  -- Total charge for a specific ride
  ride_distance : ℝ  -- Distance of the ride in miles

/-- Theorem stating the initial charge for the first 1/5 mile -/
theorem initial_charge_calculation (tp : TaxiPricing) 
  (h1 : tp.additional_charge = 0.40)
  (h2 : tp.total_charge = 18.40)
  (h3 : tp.ride_distance = 8) :
  tp.initial_charge = 2.80 := by
  sorry

end initial_charge_calculation_l4041_404151


namespace old_cards_count_l4041_404170

def cards_per_page : ℕ := 3
def new_cards : ℕ := 8
def total_pages : ℕ := 6

theorem old_cards_count : 
  (total_pages * cards_per_page) - new_cards = 10 := by
  sorry

end old_cards_count_l4041_404170


namespace probability_product_eight_l4041_404140

/-- A standard 6-sided die -/
def StandardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The sample space of rolling a standard 6-sided die twice -/
def TwoRollsSampleSpace : Finset (ℕ × ℕ) :=
  (StandardDie.product StandardDie)

/-- The favorable outcomes where the product of two rolls is 8 -/
def FavorableOutcomes : Finset (ℕ × ℕ) :=
  {(2, 4), (4, 2)}

/-- Probability of the product of two rolls being 8 -/
theorem probability_product_eight :
  (FavorableOutcomes.card : ℚ) / TwoRollsSampleSpace.card = 1 / 18 :=
sorry

end probability_product_eight_l4041_404140


namespace road_trip_total_hours_l4041_404176

/-- Calculates the total hours driven during a road trip -/
def total_hours_driven (days : ℕ) (hours_per_day_person1 : ℕ) (hours_per_day_person2 : ℕ) : ℕ :=
  days * (hours_per_day_person1 + hours_per_day_person2)

/-- Proves that the total hours driven in the given scenario is 42 -/
theorem road_trip_total_hours : total_hours_driven 3 8 6 = 42 := by
  sorry

end road_trip_total_hours_l4041_404176


namespace eleven_divides_difference_l4041_404198

/-- Represents a three-digit number ABC where A, B, and C are distinct digits and A ≠ 0 -/
structure ThreeDigitNumber where
  A : Nat
  B : Nat
  C :Nat
  h1 : A ≠ 0
  h2 : A < 10
  h3 : B < 10
  h4 : C < 10
  h5 : A ≠ B
  h6 : B ≠ C
  h7 : A ≠ C

/-- Converts a ThreeDigitNumber to its numerical value -/
def toNumber (n : ThreeDigitNumber) : Nat :=
  100 * n.A + 10 * n.B + n.C

/-- Reverses a ThreeDigitNumber -/
def reverse (n : ThreeDigitNumber) : Nat :=
  100 * n.C + 10 * n.B + n.A

theorem eleven_divides_difference (n : ThreeDigitNumber) :
  11 ∣ (toNumber n - reverse n) := by
  sorry

#check eleven_divides_difference

end eleven_divides_difference_l4041_404198


namespace inverse_undefined_at_one_l4041_404174

/-- Given a function g(x) = (x - 5) / (x - 6), prove that its inverse g⁻¹(x) is undefined when x = 1 -/
theorem inverse_undefined_at_one (g : ℝ → ℝ) (h : ∀ x, g x = (x - 5) / (x - 6)) :
  ¬∃ y, g y = 1 := by
sorry

end inverse_undefined_at_one_l4041_404174


namespace melted_spheres_radius_l4041_404119

theorem melted_spheres_radius (r : ℝ) : r > 0 → (4 / 3 * Real.pi * r^3 = 8 / 3 * Real.pi) → r = 2^(1/3) := by
  sorry

end melted_spheres_radius_l4041_404119


namespace price_increase_percentage_l4041_404107

theorem price_increase_percentage (lower_price higher_price : ℝ) 
  (h1 : lower_price > 0)
  (h2 : higher_price > lower_price)
  (h3 : higher_price = lower_price * 1.4) :
  (higher_price - lower_price) / lower_price * 100 = 40 := by
sorry

end price_increase_percentage_l4041_404107


namespace arithmetic_sequence_difference_l4041_404182

/-- An arithmetic sequence with sum Sn for first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function

/-- The common difference of the arithmetic sequence is -2 -/
theorem arithmetic_sequence_difference (seq : ArithmeticSequence) 
  (h1 : seq.S 3 = 6)
  (h2 : seq.a 3 = 0) : 
  seq.d = -2 := by
sorry

end arithmetic_sequence_difference_l4041_404182


namespace cube_surface_area_l4041_404118

theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 1) : 
  6 * edge_length^2 = 6 := by
  sorry

end cube_surface_area_l4041_404118


namespace one_positive_real_solution_l4041_404195

def f (x : ℝ) := x^4 + 8*x^3 + 16*x^2 + 2023*x - 2023

theorem one_positive_real_solution :
  ∃! x : ℝ, x > 0 ∧ x^10 + 8*x^9 + 16*x^8 + 2023*x^7 - 2023*x^6 = 0 :=
by
  sorry

end one_positive_real_solution_l4041_404195


namespace f_simplification_g_definition_g_value_at_pi_over_6_l4041_404146

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * sin (π - x) * sin x - (sin x - cos x)^2

noncomputable def g (x : ℝ) : ℝ := 2 * sin x + Real.sqrt 3 - 1

theorem f_simplification (x : ℝ) : f x = 2 * sin (2*x - π/3) + Real.sqrt 3 - 1 := by sorry

theorem g_definition (x : ℝ) : g x = 2 * sin x + Real.sqrt 3 - 1 := by sorry

theorem g_value_at_pi_over_6 : g (π/6) = Real.sqrt 3 := by sorry

end f_simplification_g_definition_g_value_at_pi_over_6_l4041_404146


namespace initial_nurses_count_l4041_404114

/-- Proves that the initial number of nurses is 18 given the conditions of the problem -/
theorem initial_nurses_count (initial_doctors : ℕ) (quit_doctors quit_nurses remaining_staff : ℕ) 
  (h1 : initial_doctors = 11)
  (h2 : quit_doctors = 5)
  (h3 : quit_nurses = 2)
  (h4 : remaining_staff = 22)
  (h5 : initial_doctors - quit_doctors + (initial_nurses - quit_nurses) = remaining_staff) :
  initial_nurses = 18 :=
by
  sorry
where
  initial_nurses : ℕ := by sorry

end initial_nurses_count_l4041_404114


namespace smallest_n_divisible_by_ten_l4041_404152

def A (n : ℕ) : ℕ := (List.range n).foldl (λ acc k => acc * Nat.choose (k^2) k) 1

theorem smallest_n_divisible_by_ten : 
  (∀ m < 4, ¬(10 ∣ A m)) ∧ (10 ∣ A 4) := by sorry

end smallest_n_divisible_by_ten_l4041_404152


namespace machine_does_not_require_repair_l4041_404113

/-- Represents a weighing machine for food portions --/
structure WeighingMachine where
  max_deviation : ℝ
  nominal_mass : ℝ
  unreadable_deviation_bound : ℝ

/-- Determines if a weighing machine requires repair --/
def requires_repair (m : WeighingMachine) : Prop :=
  m.max_deviation > 0.1 * m.nominal_mass ∨ 
  m.max_deviation < m.unreadable_deviation_bound

/-- Theorem: The weighing machine does not require repair --/
theorem machine_does_not_require_repair (m : WeighingMachine) 
  (h1 : m.max_deviation = 37)
  (h2 : m.max_deviation ≤ 0.1 * m.nominal_mass)
  (h3 : m.unreadable_deviation_bound < m.max_deviation) :
  ¬(requires_repair m) := by
  sorry

#check machine_does_not_require_repair

end machine_does_not_require_repair_l4041_404113


namespace lisas_number_l4041_404109

theorem lisas_number (n : ℕ) : 
  (∃ k : ℕ, n = 150 * k) ∧ 
  (∃ m : ℕ, n = 45 * m) ∧ 
  1000 ≤ n ∧ n < 3000 ∧
  (∀ x : ℕ, (∃ i : ℕ, x = 150 * i) ∧ (∃ j : ℕ, x = 45 * j) ∧ 1000 ≤ x ∧ x < 3000 → n ≤ x) →
  n = 1350 := by
sorry

end lisas_number_l4041_404109


namespace vieta_relation_l4041_404115

/-- The quadratic equation x^2 - x - 1 = 0 --/
def quadratic_equation (x : ℝ) : Prop := x^2 - x - 1 = 0

/-- Definition of S_n --/
def S (n : ℕ) (M N : ℝ) : ℝ := M^n + N^n

/-- Theorem: Relationship between S_n, S_{n-1}, and S_{n-2} --/
theorem vieta_relation (M N : ℝ) (h : quadratic_equation M ∧ quadratic_equation N) :
  ∀ n ≥ 3, S n M N = S (n-1) M N + S (n-2) M N :=
sorry

end vieta_relation_l4041_404115


namespace percent_of_y_l4041_404155

theorem percent_of_y (y : ℝ) (h : y > 0) : ((8 * y) / 20 + (3 * y) / 10) / y = 0.7 := by
  sorry

end percent_of_y_l4041_404155


namespace function_periodic_l4041_404129

/-- A function satisfying the given conditions is periodic with period 1 -/
theorem function_periodic (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, |f x| ≤ 1)
  (h2 : ∀ x : ℝ, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  ∀ x : ℝ, f (x + 1) = f x := by
  sorry

end function_periodic_l4041_404129


namespace inequality_solution_set_l4041_404120

theorem inequality_solution_set (a b m : ℝ) : 
  (∀ x, x^2 - a*x - 2 > 0 ↔ (x < -1 ∨ x > b)) →
  b > -1 →
  m > -1/2 →
  (a = 1 ∧ b = 2) ∧
  (
    (m > 0 → ∀ x, (m*x + a)*(x - b) > 0 ↔ (x < -1/m ∨ x > 2)) ∧
    (m = 0 → ∀ x, (m*x + a)*(x - b) > 0 ↔ x > 2) ∧
    (-1/2 < m ∧ m < 0 → ∀ x, (m*x + a)*(x - b) > 0 ↔ (2 < x ∧ x < -1/m))
  ) := by sorry

end inequality_solution_set_l4041_404120


namespace henrys_score_l4041_404122

theorem henrys_score (june patty josh henry : ℕ) : 
  june = 97 → patty = 85 → josh = 100 →
  (june + patty + josh + henry) / 4 = 94 →
  henry = 94 := by
sorry

end henrys_score_l4041_404122


namespace scale_division_l4041_404133

/-- Given a scale of length 188 inches divided into 8 equal parts, 
    the length of each part is 23.5 inches. -/
theorem scale_division (total_length : ℝ) (num_parts : ℕ) 
  (h1 : total_length = 188) 
  (h2 : num_parts = 8) :
  total_length / num_parts = 23.5 := by
  sorry

end scale_division_l4041_404133


namespace inscribed_circle_radius_l4041_404187

/-- The radius of a circle inscribed in a rectangle and tangent to four circles -/
theorem inscribed_circle_radius (AB BC : ℝ) (h_AB : AB = 8) (h_BC : BC = 6) : ∃ r : ℝ,
  r > 0 ∧ r < 6 ∧
  (r + 4)^2 = r^2 + r^2 ∧
  (r + 3)^2 = (8 - r)^2 + r^2 ∧
  r = 11 - Real.sqrt 66 := by
sorry

end inscribed_circle_radius_l4041_404187


namespace optimal_fraction_sum_l4041_404199

theorem optimal_fraction_sum (A B C D : ℕ) : 
  (A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9) →  -- A, B, C, D are digits
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →  -- A, B, C, D are different
  (C + D ≥ 5) →  -- C + D is at least 5
  (∃ k : ℕ, k * (C + D) = A + B) →  -- (A+B)/(C+D) is an integer
  (A + B ≤ 14) :=  -- The maximum possible value of A+B is 14
by sorry

end optimal_fraction_sum_l4041_404199


namespace trig_identity_l4041_404147

theorem trig_identity : 
  Real.sin (47 * π / 180) * Real.cos (17 * π / 180) - 
  Real.cos (47 * π / 180) * Real.cos (73 * π / 180) = 1/2 := by
  sorry

end trig_identity_l4041_404147


namespace largest_solution_is_three_l4041_404137

theorem largest_solution_is_three :
  let f (x : ℝ) := (15 * x^2 - 40 * x + 18) / (4 * x - 3) + 7 * x
  ∃ (max : ℝ), max = 3 ∧ 
    (∀ x : ℝ, f x = 8 * x - 2 → x ≤ max) ∧
    (f max = 8 * max - 2) := by
  sorry

end largest_solution_is_three_l4041_404137


namespace rectangular_parallelepiped_surface_area_equals_volume_l4041_404142

theorem rectangular_parallelepiped_surface_area_equals_volume :
  ∃ (a b c : ℕ+), 2 * (a * b + b * c + a * c) = a * b * c :=
sorry

end rectangular_parallelepiped_surface_area_equals_volume_l4041_404142


namespace value_of_expression_l4041_404189

theorem value_of_expression (x : ℝ) (h : x = 5) : 3 * x^2 + 2 = 77 := by
  sorry

end value_of_expression_l4041_404189


namespace fifteenth_student_age_l4041_404127

theorem fifteenth_student_age
  (total_students : Nat)
  (average_age : ℝ)
  (group1_count : Nat)
  (group1_average : ℝ)
  (group2_count : Nat)
  (group2_average : ℝ)
  (h1 : total_students = 15)
  (h2 : average_age = 15)
  (h3 : group1_count = 4)
  (h4 : group1_average = 14)
  (h5 : group2_count = 9)
  (h6 : group2_average = 16)
  (h7 : group1_count + group2_count + 1 = total_students) :
  ∃ (fifteenth_age : ℝ),
    fifteenth_age = total_students * average_age - (group1_count * group1_average + group2_count * group2_average) :=
by sorry

end fifteenth_student_age_l4041_404127


namespace division_with_remainder_l4041_404126

theorem division_with_remainder : ∃ (q r : ℤ), 1234567 = 145 * q + r ∧ 0 ≤ r ∧ r < 145 ∧ r = 67 := by
  sorry

end division_with_remainder_l4041_404126


namespace average_equality_implies_z_l4041_404181

theorem average_equality_implies_z (z : ℝ) : 
  (8 + 11 + 20) / 3 = (14 + z) / 2 → z = 12 := by
  sorry

end average_equality_implies_z_l4041_404181


namespace old_clock_slower_l4041_404188

/-- Represents the number of minutes between hand overlaps on the old clock -/
def overlap_interval : ℕ := 66

/-- Represents the number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculates the number of hand overlaps in a 24-hour period -/
def overlaps_per_day : ℕ := hours_per_day - 2

/-- Calculates the total minutes on the old clock for 24 hours -/
def old_clock_minutes : ℕ := overlaps_per_day * overlap_interval

/-- Calculates the total minutes in a standard 24-hour period -/
def standard_clock_minutes : ℕ := hours_per_day * minutes_per_hour

/-- Theorem stating that the old clock is 12 minutes slower over 24 hours -/
theorem old_clock_slower :
  old_clock_minutes - standard_clock_minutes = 12 := by sorry

end old_clock_slower_l4041_404188


namespace max_value_on_interval_solution_set_inequality_l4041_404101

-- Define the function f
def f (x : ℝ) : ℝ := (x + 2) * |x - 2|

-- Theorem for the maximum value of f on [-3, 1]
theorem max_value_on_interval :
  ∃ (m : ℝ), m = 4 ∧ ∀ x ∈ Set.Icc (-3) 1, f x ≤ m :=
sorry

-- Theorem for the solution set of f(x) > 3x
theorem solution_set_inequality :
  {x : ℝ | f x > 3 * x} = {x : ℝ | x > 4 ∨ (-4 < x ∧ x < 1)} :=
sorry

end max_value_on_interval_solution_set_inequality_l4041_404101


namespace marina_olympiad_supplies_l4041_404128

/-- The cost of school supplies for Marina's olympiad participation. -/
def school_supplies_cost 
  (notebook : ℕ) 
  (pencil : ℕ) 
  (eraser : ℕ) 
  (ruler : ℕ) 
  (pen : ℕ) : Prop :=
  notebook = 15 ∧ 
  notebook + pencil + eraser = 47 ∧
  notebook + ruler + pen = 58 →
  notebook + pencil + eraser + ruler + pen = 90

theorem marina_olympiad_supplies : 
  ∃ (notebook pencil eraser ruler pen : ℕ), 
  school_supplies_cost notebook pencil eraser ruler pen :=
sorry

end marina_olympiad_supplies_l4041_404128


namespace max_fourth_term_arithmetic_sequence_l4041_404112

theorem max_fourth_term_arithmetic_sequence (a d : ℕ) (h1 : 0 < a) (h2 : 0 < d) :
  (∀ k : Fin 5, 0 < a + k * d) →
  (5 * a + 10 * d = 75) →
  (∀ a' d' : ℕ, (∀ k : Fin 5, 0 < a' + k * d') → (5 * a' + 10 * d' = 75) → a + 3 * d ≥ a' + 3 * d') →
  a + 3 * d = 22 := by
sorry

end max_fourth_term_arithmetic_sequence_l4041_404112


namespace quadrilateral_area_l4041_404139

/-- Represents a triangle divided into three triangles and one quadrilateral -/
structure DividedTriangle where
  -- Areas of the three triangles
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  -- Area of the quadrilateral
  quad_area : ℝ

/-- Theorem stating that if the areas of the three triangles are 6, 9, and 15,
    then the area of the quadrilateral is 65 -/
theorem quadrilateral_area (t : DividedTriangle) :
  t.area1 = 6 ∧ t.area2 = 9 ∧ t.area3 = 15 → t.quad_area = 65 := by
  sorry

end quadrilateral_area_l4041_404139


namespace cos_120_degrees_l4041_404156

theorem cos_120_degrees : Real.cos (2 * Real.pi / 3) = -1/2 := by sorry

end cos_120_degrees_l4041_404156


namespace correct_sampling_methods_l4041_404157

/-- Represents a sampling method -/
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling
  | SystematicSampling

/-- Represents a region with its number of outlets -/
structure Region where
  name : String
  outlets : Nat

/-- Represents an investigation with its sample size and population size -/
structure Investigation where
  sampleSize : Nat
  populationSize : Nat

/-- The company's sales outlet data -/
def companyData : List Region :=
  [⟨"A", 150⟩, ⟨"B", 120⟩, ⟨"C", 180⟩, ⟨"D", 150⟩]

/-- Total number of outlets -/
def totalOutlets : Nat := (companyData.map Region.outlets).sum

/-- Investigation ① -/
def investigation1 : Investigation :=
  ⟨100, totalOutlets⟩

/-- Investigation ② -/
def investigation2 : Investigation :=
  ⟨7, 10⟩

/-- Determines the appropriate sampling method for an investigation -/
def appropriateSamplingMethod (i : Investigation) : SamplingMethod :=
  sorry

theorem correct_sampling_methods :
  appropriateSamplingMethod investigation1 = SamplingMethod.StratifiedSampling ∧
  appropriateSamplingMethod investigation2 = SamplingMethod.SimpleRandomSampling :=
  sorry

end correct_sampling_methods_l4041_404157


namespace apples_buyers_l4041_404130

theorem apples_buyers (men_apples : ℕ) (women_apples : ℕ) (total_apples : ℕ) :
  men_apples = 30 →
  women_apples = men_apples + 20 →
  total_apples = 210 →
  ∃ (num_men : ℕ), num_men * men_apples + 3 * women_apples = total_apples ∧ num_men = 2 :=
by
  sorry

end apples_buyers_l4041_404130


namespace unique_k_is_zero_l4041_404138

/-- A function f: ℕ → ℕ satisfying f^n(n) = n + k for all n ∈ ℕ, where k is a non-negative integer -/
def SatisfiesCondition (f : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ n : ℕ, (f^[n] n) = n + k

/-- Theorem stating that if a function satisfies the condition, then k must be 0 -/
theorem unique_k_is_zero (f : ℕ → ℕ) (k : ℕ) (h : SatisfiesCondition f k) : k = 0 := by
  sorry

end unique_k_is_zero_l4041_404138


namespace prob_at_least_one_head_correct_l4041_404102

/-- The probability of getting at least one head when tossing 3 coins simultaneously -/
def prob_at_least_one_head : ℚ :=
  7/8

/-- The number of coins being tossed simultaneously -/
def num_coins : ℕ := 3

/-- The probability of getting heads on a single coin toss -/
def prob_heads : ℚ := 1/2

theorem prob_at_least_one_head_correct :
  prob_at_least_one_head = 1 - (1 - prob_heads) ^ num_coins :=
by sorry


end prob_at_least_one_head_correct_l4041_404102


namespace counterexample_exists_l4041_404136

theorem counterexample_exists : ∃ n : ℕ+, ¬(Nat.Prime (6 * n - 1)) ∧ ¬(Nat.Prime (6 * n + 1)) := by
  sorry

end counterexample_exists_l4041_404136


namespace calculation_proof_l4041_404104

theorem calculation_proof : 
  (((15^15 / 15^10)^3 * 5^6) / 25^2) = 3^15 * 5^17 := by
  sorry

end calculation_proof_l4041_404104


namespace modular_congruence_existence_l4041_404194

theorem modular_congruence_existence (a c : ℕ+) (b : ℤ) :
  ∃ x : ℕ+, (c : ℤ) ∣ ((a : ℤ)^(x : ℕ) + x - b) := by
  sorry

end modular_congruence_existence_l4041_404194


namespace abs_nine_sqrt_l4041_404192

theorem abs_nine_sqrt : Real.sqrt (abs (-9)) = 3 := by sorry

end abs_nine_sqrt_l4041_404192


namespace jane_apples_l4041_404166

theorem jane_apples (num_baskets : ℕ) (apples_taken : ℕ) (apples_remaining : ℕ) : 
  num_baskets = 4 → 
  apples_taken = 3 → 
  apples_remaining = 13 → 
  num_baskets * (apples_remaining + apples_taken) = 64 := by
sorry

end jane_apples_l4041_404166


namespace selma_has_50_marbles_l4041_404161

/-- The number of marbles Selma has -/
def selma_marbles (merill_marbles elliot_marbles : ℕ) : ℕ :=
  merill_marbles + elliot_marbles + 5

/-- Theorem stating the number of marbles Selma has -/
theorem selma_has_50_marbles :
  ∀ (merill_marbles elliot_marbles : ℕ),
    merill_marbles = 30 →
    merill_marbles = 2 * elliot_marbles →
    selma_marbles merill_marbles elliot_marbles = 50 :=
by
  sorry

#check selma_has_50_marbles

end selma_has_50_marbles_l4041_404161


namespace unique_three_digit_number_l4041_404145

theorem unique_three_digit_number : ∃! (n : ℕ), 
  (100 ≤ n ∧ n < 1000) ∧ 
  (∃ (π b γ : ℕ), 
    π ≠ b ∧ π ≠ γ ∧ b ≠ γ ∧
    π < 10 ∧ b < 10 ∧ γ < 10 ∧
    n = 100 * π + 10 * b + γ ∧
    n = (π + b + γ) * (π + b + γ + 1)) ∧
  n = 156 := by
sorry

end unique_three_digit_number_l4041_404145


namespace arithmetic_sequence_problem_l4041_404131

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 6 + a 10 = 16)
  (h_a4 : a 4 = 1) :
  a 12 = 15 := by
sorry

end arithmetic_sequence_problem_l4041_404131


namespace current_speed_l4041_404193

/-- Proves that the speed of the current is 8.5 kmph given the specified conditions -/
theorem current_speed (rowing_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  rowing_speed = 9.5 →
  distance = 45.5 →
  time = 9.099272058235341 →
  let downstream_speed := distance / 1000 / (time / 3600)
  downstream_speed = rowing_speed + 8.5 := by
sorry

end current_speed_l4041_404193


namespace greatest_solution_quadratic_l4041_404149

theorem greatest_solution_quadratic : 
  ∃ (x : ℝ), x = 4/5 ∧ 5*x^2 - 3*x - 4 = 0 ∧ 
  ∀ (y : ℝ), 5*y^2 - 3*y - 4 = 0 → y ≤ x :=
by sorry

end greatest_solution_quadratic_l4041_404149


namespace x_squared_eq_one_is_quadratic_l4041_404172

/-- Definition of a quadratic equation in one variable x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c)

/-- The equation x² = 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: x² = 1 is a quadratic equation -/
theorem x_squared_eq_one_is_quadratic : is_quadratic_equation f := by
  sorry


end x_squared_eq_one_is_quadratic_l4041_404172
