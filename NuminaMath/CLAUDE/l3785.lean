import Mathlib

namespace smallest_coprime_to_210_l3785_378501

def is_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem smallest_coprime_to_210 :
  ∀ x : ℕ, x > 1 → x < 11 → ¬(is_relatively_prime x 210) ∧ is_relatively_prime 11 210 :=
by sorry

end smallest_coprime_to_210_l3785_378501


namespace initial_marbles_l3785_378513

/-- Proves that if a person has 7 marbles left after sharing 3 marbles, 
    then they initially had 10 marbles. -/
theorem initial_marbles (shared : ℕ) (left : ℕ) (initial : ℕ) : 
  shared = 3 → left = 7 → initial = shared + left → initial = 10 := by
  sorry

end initial_marbles_l3785_378513


namespace min_value_expression_l3785_378590

theorem min_value_expression (x y : ℝ) :
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 ∧
  ∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 8*x₀ + 6*y₀ + 25 = 0 :=
by
  sorry

end min_value_expression_l3785_378590


namespace remainder_of_large_number_l3785_378561

theorem remainder_of_large_number (n : ℕ) (d : ℕ) (h : n = 123456789012 ∧ d = 210) :
  n % d = 17 := by
  sorry

end remainder_of_large_number_l3785_378561


namespace glorias_turtle_time_l3785_378544

/-- The time it takes for Gloria's turtle to finish the race -/
def glorias_time (gretas_time georges_time : ℕ) : ℕ :=
  2 * georges_time

/-- Theorem stating that Gloria's turtle finished in 8 minutes -/
theorem glorias_turtle_time :
  let gretas_time := 6
  let georges_time := gretas_time - 2
  glorias_time gretas_time georges_time = 8 := by sorry

end glorias_turtle_time_l3785_378544


namespace necessary_not_sufficient_condition_l3785_378512

theorem necessary_not_sufficient_condition (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (∃ y, 0 < y ∧ y < Real.pi / 2 ∧ (Real.sqrt y - 1 / Real.sin y < 0) ∧ ¬(1 / Real.sin y - y > 0)) ∧
  (∀ z, 0 < z ∧ z < Real.pi / 2 ∧ (1 / Real.sin z - z > 0) → (Real.sqrt z - 1 / Real.sin z < 0)) :=
by sorry

end necessary_not_sufficient_condition_l3785_378512


namespace charlies_metal_storage_l3785_378598

/-- The amount of metal Charlie has in storage -/
def metal_in_storage (total_needed : ℕ) (to_buy : ℕ) : ℕ :=
  total_needed - to_buy

/-- Theorem: Charlie's metal in storage is the difference between total needed and amount to buy -/
theorem charlies_metal_storage :
  metal_in_storage 635 359 = 276 := by
  sorry

end charlies_metal_storage_l3785_378598


namespace diamond_computation_l3785_378550

-- Define the ⋄ operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Theorem statement
theorem diamond_computation :
  (diamond (diamond 4 5) 6) - (diamond 4 (diamond 5 6)) = -139 / 870 := by
  sorry

end diamond_computation_l3785_378550


namespace f_properties_l3785_378585

/-- The function f(x) = -x^3 + ax^2 - 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

theorem f_properties (a : ℝ) :
  /- 1. Tangent line equation when a = 3 at x = 1 -/
  (a = 3 → ∃ (m b : ℝ), m = 3 ∧ b = -5 ∧ ∀ x y, y = f 3 x ↔ m*x - y - b = 0) ∧
  /- 2. Monotonicity depends on a -/
  (∃ (x1 x2 : ℝ), x1 < x2 ∧ f a x1 > f a x2) ∧
  (∃ (x3 x4 : ℝ), x3 < x4 ∧ f a x3 < f a x4) ∧
  /- 3. Condition for f(x0) > 0 -/
  (∃ (x0 : ℝ), x0 > 0 ∧ f a x0 > 0) ↔ a > 3 :=
sorry

end f_properties_l3785_378585


namespace simplify_sqrt_expression_l3785_378582

theorem simplify_sqrt_expression : 
  (Real.sqrt 192 / Real.sqrt 27) - (Real.sqrt 500 / Real.sqrt 125) = 2 / 3 := by
  sorry

end simplify_sqrt_expression_l3785_378582


namespace final_price_is_correct_l3785_378542

-- Define the initial price, discounts, and conversion rate
def initial_price : ℝ := 150
def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.15
def discount3 : ℝ := 0.05
def usd_to_inr : ℝ := 75

-- Define the function to calculate the final price
def final_price : ℝ :=
  let price1 := initial_price * (1 - discount1)
  let price2 := price1 * (1 - discount2)
  let price3 := price2 * (1 - discount3)
  price3 * usd_to_inr

-- Theorem statement
theorem final_price_is_correct : final_price = 7267.5 := by
  sorry

end final_price_is_correct_l3785_378542


namespace chord_diameter_relationship_l3785_378516

/-- Represents a sphere with a chord and diameter -/
structure SphereWithChord where
  /-- The radius of the sphere -/
  radius : ℝ
  /-- The length of the chord AB -/
  chord_length : ℝ
  /-- The angle between the chord AB and the diameter CD -/
  angle : ℝ
  /-- The distance from C to A -/
  distance_CA : ℝ

/-- Theorem stating the relationship between the given conditions and BD -/
theorem chord_diameter_relationship (s : SphereWithChord) 
  (h1 : s.radius = 1)
  (h2 : s.chord_length = 1)
  (h3 : s.angle = Real.pi / 3)  -- 60 degrees in radians
  (h4 : s.distance_CA = Real.sqrt 2) :
  ∃ (BD : ℝ), BD = 1 := by
  sorry


end chord_diameter_relationship_l3785_378516


namespace angle_parallel_lines_l3785_378581

-- Define the types for lines and angles
variable (Line : Type) (Angle : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the angle between two lines
variable (angle_between : Line → Line → Angle)

-- Define equality for angles
variable (angle_eq : Angle → Angle → Prop)

-- Theorem statement
theorem angle_parallel_lines 
  (a b c : Line) (θ : Angle)
  (h1 : parallel a b)
  (h2 : angle_eq (angle_between a c) θ) :
  angle_eq (angle_between b c) θ :=
sorry

end angle_parallel_lines_l3785_378581


namespace cherry_pie_count_l3785_378511

theorem cherry_pie_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) 
  (h1 : total_pies = 24)
  (h2 : apple_ratio = 1)
  (h3 : blueberry_ratio = 4)
  (h4 : cherry_ratio = 3) : 
  (total_pies * cherry_ratio) / (apple_ratio + blueberry_ratio + cherry_ratio) = 9 := by
sorry

end cherry_pie_count_l3785_378511


namespace negation_of_positive_product_l3785_378552

theorem negation_of_positive_product (x y : ℝ) :
  ¬(x > 0 ∧ y > 0 → x * y > 0) ↔ (x ≤ 0 ∨ y ≤ 0 → x * y ≤ 0) := by
  sorry

end negation_of_positive_product_l3785_378552


namespace complex_magnitude_l3785_378548

theorem complex_magnitude (z : ℂ) (h : z / (1 - Complex.I)^2 = (1 + Complex.I) / 2) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_l3785_378548


namespace count_numbers_correct_l3785_378537

/-- The count of n-digit numbers composed of digits 1, 2, and 3, where each digit appears at least once -/
def count_numbers (n : ℕ) : ℕ :=
  3^n - 3 * 2^n + 3

/-- Theorem stating that count_numbers gives the correct count -/
theorem count_numbers_correct (n : ℕ) :
  count_numbers n = (3^n : ℕ) - 3 * (2^n : ℕ) + 3 :=
by sorry

end count_numbers_correct_l3785_378537


namespace coinciding_rest_days_theorem_l3785_378577

/-- Alice's schedule cycle length -/
def alice_cycle : ℕ := 6

/-- Bob's schedule cycle length -/
def bob_cycle : ℕ := 6

/-- Number of days Alice works in her cycle -/
def alice_work_days : ℕ := 4

/-- Number of days Bob works in his cycle -/
def bob_work_days : ℕ := 5

/-- Total number of days to consider -/
def total_days : ℕ := 800

/-- Function to calculate the number of coinciding rest days -/
def coinciding_rest_days : ℕ := 
  (total_days / alice_cycle) * (alice_cycle - alice_work_days - bob_work_days + 1)

theorem coinciding_rest_days_theorem : 
  coinciding_rest_days = 133 := by sorry

end coinciding_rest_days_theorem_l3785_378577


namespace M_intersect_N_eq_solution_l3785_378502

-- Define the set M as the domain of y = 1 / √(1-2x)
def M : Set ℝ := {x : ℝ | x < 1/2}

-- Define the set N as the range of y = x^2 - 4
def N : Set ℝ := {y : ℝ | y ≥ -4}

-- Theorem stating that the intersection of M and N is {x | -4 ≤ x < 1/2}
theorem M_intersect_N_eq_solution : M ∩ N = {x : ℝ | -4 ≤ x ∧ x < 1/2} := by sorry

end M_intersect_N_eq_solution_l3785_378502


namespace sodium_bicarbonate_moles_l3785_378555

-- Define the chemical reaction
structure Reaction where
  hcl : ℝ
  nahco3 : ℝ
  nacl : ℝ

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.hcl = r.nahco3 ∧ r.hcl = r.nacl

-- Theorem statement
theorem sodium_bicarbonate_moles 
  (r : Reaction) 
  (h1 : r.hcl = 1) 
  (h2 : r.nacl = 1) 
  (h3 : balanced_equation r) : 
  r.nahco3 = 1 := by
  sorry

end sodium_bicarbonate_moles_l3785_378555


namespace range_of_a_l3785_378596

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → |x - a| < 2) ∧ 
  (∃ x : ℝ, |x - a| < 2 ∧ (x < 1 ∨ x > 3)) ↔ 
  1 < a ∧ a < 3 := by sorry

end range_of_a_l3785_378596


namespace perpendicular_lines_from_parallel_planes_l3785_378570

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes 
  (α β : Plane) (l m : Line) :
  parallel α β → perpendicular l α → line_parallel m β → 
  line_perpendicular l m :=
sorry

end perpendicular_lines_from_parallel_planes_l3785_378570


namespace specific_pyramid_lateral_area_l3785_378527

/-- Represents a pyramid with a parallelogram base -/
structure Pyramid :=
  (base_side1 : ℝ)
  (base_side2 : ℝ)
  (base_area : ℝ)
  (height : ℝ)

/-- Calculates the lateral surface area of a pyramid -/
def lateral_surface_area (p : Pyramid) : ℝ :=
  sorry

/-- Theorem stating the lateral surface area of the specific pyramid -/
theorem specific_pyramid_lateral_area :
  let p : Pyramid := { 
    base_side1 := 10,
    base_side2 := 18,
    base_area := 90,
    height := 6
  }
  lateral_surface_area p = 192 := by sorry

end specific_pyramid_lateral_area_l3785_378527


namespace g_has_three_zeros_l3785_378559

/-- A function g(x) with a parameter n -/
def g (n : ℕ) (x : ℝ) : ℝ := 2 * x^n + 10 * x^2 - 2 * x - 1

/-- Theorem stating that g(x) has exactly 3 real zeros when n > 3 and n is odd -/
theorem g_has_three_zeros (n : ℕ) (hn : n > 3) (hodd : Odd n) :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, g n x = 0 :=
sorry

end g_has_three_zeros_l3785_378559


namespace quadratic_equation_distinct_roots_l3785_378543

theorem quadratic_equation_distinct_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 4*x₁ - m = 0 ∧ x₂^2 + 4*x₂ - m = 0) ↔ m > -4 :=
by sorry

end quadratic_equation_distinct_roots_l3785_378543


namespace only_pairC_not_opposite_l3785_378583

-- Define a type for quantities
inductive Quantity
| WinGames (n : ℕ)
| LoseGames (n : ℕ)
| RotateCounterclockwise (n : ℕ)
| RotateClockwise (n : ℕ)
| ReceiveMoney (amount : ℕ)
| IncreaseMoney (amount : ℕ)
| TemperatureRise (degrees : ℕ)
| TemperatureDecrease (degrees : ℕ)

-- Define a function to check if two quantities have opposite meanings
def haveOppositeMeanings (q1 q2 : Quantity) : Prop :=
  match q1, q2 with
  | Quantity.WinGames n, Quantity.LoseGames m => true
  | Quantity.RotateCounterclockwise n, Quantity.RotateClockwise m => true
  | Quantity.ReceiveMoney n, Quantity.IncreaseMoney m => false
  | Quantity.TemperatureRise n, Quantity.TemperatureDecrease m => true
  | _, _ => false

-- Define the pairs of quantities
def pairA := (Quantity.WinGames 3, Quantity.LoseGames 3)
def pairB := (Quantity.RotateCounterclockwise 3, Quantity.RotateClockwise 5)
def pairC := (Quantity.ReceiveMoney 3000, Quantity.IncreaseMoney 3000)
def pairD := (Quantity.TemperatureRise 4, Quantity.TemperatureDecrease 10)

-- Theorem statement
theorem only_pairC_not_opposite : 
  (haveOppositeMeanings pairA.1 pairA.2) ∧
  (haveOppositeMeanings pairB.1 pairB.2) ∧
  ¬(haveOppositeMeanings pairC.1 pairC.2) ∧
  (haveOppositeMeanings pairD.1 pairD.2) :=
by sorry

end only_pairC_not_opposite_l3785_378583


namespace gcd_of_B_is_two_l3785_378538

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, x > 0 ∧ n = (x - 1) + x + (x + 1) + (x + 2)}

theorem gcd_of_B_is_two :
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 :=
sorry

end gcd_of_B_is_two_l3785_378538


namespace abs_neg_three_eq_three_l3785_378536

theorem abs_neg_three_eq_three : |(-3 : ℚ)| = 3 := by sorry

end abs_neg_three_eq_three_l3785_378536


namespace parallel_vectors_x_value_l3785_378594

/-- Two-dimensional vector -/
def Vector2D := ℝ × ℝ

/-- Parallel vectors are scalar multiples of each other -/
def is_parallel (v w : Vector2D) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let a : Vector2D := (1, -2)
  let b : Vector2D := (-2, x)
  is_parallel a b → x = 4 := by
  sorry

end parallel_vectors_x_value_l3785_378594


namespace codecracker_codes_count_l3785_378557

/-- The number of available colors in the CodeCracker game -/
def num_colors : ℕ := 6

/-- The number of slots in a CodeCracker code -/
def code_length : ℕ := 5

/-- The number of possible secret codes in the CodeCracker game -/
def num_codes : ℕ := num_colors * (num_colors - 1)^(code_length - 1)

theorem codecracker_codes_count :
  num_codes = 3750 :=
by sorry

end codecracker_codes_count_l3785_378557


namespace yellow_marbles_count_l3785_378530

theorem yellow_marbles_count (total : ℕ) (red : ℕ) 
  (h1 : total = 140)
  (h2 : red = 10)
  (h3 : ∃ blue : ℕ, blue = (5 * red) / 2)
  (h4 : ∃ green : ℕ, green = ((13 * blue) / 10))
  (h5 : ∃ yellow : ℕ, yellow = total - (blue + red + green)) :
  yellow = 73 := by
  sorry

end yellow_marbles_count_l3785_378530


namespace toucan_count_l3785_378535

/-- The total number of toucans after joining all limbs -/
def total_toucans (initial_first initial_second initial_third joining_first joining_second joining_third : ℝ) : ℝ :=
  (initial_first + joining_first) + (initial_second + joining_second) + (initial_third + joining_third)

/-- Theorem stating the total number of toucans after joining -/
theorem toucan_count : 
  total_toucans 3.5 4.25 2.75 1.5 0.6 1.2 = 13.8 := by
  sorry

end toucan_count_l3785_378535


namespace connecting_line_is_correct_l3785_378505

/-- The equation of a circle in the form (x-h)^2 + (y-k)^2 = r^2 -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given two circles, returns the line connecting their centers -/
def line_connecting_centers (c1 c2 : Circle) : Line :=
  sorry

/-- The first circle: x^2+y^2-4x+6y=0 -/
def circle1 : Circle :=
  { h := 2, k := -3, r := 5 }

/-- The second circle: x^2+y^2-6x=0 -/
def circle2 : Circle :=
  { h := 3, k := 0, r := 3 }

/-- The expected line: 3x-y-9=0 -/
def expected_line : Line :=
  { a := 3, b := -1, c := -9 }

theorem connecting_line_is_correct :
  line_connecting_centers circle1 circle2 = expected_line :=
sorry

end connecting_line_is_correct_l3785_378505


namespace vasya_no_purchase_days_l3785_378522

/-- Represents Vasya's purchases over 15 school days -/
structure VasyaPurchases where
  marshmallow_days : ℕ -- Days buying 9 marshmallows
  meatpie_days : ℕ -- Days buying 2 meat pies
  combo_days : ℕ -- Days buying 4 marshmallows and 1 meat pie
  nothing_days : ℕ -- Days buying nothing

/-- Theorem stating the number of days Vasya didn't buy anything -/
theorem vasya_no_purchase_days (p : VasyaPurchases) : 
  p.marshmallow_days + p.meatpie_days + p.combo_days + p.nothing_days = 15 → 
  9 * p.marshmallow_days + 4 * p.combo_days = 30 →
  2 * p.meatpie_days + p.combo_days = 9 →
  p.nothing_days = 7 := by
  sorry

#check vasya_no_purchase_days

end vasya_no_purchase_days_l3785_378522


namespace no_valid_k_exists_l3785_378517

/-- The nth odd prime number -/
def nthOddPrime (n : ℕ) : ℕ := sorry

/-- The product of the first n odd prime numbers -/
def productFirstNOddPrimes (n : ℕ) : ℕ := sorry

/-- Statement: There does not exist a natural number k such that the product 
    of the first k odd prime numbers, decreased by 1, is an exact power 
    of a natural number greater than 1 -/
theorem no_valid_k_exists : 
  ¬ ∃ (k : ℕ), ∃ (a n : ℕ), n > 1 ∧ productFirstNOddPrimes k - 1 = a^n := by
  sorry


end no_valid_k_exists_l3785_378517


namespace chess_tournament_games_l3785_378587

/-- The number of games played in a chess tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 20 participants, where each participant
    plays exactly one game with each of the remaining participants,
    the total number of games played is 190. -/
theorem chess_tournament_games :
  num_games 20 = 190 := by
  sorry

end chess_tournament_games_l3785_378587


namespace unique_triple_solution_l3785_378509

theorem unique_triple_solution : 
  ∃! (p x y : ℕ), 
    Prime p ∧ 
    p ^ x = y ^ 4 + 4 ∧ 
    p = 5 ∧ x = 1 ∧ y = 1 := by
  sorry

end unique_triple_solution_l3785_378509


namespace monotonicity_condition_inequality_solution_correct_l3785_378503

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - (3*m - 1) * x + m - 2

-- Part 1: Monotonicity condition
theorem monotonicity_condition (m : ℝ) :
  (∀ x ∈ Set.Icc 2 3, Monotone (f m)) ↔ m ≥ -1/3 :=
sorry

-- Part 2: Inequality solution
def inequality_solution (m : ℝ) : Set ℝ :=
  if m = 0 then Set.Ioi 2
  else if m > 0 then Set.Iio ((m-1)/m) ∪ Set.Ioi 2
  else if -1 < m ∧ m < 0 then Set.Ioo 2 ((m-1)/m)
  else if m = -1 then ∅
  else Set.Ioo ((m-1)/m) 2

theorem inequality_solution_correct (m : ℝ) (x : ℝ) :
  x ∈ inequality_solution m ↔ f m x + m > 0 :=
sorry

end monotonicity_condition_inequality_solution_correct_l3785_378503


namespace quadratic_inequality_range_l3785_378572

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → 0 < a ∧ a < 1 := by
  sorry

end quadratic_inequality_range_l3785_378572


namespace quadratic_equation_has_two_distinct_real_roots_l3785_378568

/-- Proves that the quadratic equation (2kx^2 + 5kx + 2) = 0, where k = 0.64, has two distinct real roots -/
theorem quadratic_equation_has_two_distinct_real_roots :
  let k : ℝ := 0.64
  let a : ℝ := 2 * k
  let b : ℝ := 5 * k
  let c : ℝ := 2
  let discriminant := b^2 - 4*a*c
  discriminant > 0 ∧ a ≠ 0 := by sorry

end quadratic_equation_has_two_distinct_real_roots_l3785_378568


namespace unique_prime_base_l3785_378591

theorem unique_prime_base (b : ℕ) : 
  Prime b ∧ (b + 5)^2 = 3*b^2 + 6*b + 1 → b = 3 := by
sorry

end unique_prime_base_l3785_378591


namespace mityas_age_l3785_378529

theorem mityas_age (shura_age mitya_age : ℚ) : 
  (mitya_age = shura_age + 11) →
  (mitya_age - shura_age = 2 * (shura_age - (mitya_age - shura_age))) →
  mitya_age = 27.5 := by sorry

end mityas_age_l3785_378529


namespace inequality_proofs_l3785_378519

theorem inequality_proofs :
  (∀ a b : ℝ, a > 0 → b > 0 → (a + b) * (1 / a + 1 / b) ≥ 4) ∧
  (Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5) :=
by sorry

end inequality_proofs_l3785_378519


namespace painting_cost_is_474_l3785_378523

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a wall -/
def wallArea (d : Dimensions) : ℝ := d.length * d.height

/-- Calculates the area of a rectangular opening -/
def openingArea (d : Dimensions) : ℝ := d.length * d.width

/-- Represents a room with its dimensions and openings -/
structure Room where
  dimensions : Dimensions
  doorDimensions : Dimensions
  largeDoorCount : ℕ
  largeWindowDimensions : Dimensions
  largeWindowCount : ℕ
  smallWindowDimensions : Dimensions
  smallWindowCount : ℕ

/-- Calculates the total wall area of a room -/
def totalWallArea (r : Room) : ℝ :=
  2 * (wallArea r.dimensions + wallArea { length := r.dimensions.width, width := r.dimensions.width, height := r.dimensions.height })

/-- Calculates the total area of openings in a room -/
def totalOpeningArea (r : Room) : ℝ :=
  (r.largeDoorCount : ℝ) * openingArea r.doorDimensions +
  (r.largeWindowCount : ℝ) * openingArea r.largeWindowDimensions +
  (r.smallWindowCount : ℝ) * openingArea r.smallWindowDimensions

/-- Calculates the paintable area of a room -/
def paintableArea (r : Room) : ℝ :=
  totalWallArea r - totalOpeningArea r

/-- Theorem: The cost of painting the given room is Rs. 474 -/
theorem painting_cost_is_474 (r : Room)
  (h1 : r.dimensions = { length := 10, width := 7, height := 5 })
  (h2 : r.doorDimensions = { length := 1, width := 3, height := 3 })
  (h3 : r.largeDoorCount = 2)
  (h4 : r.largeWindowDimensions = { length := 2, width := 1.5, height := 1.5 })
  (h5 : r.largeWindowCount = 1)
  (h6 : r.smallWindowDimensions = { length := 1, width := 1.5, height := 1.5 })
  (h7 : r.smallWindowCount = 2)
  : paintableArea r * 3 = 474 := by
  sorry

end painting_cost_is_474_l3785_378523


namespace square_root_of_1_5625_l3785_378531

theorem square_root_of_1_5625 : Real.sqrt 1.5625 = 1.25 := by
  sorry

end square_root_of_1_5625_l3785_378531


namespace circle_parabola_intersection_l3785_378565

/-- A circle with center on y = b intersects y = (4/3)x^2 at least thrice, including the origin --/
def CircleIntersectsParabola (b : ℝ) : Prop :=
  ∃ (r : ℝ) (a : ℝ), (a^2 + b^2 = r^2) ∧ 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧
    ((x₁^2 + ((4/3)*x₁^2 - b)^2 = r^2) ∧
     (x₂^2 + ((4/3)*x₂^2 - b)^2 = r^2)))

/-- Two non-origin intersection points lie on y = (4/3)x + b --/
def IntersectionPointsOnLine (b : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧
    ((4/3)*x₁^2 = (4/3)*x₁ + b) ∧
    ((4/3)*x₂^2 = (4/3)*x₂ + b)

/-- The theorem to be proved --/
theorem circle_parabola_intersection (b : ℝ) :
  (CircleIntersectsParabola b ∧ IntersectionPointsOnLine b) ↔ b = 25/12 :=
sorry

end circle_parabola_intersection_l3785_378565


namespace complement_intersection_equals_set_l3785_378593

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4, 5}

theorem complement_intersection_equals_set : 
  (Aᶜ : Set ℕ) ∩ B = {1, 4, 5, 6, 7, 8} := by sorry

end complement_intersection_equals_set_l3785_378593


namespace total_distance_is_15_l3785_378575

def morning_ride : ℕ := 2

def evening_ride (m : ℕ) : ℕ := 5 * m

def third_ride (m : ℕ) : ℕ := 2 * m - 1

def total_distance (m : ℕ) : ℕ := m + evening_ride m + third_ride m

theorem total_distance_is_15 : total_distance morning_ride = 15 := by
  sorry

end total_distance_is_15_l3785_378575


namespace quadratic_function_max_value_l3785_378507

def f (a x : ℝ) : ℝ := a * x^2 + (2*a - 1) * x - 3

theorem quadratic_function_max_value (a : ℝ) (h1 : a ≠ 0) :
  (∀ x ∈ Set.Icc (-3/2 : ℝ) 2, f a x ≤ 1) ∧
  (∃ x ∈ Set.Icc (-3/2 : ℝ) 2, f a x = 1) →
  a = 3/4 ∨ a = 1/2 := by
sorry

end quadratic_function_max_value_l3785_378507


namespace symmetric_derivative_implies_cosine_possible_l3785_378562

/-- A function whose derivative's graph is symmetric about the origin -/
class SymmetricDerivative (f : ℝ → ℝ) : Prop :=
  (symmetric : ∀ x : ℝ, (deriv f) x = -(deriv f) (-x))

/-- The theorem stating that if f'(x) is symmetric about the origin, 
    then f(x) = 3cos(x) is a possible expression for f(x) -/
theorem symmetric_derivative_implies_cosine_possible 
  (f : ℝ → ℝ) [SymmetricDerivative f] : 
  ∃ g : ℝ → ℝ, (∀ x, f x = 3 * Real.cos x) ∧ SymmetricDerivative g :=
sorry

end symmetric_derivative_implies_cosine_possible_l3785_378562


namespace simplify_linear_expression_l3785_378533

theorem simplify_linear_expression (y : ℝ) : 2*y + 3*y + 4*y = 9*y := by
  sorry

end simplify_linear_expression_l3785_378533


namespace jogger_train_distance_jogger_train_problem_l3785_378524

theorem jogger_train_distance (jogger_speed : Real) (train_speed : Real) 
  (train_length : Real) (passing_time : Real) : Real :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let distance_covered := relative_speed * passing_time
  distance_covered - train_length

theorem jogger_train_problem :
  jogger_train_distance 9 45 120 40 = 280 := by
  sorry

end jogger_train_distance_jogger_train_problem_l3785_378524


namespace park_area_l3785_378526

/-- Represents a rectangular park with sides in ratio 3:2 -/
structure RectangularPark where
  x : ℝ
  length : ℝ := 3 * x
  width : ℝ := 2 * x

/-- The cost of fencing in pence per meter -/
def fencing_cost_per_meter : ℝ := 40

/-- The total cost of fencing in dollars -/
def total_fencing_cost : ℝ := 100

theorem park_area (park : RectangularPark) : 
  (2 * (park.length + park.width) * fencing_cost_per_meter / 100 = total_fencing_cost) →
  (park.length * park.width = 3750) := by
  sorry

end park_area_l3785_378526


namespace solution_value_l3785_378586

theorem solution_value (m : ℝ) : 
  (∃ x : ℝ, x = 1 ∧ (m / (2 - x)) - (1 / (x - 2)) = 3) → m = 2 := by
  sorry

end solution_value_l3785_378586


namespace third_term_of_specific_sequence_l3785_378534

/-- Represents a geometric sequence of positive integers -/
structure GeometricSequence where
  first_term : ℕ
  common_ratio : ℕ
  first_term_pos : 0 < first_term

/-- The nth term of a geometric sequence -/
def nth_term (seq : GeometricSequence) (n : ℕ) : ℕ :=
  seq.first_term * seq.common_ratio ^ (n - 1)

theorem third_term_of_specific_sequence :
  ∀ (seq : GeometricSequence),
    seq.first_term = 5 →
    nth_term seq 4 = 320 →
    nth_term seq 3 = 80 := by
  sorry

end third_term_of_specific_sequence_l3785_378534


namespace parallel_vectors_x_value_l3785_378579

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-3, 2)
  let b : ℝ × ℝ := (x, -4)
  are_parallel a b → x = 6 := by
  sorry

end parallel_vectors_x_value_l3785_378579


namespace gardener_roses_order_l3785_378554

/-- The number of roses ordered by the gardener -/
def roses : ℕ := 320

/-- The number of tulips ordered -/
def tulips : ℕ := 250

/-- The number of carnations ordered -/
def carnations : ℕ := 375

/-- The cost of each flower in euros -/
def flower_cost : ℕ := 2

/-- The total expenses in euros -/
def total_expenses : ℕ := 1890

theorem gardener_roses_order :
  roses = (total_expenses - (tulips + carnations) * flower_cost) / flower_cost := by
  sorry

end gardener_roses_order_l3785_378554


namespace remainder_theorem_l3785_378578

/-- Given a polynomial p(x) satisfying p(0) = 2 and p(2) = 6,
    prove that the remainder when p(x) is divided by x(x-2) is 2x + 2 -/
theorem remainder_theorem (p : ℝ → ℝ) (h1 : p 0 = 2) (h2 : p 2 = 6) :
  ∃ (q : ℝ → ℝ), ∀ x, p x = q x * (x * (x - 2)) + (2 * x + 2) := by
  sorry

end remainder_theorem_l3785_378578


namespace second_price_increase_l3785_378549

theorem second_price_increase (P : ℝ) (x : ℝ) (h : P > 0) :
  (1.15 * P) * (1 + x / 100) = 1.4375 * P → x = 25 := by
sorry

end second_price_increase_l3785_378549


namespace inequality_proof_l3785_378588

theorem inequality_proof (x y : ℝ) : 
  (∀ x, |x| + |x - 3| < x + 6 ↔ -1 < x ∧ x < 9) →
  x > 0 →
  y > 0 →
  9*x + y - 1 = 0 →
  x + y ≥ 16*x*y := by
sorry

end inequality_proof_l3785_378588


namespace complex_fraction_sum_l3785_378558

theorem complex_fraction_sum (x y : ℂ) 
  (h : (x + y) / (x - y) - 3 * (x - y) / (x + y) = 2) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 8320 / 4095 := by
  sorry

end complex_fraction_sum_l3785_378558


namespace downstream_distance_l3785_378532

/-- Proves that the distance traveled downstream is 420 km given the conditions -/
theorem downstream_distance
  (downstream_time : ℝ)
  (upstream_speed : ℝ)
  (total_speed : ℝ)
  (h1 : downstream_time = 20)
  (h2 : upstream_speed = 12)
  (h3 : total_speed = 21) :
  downstream_time * total_speed = 420 := by
  sorry

end downstream_distance_l3785_378532


namespace bisecting_plane_intersects_sixteen_cubes_l3785_378589

/-- Represents a cube composed of unit cubes -/
structure UnitCube where
  side_length : ℕ

/-- Represents a plane that bisects a face diagonal of a cube -/
structure BisectingPlane where
  cube : UnitCube

/-- Counts the number of unit cubes intersected by a bisecting plane -/
def count_intersected_cubes (plane : BisectingPlane) : ℕ :=
  sorry

/-- Theorem stating that a plane bisecting a face diagonal of a 4x4x4 cube intersects 16 unit cubes -/
theorem bisecting_plane_intersects_sixteen_cubes 
  (cube : UnitCube) 
  (plane : BisectingPlane) 
  (h1 : cube.side_length = 4) 
  (h2 : plane.cube = cube) : 
  count_intersected_cubes plane = 16 := by
  sorry

end bisecting_plane_intersects_sixteen_cubes_l3785_378589


namespace pizza_size_increase_l3785_378539

theorem pizza_size_increase (r : ℝ) (h : r > 0) :
  let R := r * (1 + 0.5)
  (π * R^2) / (π * r^2) = 2.25 := by
  sorry

end pizza_size_increase_l3785_378539


namespace factor_polynomial_l3785_378592

theorem factor_polynomial (x : ℝ) : 
  36 * x^6 - 189 * x^12 + 81 * x^9 = 9 * x^6 * (4 + 9 * x^3 - 21 * x^6) := by sorry

end factor_polynomial_l3785_378592


namespace gain_percent_calculation_l3785_378595

def cost_price : ℝ := 900
def selling_price : ℝ := 1080

theorem gain_percent_calculation :
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end gain_percent_calculation_l3785_378595


namespace ship_speed_and_distance_l3785_378518

theorem ship_speed_and_distance 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (current_speed : ℝ) 
  (h1 : downstream_time = 3)
  (h2 : upstream_time = 4)
  (h3 : current_speed = 3) :
  ∃ (still_water_speed : ℝ) (distance : ℝ),
    still_water_speed = 21 ∧
    distance = 72 ∧
    downstream_time * (still_water_speed + current_speed) = distance ∧
    upstream_time * (still_water_speed - current_speed) = distance :=
by sorry

end ship_speed_and_distance_l3785_378518


namespace imaginary_part_of_complex_fraction_l3785_378564

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im ((3 + i) / i) = -3 := by
sorry

end imaginary_part_of_complex_fraction_l3785_378564


namespace monkey_climb_distance_l3785_378560

/-- Represents the climbing behavior of a monkey -/
structure MonkeyClimb where
  climb_distance : ℝ  -- Distance the monkey climbs in one minute
  slip_distance : ℝ   -- Distance the monkey slips in the next minute
  total_time : ℕ      -- Total time taken to reach the top
  total_height : ℝ    -- Total height reached

/-- Theorem stating that given the monkey's climbing behavior, 
    if it takes 37 minutes to reach 60 meters, then it climbs 6 meters per minute -/
theorem monkey_climb_distance 
  (m : MonkeyClimb) 
  (h1 : m.slip_distance = 3) 
  (h2 : m.total_time = 37) 
  (h3 : m.total_height = 60) : 
  m.climb_distance = 6 := by
  sorry

#check monkey_climb_distance

end monkey_climb_distance_l3785_378560


namespace area_ratio_of_squares_l3785_378553

/-- Given three square regions A, B, and C with perimeters 16, 32, and 20 units respectively,
    prove that the ratio of the area of region B to the area of region C is 64/25. -/
theorem area_ratio_of_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (perim_a : 4 * a = 16) (perim_b : 4 * b = 32) (perim_c : 4 * c = 20) :
  (b * b) / (c * c) = 64 / 25 := by
  sorry

end area_ratio_of_squares_l3785_378553


namespace cos_arctan_squared_l3785_378569

theorem cos_arctan_squared (x : ℝ) (h1 : x > 0) (h2 : Real.cos (Real.arctan x) = x) :
  x^2 = (Real.sqrt 5 - 1) / 2 := by sorry

end cos_arctan_squared_l3785_378569


namespace tangent_circles_a_value_l3785_378514

/-- Circle C₁ with equation x² + y² = 16 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 16}

/-- Circle C₂ with equation (x - a)² + y² = 1, parameterized by a -/
def C₂ (a : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + p.2^2 = 1}

/-- Two circles are tangent if they intersect at exactly one point -/
def are_tangent (S T : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ S ∧ p ∈ T

/-- The main theorem: if C₁ and C₂(a) are tangent, then a = ±5 or a = ±3 -/
theorem tangent_circles_a_value :
  ∀ a : ℝ, are_tangent C₁ (C₂ a) → a = 5 ∨ a = -5 ∨ a = 3 ∨ a = -3 :=
sorry

end tangent_circles_a_value_l3785_378514


namespace sum_of_coefficients_l3785_378508

theorem sum_of_coefficients : 
  let p (x : ℝ) := 3*(x^8 - x^5 + 2*x^3 - 6) - 5*(x^4 + 3*x^2) + 2*(x^6 - 5)
  (p 1) = -40 := by sorry

end sum_of_coefficients_l3785_378508


namespace triangle_rotation_reflection_l3785_378510

/-- Rotation of 90 degrees clockwise about the origin -/
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

/-- Reflection over the y-axis -/
def reflectOverYAxis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Given triangle ABC with vertices A(-3, 2), B(0, 5), and C(0, 2),
    prove that after rotating 90 degrees clockwise about the origin
    and then reflecting over the y-axis, point A ends up at (-2, 3) -/
theorem triangle_rotation_reflection :
  let A : ℝ × ℝ := (-3, 2)
  let B : ℝ × ℝ := (0, 5)
  let C : ℝ × ℝ := (0, 2)
  reflectOverYAxis (rotate90Clockwise A) = (-2, 3) := by
sorry


end triangle_rotation_reflection_l3785_378510


namespace product_different_from_hundred_l3785_378545

theorem product_different_from_hundred : ∃! (x y : ℚ), 
  ((x = 10 ∧ y = 10) ∨ 
   (x = 20 ∧ y = -5) ∨ 
   (x = -4 ∧ y = -25) ∨ 
   (x = 50 ∧ y = 2) ∨ 
   (x = 5/2 ∧ y = 40)) ∧ 
  x * y ≠ 100 := by
  sorry

end product_different_from_hundred_l3785_378545


namespace boat_journey_time_l3785_378540

/-- Calculates the total time for a round trip boat journey affected by a stream -/
theorem boat_journey_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed = 9) 
  (h2 : stream_speed = 6) 
  (h3 : distance = 300) : 
  (distance / (boat_speed + stream_speed)) + (distance / (boat_speed - stream_speed)) = 120 := by
  sorry

end boat_journey_time_l3785_378540


namespace min_value_quadratic_l3785_378506

theorem min_value_quadratic (x y : ℝ) :
  3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 9 ∧
  (3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 = 9 ↔ x = 0 ∧ y = 0) :=
by sorry

end min_value_quadratic_l3785_378506


namespace janet_roller_coaster_rides_l3785_378573

/-- The number of tickets required for one roller coaster ride -/
def roller_coaster_tickets : ℕ := 5

/-- The number of tickets required for one giant slide ride -/
def giant_slide_tickets : ℕ := 3

/-- The number of times Janet wants to ride the giant slide -/
def giant_slide_rides : ℕ := 4

/-- The total number of tickets Janet needs -/
def total_tickets : ℕ := 47

/-- The number of times Janet wants to ride the roller coaster -/
def roller_coaster_rides : ℕ := 7

theorem janet_roller_coaster_rides : 
  roller_coaster_tickets * roller_coaster_rides + 
  giant_slide_tickets * giant_slide_rides = total_tickets :=
by sorry

end janet_roller_coaster_rides_l3785_378573


namespace coefficient_x4_equals_240_l3785_378515

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the coefficient of x^4 in (1+2x)^6
def coefficient_x4 : ℕ := binomial 6 4 * 2^4

-- Theorem statement
theorem coefficient_x4_equals_240 : coefficient_x4 = 240 := by
  sorry

end coefficient_x4_equals_240_l3785_378515


namespace product_469111_9999_l3785_378571

theorem product_469111_9999 : 469111 * 9999 = 4690418889 := by
  sorry

end product_469111_9999_l3785_378571


namespace triangle_side_length_l3785_378525

theorem triangle_side_length (b c : ℝ) (A : ℝ) (S : ℝ) : 
  b = 2 → 
  A = 2 * π / 3 → 
  S = 2 * Real.sqrt 3 → 
  S = 1/2 * b * c * Real.sin A →
  b^2 + c^2 - 2*b*c*Real.cos A = (2 * Real.sqrt 7)^2 :=
by sorry

end triangle_side_length_l3785_378525


namespace soap_box_width_l3785_378567

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Theorem: Given the carton and soap box dimensions, and the maximum number of soap boxes,
    prove that the width of each soap box is 7 inches -/
theorem soap_box_width
  (carton : BoxDimensions)
  (soap_box : BoxDimensions)
  (max_soap_boxes : ℕ)
  (h1 : carton.length = 25)
  (h2 : carton.width = 42)
  (h3 : carton.height = 60)
  (h4 : soap_box.length = 6)
  (h5 : soap_box.height = 6)
  (h6 : max_soap_boxes = 250)
  (h7 : max_soap_boxes * boxVolume soap_box = boxVolume carton) :
  soap_box.width = 7 := by
  sorry

end soap_box_width_l3785_378567


namespace squarefree_primes_property_l3785_378521

theorem squarefree_primes_property : 
  {p : ℕ | Nat.Prime p ∧ p ≥ 3 ∧ 
    ∀ q : ℕ, Nat.Prime q → q < p → 
      Squarefree (p - p / q * q)} = {5, 7, 13} := by sorry

end squarefree_primes_property_l3785_378521


namespace f_derivative_at_zero_l3785_378500

noncomputable def f (x : ℝ) : ℝ := Real.exp (2*x + 1) - 3*x

theorem f_derivative_at_zero : 
  deriv f 0 = 2 * Real.exp 1 - 3 := by sorry

end f_derivative_at_zero_l3785_378500


namespace line_perpendicular_slope_l3785_378556

/-- Given a line l passing through points (a-2, -1) and (-a-2, 1), perpendicular to a line
    through (-2, 1) with slope -2/3, prove that a = -2/3 -/
theorem line_perpendicular_slope (a : ℝ) : 
  let l : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = ((1 - t) * (a - 2) + t * (-a - 2), (1 - t) * (-1) + t * 1)}
  let m : ℝ := (1 - (-1)) / ((-a - 2) - (a - 2))
  (∀ p ∈ l, (p.2 - 1) = -2/3 * (p.1 - (-2))) → m * (-2/3) = -1 → a = -2/3 := by
sorry

end line_perpendicular_slope_l3785_378556


namespace cross_number_intersection_l3785_378584

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def power_of_three (n : ℕ) : Prop := ∃ m : ℕ, n = 3^m

def power_of_seven (n : ℕ) : Prop := ∃ m : ℕ, n = 7^m

theorem cross_number_intersection :
  ∃! d : ℕ,
    d < 10 ∧
    ∃ (n m : ℕ),
      is_three_digit n ∧
      is_three_digit m ∧
      power_of_three n ∧
      power_of_seven m ∧
      n % 10 = d ∧
      (m / 100) % 10 = d :=
sorry

end cross_number_intersection_l3785_378584


namespace max_xy_value_l3785_378566

theorem max_xy_value (x y : ℕ) (h : 69 * x + 54 * y ≤ 2008) : x * y ≤ 270 := by
  sorry

end max_xy_value_l3785_378566


namespace new_student_weight_l3785_378599

/-- Given a group of 10 students, proves that replacing a 120 kg student with a new student
    that causes the average weight to decrease by 6 kg results in the new student weighing 60 kg. -/
theorem new_student_weight
  (n : ℕ) -- number of students
  (old_avg : ℝ) -- original average weight
  (replaced_weight : ℝ) -- weight of the replaced student
  (new_avg : ℝ) -- new average weight after replacement
  (h1 : n = 10) -- there are 10 students
  (h2 : new_avg = old_avg - 6) -- average weight decreases by 6 kg
  (h3 : replaced_weight = 120) -- replaced student weighs 120 kg
  : n * new_avg + 60 = n * old_avg - replaced_weight := by
  sorry

#check new_student_weight

end new_student_weight_l3785_378599


namespace video_game_spending_l3785_378547

/-- The total amount spent on video games is the sum of the costs of individual games -/
theorem video_game_spending (basketball_cost racing_cost : ℚ) :
  basketball_cost = 5.2 →
  racing_cost = 4.23 →
  basketball_cost + racing_cost = 9.43 := by
  sorry

end video_game_spending_l3785_378547


namespace parallelogram_count_l3785_378580

-- Define the parallelogram structure
structure Parallelogram where
  b : ℕ
  d : ℕ
  area_eq : b * d = 1728000
  b_positive : b > 0
  d_positive : d > 0

-- Define the count function
def count_parallelograms : ℕ := sorry

-- Theorem statement
theorem parallelogram_count : count_parallelograms = 56 := by sorry

end parallelogram_count_l3785_378580


namespace root_one_implies_m_three_l3785_378563

theorem root_one_implies_m_three (m : ℝ) : 
  (∃ x : ℝ, x^2 - m*x + 2 = 0 ∧ x = 1) → m = 3 := by
  sorry

end root_one_implies_m_three_l3785_378563


namespace white_smallest_probability_l3785_378504

def total_balls : ℕ := 16
def red_balls : ℕ := 9
def black_balls : ℕ := 5
def white_balls : ℕ := 2

theorem white_smallest_probability :
  (white_balls : ℚ) / total_balls < (red_balls : ℚ) / total_balls ∧
  (white_balls : ℚ) / total_balls < (black_balls : ℚ) / total_balls :=
by sorry

end white_smallest_probability_l3785_378504


namespace min_calls_for_gossip_l3785_378541

theorem min_calls_for_gossip (n : ℕ) (h : n > 0) : ℕ :=
  2 * (n - 1)

/- Proof
sorry
-/

end min_calls_for_gossip_l3785_378541


namespace wanda_blocks_calculation_l3785_378528

theorem wanda_blocks_calculation (initial_blocks : ℕ) (theresa_percentage : ℚ) (give_away_fraction : ℚ) : 
  initial_blocks = 2450 →
  theresa_percentage = 35 / 100 →
  give_away_fraction = 1 / 8 →
  (initial_blocks + Int.floor (theresa_percentage * initial_blocks) - 
   Int.floor (give_away_fraction * (initial_blocks + Int.floor (theresa_percentage * initial_blocks)))) = 2894 := by
  sorry

end wanda_blocks_calculation_l3785_378528


namespace sum_of_three_numbers_l3785_378574

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 20 := by sorry

end sum_of_three_numbers_l3785_378574


namespace enough_paint_l3785_378546

/-- Represents the dimensions of the gym --/
structure GymDimensions where
  length : ℝ
  width : ℝ

/-- Represents the paint requirements and availability --/
structure PaintInfo where
  cans : ℕ
  weight_per_can : ℝ
  paint_per_sqm : ℝ

/-- Theorem stating that there is enough paint for the gym floor --/
theorem enough_paint (gym : GymDimensions) (paint : PaintInfo) : 
  gym.length = 65 ∧ 
  gym.width = 32 ∧ 
  paint.cans = 23 ∧ 
  paint.weight_per_can = 25 ∧ 
  paint.paint_per_sqm = 0.25 → 
  (paint.cans : ℝ) * paint.weight_per_can > gym.length * gym.width * paint.paint_per_sqm := by
  sorry

#check enough_paint

end enough_paint_l3785_378546


namespace property_implies_increasing_l3785_378520

-- Define the property that (f(a) - f(b)) / (a - b) > 0 for all distinct a and b
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, a ≠ b → (f a - f b) / (a - b) > 0

-- Define what it means for a function to be increasing
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- State the theorem
theorem property_implies_increasing (f : ℝ → ℝ) :
  satisfies_property f → is_increasing f :=
by
  sorry

end property_implies_increasing_l3785_378520


namespace square_area_error_l3785_378551

theorem square_area_error (x : ℝ) (h : x > 0) :
  let measured_side := 1.25 * x
  let actual_area := x ^ 2
  let calculated_area := measured_side ^ 2
  let error_percentage := (calculated_area - actual_area) / actual_area * 100
  error_percentage = 56.25 := by
sorry

end square_area_error_l3785_378551


namespace automobile_distance_l3785_378576

/-- Proves that an automobile traveling a/4 feet in 2r seconds will travel 25a/r yards in 10 minutes -/
theorem automobile_distance (a r : ℝ) (h : r ≠ 0) : 
  let rate_feet_per_second := a / (4 * 2 * r)
  let rate_yards_per_second := rate_feet_per_second / 3
  let time_seconds := 10 * 60
  rate_yards_per_second * time_seconds = 25 * a / r := by sorry

end automobile_distance_l3785_378576


namespace sugar_bag_weight_l3785_378597

/-- The weight of a bag of sugar, given the weight of a bag of salt and their combined weight after removing 4 kg. -/
theorem sugar_bag_weight (salt_weight : ℝ) (combined_weight_minus_four : ℝ) 
  (h1 : salt_weight = 30)
  (h2 : combined_weight_minus_four = 42)
  (h3 : combined_weight_minus_four = salt_weight + sugar_weight - 4) :
  sugar_weight = 16 := by
  sorry

#check sugar_bag_weight

end sugar_bag_weight_l3785_378597
