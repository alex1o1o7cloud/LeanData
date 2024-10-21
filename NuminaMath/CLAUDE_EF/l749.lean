import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_transform_l749_74927

noncomputable def f (x : ℝ) : ℝ := Real.sin x

noncomputable def transform (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f (2 * (x - Real.pi/3))

theorem sin_transform :
  ∀ x : ℝ, transform f x = Real.sin (2*x - 2*Real.pi/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_transform_l749_74927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_last_third_l749_74950

/-- The speed of a car during the last third of a journey, given specific conditions -/
theorem car_speed_last_third (D : ℝ) (h : D > 0) : 
  let avg_speed := 30.000000000000004
  let first_third_speed := 80
  let second_third_speed := 15
  ∃ V : ℝ, 
    let total_time := D / (3 * first_third_speed) + D / (3 * second_third_speed) + D / (3 * V)
    D / total_time = avg_speed ∧ V = 35.625 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_last_third_l749_74950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bc_length_l749_74976

/-- Represents a trapezoid ABCD with given properties -/
structure Trapezoid where
  area : ℝ
  altitude : ℝ
  ab_length : ℝ
  cd_length : ℝ
  area_eq : area = 200
  altitude_eq : altitude = 10
  ab_eq : ab_length = 12
  cd_eq : cd_length = 22

/-- The length of BC in the trapezoid -/
noncomputable def bc_length (t : Trapezoid) : ℝ := 20 - 5 * (Real.sqrt 11 + 4 * Real.sqrt 6)

/-- Theorem stating that BC length is as calculated -/
theorem trapezoid_bc_length (t : Trapezoid) : 
  bc_length t = 20 - 5 * (Real.sqrt 11 + 4 * Real.sqrt 6) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bc_length_l749_74976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alexander_tyson_difference_l749_74943

-- Define the number of joggers bought by each person
def tyson_joggers : ℕ := sorry
def alexander_joggers : ℕ := sorry
def christopher_joggers : ℕ := sorry

-- Define the conditions
axiom alexander_more_than_tyson : alexander_joggers > tyson_joggers
axiom christopher_twenty_times_tyson : christopher_joggers = 20 * tyson_joggers
axiom christopher_total : christopher_joggers = 80
axiom christopher_more_than_alexander : christopher_joggers = alexander_joggers + 54

-- Theorem to prove
theorem alexander_tyson_difference : alexander_joggers - tyson_joggers = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alexander_tyson_difference_l749_74943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_is_30_l749_74963

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Returns true if the given point lies on the given line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Returns true if the given point is the projection of another point on the given line -/
def isProjection (p q : Point) (l : Line) : Prop :=
  pointOnLine q l ∧ (q.x - p.x) * l.a + (q.y - p.y) * l.b = 0

/-- Calculates the inclination angle of a line in degrees -/
noncomputable def inclinationAngle (l : Line) : ℝ :=
  Real.arctan (l.a / l.b) * (180 / Real.pi)

theorem line_inclination_angle_is_30 (l : Line) (p q : Point) :
  pointOnLine p l →
  isProjection p q l →
  p.x = -1 →
  p.y = 0 →
  q.x = -2 →
  q.y = Real.sqrt 3 →
  inclinationAngle l = 30 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_is_30_l749_74963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_monotone_interval_l749_74933

open Real Set

theorem sin_monotone_interval (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, x ∈ Icc 0 a → f x = sin (2 * x + π / 3)) →
  (∀ x y, x ∈ Icc 0 a → y ∈ Icc 0 a → x < y → f x < f y) →
  a > 0 →
  0 < a ∧ a ≤ π / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_monotone_interval_l749_74933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_528_l749_74908

theorem closest_perfect_square_to_528 :
  ∀ n : ℤ, n * n ≠ 529 → |528 - n * n| ≥ |528 - 529| :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_528_l749_74908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l749_74958

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => 2 * sequenceA n + 4

theorem sequence_general_term (n : ℕ) : 
  sequenceA n = 5 * 2^n - 4 :=
by
  induction n with
  | zero => 
    simp [sequenceA]
    norm_num
  | succ n ih =>
    simp [sequenceA]
    rw [ih]
    ring
    sorry  -- Complete the proof later

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l749_74958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_l749_74902

theorem polynomial_existence (n : ℕ) (hn : n ≥ 2) :
  ∃ (f : Polynomial ℤ), 
    (Polynomial.degree f = n) ∧ 
    (∀ i : ℕ, i < n → i > 0 → f.coeff i ≠ 0) ∧
    (Irreducible f) ∧
    (∀ x : ℤ, ¬ Nat.Prime (Int.natAbs (f.eval x))) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_l749_74902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_experiments_fibonacci_search_l749_74983

/-- Represents the number of experimental points -/
def num_points : ℕ := 33

/-- Represents that the effect is a unimodal function -/
def is_unimodal (f : ℕ → ℝ) : Prop :=
  ∃ x₀ : ℕ, 
    (∀ x y, x < y ∧ y ≤ x₀ → f x < f y) ∧
    (∀ x y, x₀ ≤ x ∧ x < y → f x > f y)

/-- Represents the Fibonacci search method -/
def fibonacci_search (f : ℕ → ℝ) (n : ℕ) : ℕ := 
  sorry

/-- Theorem stating that the maximum number of experiments is 7 -/
theorem max_experiments_fibonacci_search :
  ∀ f : ℕ → ℝ, is_unimodal f →
  fibonacci_search f num_points ≤ 7 :=
by
  sorry

#check max_experiments_fibonacci_search

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_experiments_fibonacci_search_l749_74983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_l749_74992

theorem division_remainder (x y : ℕ) 
  (hx : x > 0)
  (hy : y > 0)
  (h1 : (x : ℝ) / (y : ℝ) = 96.12)
  (h2 : (y : ℝ) = 24.99999999999905) : 
  x % y = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_l749_74992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_imply_a_greater_than_three_l749_74906

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 3 * x else x^2 - 2 * x + 2 * a

-- Define the property of having exactly two symmetric points about the origin
def has_two_symmetric_points (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = -f (-x₁) ∧ f x₂ = -f (-x₂) ∧
  ∀ x : ℝ, f x = -f (-x) → (x = x₁ ∨ x = x₂)

-- State the theorem
theorem symmetric_points_imply_a_greater_than_three (a : ℝ) :
  has_two_symmetric_points (f a) → a > 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_imply_a_greater_than_three_l749_74906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sequence_l749_74952

def is_valid_sequence (seq : List Nat) : Prop :=
  seq.length = 5 ∧
  (∀ i : Fin 4, seq[i.val + 1]! - seq[i.val]! = seq[1]! - seq[0]!) ∧
  (∀ n ∈ seq, n > 0) ∧
  (seq[4]! % 11 = 0)

def matches_encoding (seq : List Nat) : Prop :=
  seq[0]! < 10 ∧
  seq[1]! ≥ 10 ∧ seq[1]! < 20 ∧
  seq[2]! ≥ 10 ∧ seq[2]! < 20 ∧
  seq[3]! ≥ 20 ∧ seq[3]! < 30 ∧
  seq[4]! ≥ 30 ∧ seq[4]! < 40

theorem unique_sequence :
  ∃! seq : List Nat, is_valid_sequence seq ∧ matches_encoding seq ∧ seq = [5, 12, 19, 26, 33] :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sequence_l749_74952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_auction_theorem_l749_74903

/-- Represents a bidder with their bid increment -/
structure Bidder where
  increment : Nat

/-- Represents the auction state -/
structure AuctionState where
  price : Nat
  bidCounts : Fin 4 → Nat

/-- Applies a bid to the auction state -/
def applyBid (state : AuctionState) (bidder : Fin 4) (bidders : Fin 4 → Bidder) : AuctionState :=
  { price := state.price + (bidders bidder).increment,
    bidCounts := λ i => if i = bidder then state.bidCounts i + 1 else state.bidCounts i }

/-- Theorem stating that there exists a sequence of bids satisfying the conditions -/
theorem auction_theorem (bidders : Fin 4 → Bidder) 
  (h1 : (bidders 0).increment = 5)
  (h2 : (bidders 1).increment = 10)
  (h3 : (bidders 2).increment = 15)
  (h4 : (bidders 3).increment = 20) :
  ∃ (sequence : List (Fin 4)), 
    let finalState := sequence.foldl (λ state bidder => applyBid state bidder bidders) 
      { price := 15, bidCounts := λ _ => 0 }
    finalState.price = 165 ∧ 
    ∀ bidder, finalState.bidCounts bidder = 3 := by
  sorry

#eval "Auction theorem placeholder"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_auction_theorem_l749_74903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_cost_l749_74914

-- Define the constants
def apple_cost : ℚ := 1
def orange_cost : ℚ := 2
def banana_cost : ℚ := 1/2
def grape_cost : ℚ := 4
def apple_quantity : ℕ := 5
def orange_quantity : ℕ := 2
def banana_quantity : ℕ := 3
def fruit_discount : ℚ := 1/10
def total_discount : ℚ := 1/5
def tax_rate : ℚ := 2/25

-- Define the theorem
theorem shopping_cost : 
  (let total_before_discount := apple_cost * apple_quantity + orange_cost * orange_quantity + 
                               banana_cost * banana_quantity + grape_cost
   let fruit_discount_amount := fruit_discount * total_before_discount
   let after_fruit_discount := total_before_discount - fruit_discount_amount
   let total_discount_amount := total_discount * after_fruit_discount
   let after_total_discount := after_fruit_discount - total_discount_amount
   let tax_amount := tax_rate * after_total_discount
   let final_cost := after_total_discount + tax_amount
   final_cost) = 282/25 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_cost_l749_74914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_count_l749_74994

theorem four_digit_numbers_count : ∃ (count : ℕ), count = 48 :=
  let digits : Finset ℕ := {0, 1, 2, 3, 4}
  let valid_first_digits : Finset ℕ := {3, 4}
  let count := valid_first_digits.card * (digits.card - 1).factorial
  ⟨count, by sorry⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_count_l749_74994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_theorem_l749_74961

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola defined by its vertex and focus -/
structure Parabola where
  vertex : Point
  focus : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is in the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Check if a point lies on a parabola -/
noncomputable def isOnParabola (p : Point) (para : Parabola) : Prop :=
  distance p para.focus = |p.y - (2 * para.vertex.y - para.focus.y)|

/-- The main theorem to prove -/
theorem parabola_point_theorem (para : Parabola) (p : Point) :
  para.vertex = Point.mk (-2) 3 →
  para.focus = Point.mk (-2) 4 →
  isInFirstQuadrant p →
  isOnParabola p para →
  distance p para.focus = 17 →
  p = Point.mk 6 19 := by
  sorry

#check parabola_point_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_theorem_l749_74961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sin_properties_l749_74968

noncomputable def f (x : ℝ) := |Real.sin x|

theorem abs_sin_properties :
  (∃ (p : ℝ), p > 0 ∧ p = π ∧ ∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (x y : ℝ), π/2 < x ∧ x < y ∧ y < π → f y < f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sin_properties_l749_74968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_line_intersection_l749_74966

-- Define the circle equation
def circle_equation (x y a b : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x + 2*b*y + 1 = 0

-- Define the line equation
def line_equation (x y a b : ℝ) : Prop :=
  a*x + y - b = 0

-- Define the first quadrant
def first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

-- Theorem statement
theorem circle_center_and_line_intersection
  (a b : ℝ) 
  (h_center : first_quadrant a (-b)) :
  ∃ (x y : ℝ), line_equation x y a b ∧ first_quadrant x y :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_line_intersection_l749_74966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_l749_74999

/-- Given a quadrilateral ABCD with known side lengths, prove that the diagonal AC has length √(220/7) -/
theorem diagonal_length (A B C D : EuclideanSpace ℝ (Fin 2)) 
  (h1 : ‖A - B‖ = 7)
  (h2 : ‖A - D‖ = 6)
  (h3 : ‖B - C‖ = 6)
  (h4 : ‖C - D‖ = 8)
  (h5 : ‖B - D‖ = 5) :
  ‖A - C‖ = Real.sqrt (220 / 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_l749_74999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turban_price_is_10_l749_74948

/-- The price of the turban in rupees -/
noncomputable def turban_price : ℝ := 10

/-- The total salary for one year in rupees -/
noncomputable def total_salary : ℝ := 90 + turban_price

/-- The fraction of the year worked by the servant -/
noncomputable def fraction_worked : ℝ := 3 / 4

/-- The amount received by the servant for 9 months of work in rupees -/
noncomputable def amount_received : ℝ := 65 + turban_price

/-- Theorem stating that the fraction of the total salary equals the amount received -/
theorem turban_price_is_10 :
  fraction_worked * total_salary = amount_received :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_turban_price_is_10_l749_74948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_cylindrical_example_l749_74949

/-- Converts rectangular coordinates to cylindrical coordinates -/
noncomputable def rect_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 then Real.arctan (y / x) + Real.pi
           else if y ≥ 0 then Real.pi / 2
           else 3 * Real.pi / 2
  (r, θ, z)

theorem rect_to_cylindrical_example :
  let (r, θ, z) := rect_to_cylindrical 3 (-3) 4
  r = 3 * Real.sqrt 2 ∧ θ = 7 * Real.pi / 4 ∧ z = 4 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_cylindrical_example_l749_74949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_proof_l749_74915

-- Define the circles and their properties
def circle_D (diameter : ℝ) : Prop := diameter = 20

-- Define circle C as being inside circle D
def circle_C_inside_D (radius_C radius_D : ℝ) : Prop := radius_C < radius_D

-- Define the ratio of areas
noncomputable def area_ratio (radius_C radius_D : ℝ) : Prop := 
  (Real.pi * radius_D^2 - Real.pi * radius_C^2) / (Real.pi * radius_C^2) = 4

-- The theorem to prove
theorem circle_diameter_proof (diameter_D radius_C radius_D : ℝ) :
  circle_D diameter_D →
  circle_C_inside_D radius_C radius_D →
  area_ratio radius_C radius_D →
  2 * radius_C = 4 * Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_proof_l749_74915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_volume_ratio_l749_74957

/-- The volume of a right circular cylinder with given height and circumference. -/
noncomputable def cylinderVolume (height : ℝ) (circumference : ℝ) : ℝ :=
  (height * circumference^2) / (4 * Real.pi)

/-- The theorem stating that the volume of tank M is 80% of the volume of tank B. -/
theorem tank_volume_ratio :
  let tankM := cylinderVolume 10 8
  let tankB := cylinderVolume 8 10
  tankM / tankB = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_volume_ratio_l749_74957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_francesca_lemonade_calories_l749_74967

/-- Represents the ingredients and their properties in Francesca's lemonade recipe --/
structure LemonadeRecipe where
  lemon_juice : ℚ
  sugar : ℚ
  water : ℚ
  lemon_juice_calories : ℚ
  sugar_calories : ℚ

/-- Calculates the calories in a given amount of lemonade --/
def calories_in_lemonade (recipe : LemonadeRecipe) (amount : ℚ) : ℚ :=
  let total_weight := recipe.lemon_juice + recipe.sugar + recipe.water
  let total_calories := recipe.lemon_juice_calories * (recipe.lemon_juice / 100) + 
                        recipe.sugar_calories * (recipe.sugar / 100)
  (total_calories / total_weight) * amount

/-- Theorem stating that 200g of Francesca's lemonade contains 137 calories --/
theorem francesca_lemonade_calories :
  let recipe : LemonadeRecipe := {
    lemon_juice := 100,
    sugar := 100,
    water := 400,
    lemon_juice_calories := 25,
    sugar_calories := 386
  }
  calories_in_lemonade recipe 200 = 137 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_francesca_lemonade_calories_l749_74967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l749_74964

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3 * x + 1) / (2 - x)

-- Theorem statement
theorem range_of_f :
  Set.range f = {y : ℝ | y ≠ -3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l749_74964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l749_74929

noncomputable def f (x : ℝ) := 2 * Real.sin (1/3 * x - Real.pi/6)

theorem problem_solution (α β : ℝ) 
  (h_alpha : 0 ≤ α ∧ α ≤ Real.pi/2)
  (h_beta : 0 ≤ β ∧ β ≤ Real.pi/2)
  (h_f_alpha : f (3*α + Real.pi/2) = 10/13)
  (h_f_beta : f (3*β + 2*Real.pi) = 6/5) :
  f (5*Real.pi/4) = Real.sqrt 2 ∧ Real.cos (α + β) = 16/65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l749_74929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_payment_l749_74917

/-- The amount Jane paid to the cashier -/
def amount_paid : ℕ → ℕ := sorry

/-- The number of skirts Jane bought -/
def num_skirts : ℕ := 2

/-- The price of each skirt -/
def skirt_price : ℕ := 13

/-- The number of blouses Jane bought -/
def num_blouses : ℕ := 3

/-- The price of each blouse -/
def blouse_price : ℕ := 6

/-- The amount of change Jane received -/
def change_received : ℕ := 56

/-- Theorem stating that Jane paid $100 to the cashier -/
theorem jane_payment :
  amount_paid 0 = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_payment_l749_74917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_sum_is_five_l749_74996

def letter_value (n : ℕ) : ℤ :=
  match n % 10 with
  | 0 => 2
  | 1 => 3
  | 2 => 2
  | 3 => 1
  | 4 => 0
  | 5 => -1
  | 6 => -2
  | 7 => -3
  | 8 => -2
  | 9 => -1
  | _ => 0  -- This case should never occur due to % 10

def letter_position (c : Char) : ℕ :=
  match c with
  | 'p' => 16
  | 'r' => 18
  | 'o' => 15
  | 'b' => 2
  | 'l' => 12
  | 'e' => 5
  | 'm' => 13
  | _ => 0  -- This case should never occur for our specific problem

def problem_sum : ℤ :=
  List.sum (List.map (λ c => letter_value (letter_position c)) ['p', 'r', 'o', 'b', 'l', 'e', 'm'])

theorem problem_sum_is_five : problem_sum = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_sum_is_five_l749_74996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walker_speed_l749_74947

/-- Represents a track with straight sides and semicircular ends -/
structure Track where
  innerRadius : ℝ
  width : ℝ
  straightLength : ℝ

/-- Calculates the time difference between walking the outer and inner edges of the track -/
noncomputable def timeDifference (track : Track) (speed : ℝ) : ℝ :=
  (2 * track.straightLength + 2 * Real.pi * (track.innerRadius + track.width)) / speed -
  (2 * track.straightLength + 2 * Real.pi * track.innerRadius) / speed

/-- The theorem stating the walker's speed given the track conditions -/
theorem walker_speed (track : Track) (speed : ℝ) :
  track.width = 8 ∧ timeDifference track speed = 48 → speed = Real.pi / 3 := by
  sorry

#check walker_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walker_speed_l749_74947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_train_speed_l749_74991

-- Define the properties of the trains
noncomputable def train1_length : ℝ := 240
noncomputable def train1_time : ℝ := 16
noncomputable def train2_length : ℝ := 320
noncomputable def train2_time : ℝ := 20

-- Define the speeds of the trains
noncomputable def speed1 : ℝ := train1_length / train1_time
noncomputable def speed2 : ℝ := train2_length / train2_time

-- Define the combined speed
noncomputable def combined_speed : ℝ := speed1 + speed2

-- Theorem statement
theorem combined_train_speed : combined_speed = 31 := by
  -- Unfold definitions
  unfold combined_speed speed1 speed2 train1_length train1_time train2_length train2_time
  -- Perform arithmetic
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_train_speed_l749_74991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_rpq_l749_74962

theorem sum_of_rpq (t : ℝ) (p q r : ℕ) (h1 : (1 + Real.sin t) * (1 + Real.cos t) = 9/4)
  (h2 : (1 - Real.sin t) * (1 - Real.cos t) = (p : ℝ)/q - Real.sqrt (r : ℝ))
  (h3 : Nat.Coprime p q) (h4 : p > 0) (h5 : q > 0) (h6 : r > 0) : r + p + q = 46 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_rpq_l749_74962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paving_stones_for_courtyard_l749_74942

/-- The number of paving stones required to cover a rectangular courtyard -/
def paving_stones_required (courtyard_length courtyard_width stone_length stone_width : ℚ) : ℕ :=
  (courtyard_length * courtyard_width / (stone_length * stone_width)).ceil.toNat

/-- Theorem stating the number of paving stones required for the given dimensions -/
theorem paving_stones_for_courtyard :
  paving_stones_required 75 (20 + 3/4) (3 + 1/4) (2 + 1/2) = 192 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paving_stones_for_courtyard_l749_74942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_6_44_l749_74960

/-- The acute angle formed by the hands of a clock at 6:44 -/
noncomputable def clock_angle : ℝ :=
  let hours_on_clock : ℕ := 12
  let degrees_per_hour : ℝ := 360 / hours_on_clock
  let minutes_past : ℝ := 44
  let hour_hand_position : ℝ := 6 + minutes_past / 60
  let minute_hand_position : ℝ := minutes_past / 5
  let angle := |hour_hand_position * degrees_per_hour - minute_hand_position * degrees_per_hour|
  min angle (360 - angle)

theorem clock_angle_at_6_44 : clock_angle = 62 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_6_44_l749_74960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_and_area_ratio_l749_74973

-- Define the curves and points
def C₁ (x y : ℝ) : Prop := x^2/4 + y^2 = 1
def C₂ (x y : ℝ) : Prop := y = x^2 - 1
def M : ℝ × ℝ := (0, -1)
def O : ℝ × ℝ := (0, 0)

-- Define the line l passing through the origin
def line_through_origin (k : ℝ) (x y : ℝ) : Prop := y = k * x

-- Define the intersection points
def intersect_C₂_with_line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ C₂ x y ∧ line_through_origin k x y}

-- Define the area of a triangle
noncomputable def triangle_area (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem perpendicular_and_area_ratio :
  ∃ (k : ℝ) (A B D E : ℝ × ℝ),
    A ∈ intersect_C₂_with_line k ∧
    B ∈ intersect_C₂_with_line k ∧
    C₁ D.1 D.2 ∧
    C₁ E.1 E.2 ∧
    (∃ (t : ℝ), D = t • (A - M) + M) ∧
    (∃ (t : ℝ), E = t • (B - M) + M) ∧
    ((D.1 - M.1) * (E.1 - M.1) + (D.2 - M.2) * (E.2 - M.2) = 0) ∧
    (triangle_area M A B / triangle_area M D E = 17/32) ∧
    (k = 3/2 ∨ k = -3/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_and_area_ratio_l749_74973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inequality_l749_74921

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def area (t : Triangle) : ℝ := sorry

def incenter (t : Triangle) : ℝ × ℝ := sorry

def excircle (t : Triangle) (vertex : Fin 3) : Set (ℝ × ℝ) := sorry

def tangentLength (point : ℝ × ℝ) (circle : Set (ℝ × ℝ)) : ℝ := sorry

theorem triangle_area_inequality 
  (ABC DEF : Triangle)
  (h1 : ∀ (v : Fin 3), ∃ (side : Fin 3), 
    (match v with
    | 0 => DEF.A
    | 1 => DEF.B
    | _ => DEF.C) ∈ Set.Icc 
      (match side with
      | 0 => ABC.A
      | 1 => ABC.B
      | _ => ABC.C) 
      (match (side + 1 : Fin 3) with
      | 0 => ABC.A
      | 1 => ABC.B
      | _ => ABC.C))
  (h2 : ∀ (v1 v2 : Fin 3), v1 ≠ v2 → 
    tangentLength (incenter DEF) (excircle ABC v1) = 
    tangentLength (incenter DEF) (excircle ABC v2)) :
  4 * area DEF ≥ area ABC := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inequality_l749_74921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_right_triangle_l749_74911

/-- The smallest area of a right-angled triangle with given conditions -/
theorem min_area_right_triangle : ∃ (S : ℝ),
  (∀ (A B : ℝ × ℝ) (k : ℝ),
    (A.1 = A.2) ∧                            -- One leg on y = x
    (B.1 = -B.2) ∧                           -- Other leg on y = -x
    (3 = k * 1 + (3 - k)) ∧                  -- Hypotenuse contains (1, 3)
    (A.2 = k * A.1 + (3 - k)) ∧              -- A is on the hypotenuse
    (B.2 = k * B.1 + (3 - k)) →              -- B is on the hypotenuse
    S ≤ (1/2) * abs (A.1 * (B.1 - A.1))) ∧
  S = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_right_triangle_l749_74911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_is_zero_l749_74920

/-- The limit of the sequence (∛(n³-7) + ∛(n²+4)) / (⁴√(n⁵+5) + √n) as n approaches infinity is 0. -/
theorem limit_of_sequence_is_zero : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |((n^3 - 7 : ℝ)^(1/3) + (n^2 + 4 : ℝ)^(1/3)) / ((n^5 + 5 : ℝ)^(1/4) + n^(1/2)) - 0| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_is_zero_l749_74920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l749_74997

-- Define the given parameters
noncomputable def train_length : ℝ := 250
noncomputable def train_speed_kmph : ℝ := 72
noncomputable def crossing_time : ℝ := 20

-- Define the function to convert km/h to m/s
noncomputable def kmph_to_mps (speed : ℝ) : ℝ := speed * (1000 / 3600)

-- Define the theorem
theorem platform_length_calculation :
  let train_speed_mps := kmph_to_mps train_speed_kmph
  let total_distance := train_speed_mps * crossing_time
  let platform_length := total_distance - train_length
  platform_length = 150 := by
  -- Proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l749_74997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l749_74978

-- Define the function f as noncomputable
noncomputable def f (x θ : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x + θ) + Real.cos (2 * x + θ)

-- State the theorem
theorem min_value_of_f (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π) 
  (h_sym : ∀ x, f x θ = f (π - x) θ) : 
  ∃ x₀ ∈ Set.Icc (-π/4 : ℝ) (π/6 : ℝ), f x₀ θ = -Real.sqrt 3 ∧ 
  ∀ x ∈ Set.Icc (-π/4 : ℝ) (π/6 : ℝ), f x θ ≥ -Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l749_74978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l749_74980

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x - 4 * y - 15 = 0

-- Define the circle C
def circle_C (x y r : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 5 - r^2 = 0

-- Define the intersection of line l and circle C
def intersects (A B : ℝ × ℝ) (r : ℝ) : Prop :=
  line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
  circle_C A.1 A.2 r ∧ circle_C B.1 B.2 r

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem circle_equation (A B : ℝ × ℝ) (r : ℝ) :
  r > 0 →
  intersects A B r →
  distance A B = 6 →
  ∀ (x y : ℝ), circle_C x y r ↔ (x - 1)^2 + (y - 2)^2 = 25 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l749_74980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_symmetry_origin_cubic_symmetry_vertical_shift_general_cubic_symmetry_l749_74909

/-- A function has a center of symmetry at (a,b) if f(a+x) + f(a-x) = 2*f(a) for all x -/
def has_center_of_symmetry (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (a + x) + f (a - x) = 2 * f a

/-- The graph of x^3 + px has a center of symmetry at (0,0) for any real p -/
theorem cubic_symmetry_origin (p : ℝ) :
  has_center_of_symmetry (fun x ↦ x^3 + p*x) 0 0 := by sorry

/-- The graph of x^3 + px + q has a center of symmetry at (0,q) for any real p and q -/
theorem cubic_symmetry_vertical_shift (p q : ℝ) :
  has_center_of_symmetry (fun x ↦ x^3 + p*x + q) 0 q := by sorry

/-- The graph of ax^3 + bx^2 + cx + d has a center of symmetry for any real a, b, c, and d -/
theorem general_cubic_symmetry (a b c d : ℝ) :
  ∃ x y, has_center_of_symmetry (fun t ↦ a*t^3 + b*t^2 + c*t + d) x y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_symmetry_origin_cubic_symmetry_vertical_shift_general_cubic_symmetry_l749_74909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l749_74928

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The distance from the origin to the line connecting the ellipse's intersections with positive x and y axes -/
noncomputable def distance_to_intersection_line (e : Ellipse) : ℝ := 2 * Real.sqrt 5 / 5

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt 3 / 2

/-- A line passing through a point (0, y) -/
structure Line where
  y : ℝ

/-- The y-coordinate of the point where the line intersects the y-axis -/
def y_intercept (l : Line) : ℝ := l.y

/-- The range of y-intercepts for perpendicular bisectors -/
def y_intercept_range : Set ℝ := Set.Ioc (-(9/5)) 0

/-- Calculate the y-intercept of the perpendicular bisector -/
noncomputable def perpendicular_bisector_y_intercept (e : Ellipse) (l : Line) : ℝ := 
  sorry -- Implementation would go here

theorem ellipse_properties (e : Ellipse) 
  (h_dist : distance_to_intersection_line e = 2 * Real.sqrt 5 / 5)
  (h_ecc : eccentricity e = Real.sqrt 3 / 2) :
  (∃ x y, x^2/4 + y^2 = 1 ↔ x^2/e.a^2 + y^2/e.b^2 = 1) ∧
  (∀ l : Line, l.y = 5/3 → 
    ∃ y_perp_bisector, y_perp_bisector ∈ y_intercept_range ↔ 
      y_perp_bisector = y_intercept (Line.mk (perpendicular_bisector_y_intercept e l))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l749_74928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_conjunction_false_l749_74941

/-- The smallest positive period of the function y = sin 2x is π -/
def proposition_p : Prop := 
  ∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), Real.sin (2 * x) = Real.sin (2 * (x + T))

/-- The graph of the function y = cos x is symmetric about the line x = π -/
def proposition_q : Prop := 
  ∀ (x : ℝ), Real.cos x = Real.cos (2 * Real.pi - x)

/-- The conjunction of proposition_p and proposition_q is false -/
theorem propositions_conjunction_false : ¬(proposition_p ∧ proposition_q) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_conjunction_false_l749_74941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_for_inclination_angle_l749_74905

theorem slope_range_for_inclination_angle (α : Real) (h : α ∈ Set.Ioo (π/3) (5*π/6)) :
  let k := Real.tan α
  k ∈ Set.union (Set.Ioi (-Real.sqrt 3 / 3)) (Set.Iic (Real.sqrt 3))ᶜ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_for_inclination_angle_l749_74905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l749_74977

-- Define the original function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the function C (translated f)
noncomputable def C (x : ℝ) : ℝ := f (x - 2)

-- Define the exponential function
noncomputable def exp2 (x : ℝ) : ℝ := 2^x

-- State the theorem
theorem function_symmetry :
  (∀ x, C x = -(exp2 x)) →  -- C is symmetric to 2^x about x-axis
  (∀ x, f x = -(2^(x + 2))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l749_74977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l749_74955

/-- The ellipse C in the Cartesian coordinate system -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 12 + y^2 / 6 = 1

/-- The circle to which tangent lines are drawn -/
def circle_tangent (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- Point P on the ellipse -/
noncomputable def point_P : ℝ × ℝ := (2 * Real.sqrt 3, 0)

/-- Function to calculate the area of triangle PAB -/
noncomputable def triangle_area (px py : ℝ) : ℝ :=
  let m := (px - 2 + Real.sqrt ((px - 2)^2 + 4 * py^2)) / (2 * (px - 2))
  let n := (px - 2 - Real.sqrt ((px - 2)^2 + 4 * py^2)) / (2 * (px - 2))
  1/2 * (m - n) * px

/-- Main theorem statement -/
theorem max_triangle_area :
  ellipse_C (Real.sqrt 2) (Real.sqrt 5) →
  (∀ x y, ellipse_C x y → x > 2 → triangle_area x y ≤ triangle_area point_P.1 point_P.2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l749_74955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_contest_results_l749_74939

-- Define the number of students
def num_students : Nat := 8

-- Define the number of bananas eaten by each student
def ben_bananas : Nat := 9
def kim_bananas : Nat := 2
def alice_bananas : Nat := 3
def john_bananas : Nat := 5
def lisa_bananas : Nat := 4
def mark_bananas : Nat := 6
def chris_bananas : Nat := 7
def olivia_bananas : Nat := 5

-- Define the total number of bananas eaten
def total_bananas : Nat := ben_bananas + kim_bananas + alice_bananas + john_bananas + 
                           lisa_bananas + mark_bananas + chris_bananas + olivia_bananas

-- Define the average number of bananas eaten
noncomputable def average_bananas : ℚ := (total_bananas : ℚ) / num_students

-- Theorem statement
theorem banana_contest_results : 
  (ben_bananas - kim_bananas = 7) ∧ 
  ((ben_bananas : ℚ) - average_bananas = 31/8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_contest_results_l749_74939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_elephantPlacements_l749_74925

/-- 
Represents a chessboard configuration where elephants (bishops) are placed
such that no two elephants attack each other.
-/
def ValidElephantPlacement (n : ℕ) := Fin n → Fin 2

/-- 
The number of valid elephant placements on a 2 × n chessboard.
-/
def numValidPlacements (n : ℕ) : ℕ := 2^n

/-- 
Theorem stating that the number of valid elephant placements
on a 2 × n chessboard is 2^n.
-/
theorem elephantPlacements (n : ℕ) : numValidPlacements n = 2^n := by
  rfl

#eval numValidPlacements 2015

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_elephantPlacements_l749_74925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l749_74918

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Add conditions here
  b = 2 ∧ 
  B = Real.pi / 3 ∧
  Real.sin (2 * A) + Real.sin (A - C) = Real.sin B

-- Define the area function
noncomputable def area_triangle (A B C a b c : ℝ) : ℝ :=
  1 / 2 * a * b * Real.sin C

-- State the theorem
theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  triangle_ABC A B C a b c →
  (area_triangle A B C a b c = Real.sqrt 3 ∨ 
   area_triangle A B C a b c = 2 * Real.sqrt 3 / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l749_74918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l749_74901

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.sin (Real.pi * x / 2)
  else 1/6 - Real.log x / Real.log 3

theorem f_composition_value :
  f (f (3 * Real.sqrt 3)) = -(Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l749_74901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_not_equal_l749_74926

-- Define the function pairs
noncomputable def y1_1 (x : ℝ) : ℝ := (x + 3) * (x - 5) / (x + 3)
def y2_1 (x : ℝ) : ℝ := x - 5

noncomputable def y1_2 (x : ℝ) : ℝ := Real.sqrt (x + 1) * Real.sqrt (x - 1)
noncomputable def y2_2 (x : ℝ) : ℝ := Real.sqrt ((x + 1) * (x - 1))

def f3 (x : ℝ) : ℝ := x
noncomputable def g3 (x : ℝ) : ℝ := Real.sqrt (x^2)

def f4 (x : ℝ) : ℝ := 3 * x^4 - x^3
noncomputable def F4 (x : ℝ) : ℝ := x^3 * Real.sqrt (x - 1)

noncomputable def f1_5 (x : ℝ) : ℝ := (Real.sqrt (2 * x - 5))^2
def f2_5 (x : ℝ) : ℝ := 2 * x - 5

-- Theorem stating that none of the function pairs are equal
theorem functions_not_equal :
  (∃ x, y1_1 x ≠ y2_1 x ∨ x = -3) ∧
  (∃ x, y1_2 x ≠ y2_2 x) ∧
  (∃ x, f3 x ≠ g3 x) ∧
  (∃ x, f4 x ≠ F4 x) ∧
  (∃ x, f1_5 x ≠ f2_5 x ∨ x < 5/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_not_equal_l749_74926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l749_74988

/-- The length of a bridge crossed by a man -/
noncomputable def bridge_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time / 60

/-- Theorem: The length of a bridge is 2.25 km when a man walking at 9 km/hr crosses it in 15 minutes -/
theorem bridge_length_calculation :
  bridge_length 9 15 = 2.25 := by
  -- Unfold the definition of bridge_length
  unfold bridge_length
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l749_74988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_properties_two_zeros_condition_l749_74900

noncomputable section

open Real Set

def f (a : ℝ) (x : ℝ) : ℝ := log x / log a

def g (a : ℝ) (x : ℝ) : ℝ := a^x

def h (x : ℝ) : ℝ := x / exp x

theorem h_properties :
  (∀ x ∈ Ioo 0 1, MonotoneOn h (Ioo 0 1)) ∧
  (∀ x ∈ Ioi 1, AntitoneOn h (Ioi 1)) ∧
  (∃ x₀ ∈ Ioi 0, ∀ x ∈ Ioi 0, h x ≤ h x₀) ∧
  (∀ c : ℝ, ∃ x : ℝ, x > 0 ∧ h x < c) :=
sorry

def has_two_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g a x₁ - f a x₁ / x₁ = 1 ∧ g a x₂ - f a x₂ / x₂ = 1

theorem two_zeros_condition (a : ℝ) :
  has_two_zeros a ↔ 1 < a ∧ a < exp (1 / exp 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_properties_two_zeros_condition_l749_74900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_time_approx_2_02_years_l749_74990

/-- Calculates the time required for an investment to reach a specific amount -/
noncomputable def investment_time (principal : ℝ) (rate : ℝ) (compound_freq : ℝ) (final_amount : ℝ) : ℝ :=
  Real.log (final_amount / principal) / (compound_freq * Real.log (1 + rate / compound_freq))

/-- Theorem stating the investment time for given conditions -/
theorem investment_time_approx_2_02_years :
  let principal : ℝ := 10000
  let rate : ℝ := 0.0396
  let compound_freq : ℝ := 2
  let final_amount : ℝ := 10815.834432633617
  abs (investment_time principal rate compound_freq final_amount - 2.02) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_time_approx_2_02_years_l749_74990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_l749_74954

def regular_price : ℝ := 50
def num_shirts : ℕ := 6
def sale_price : ℝ := 240

theorem discount_percentage : 
  (regular_price * (num_shirts : ℝ) - sale_price) / (regular_price * (num_shirts : ℝ)) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_l749_74954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_EFGH_area_l749_74971

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a trapezoid defined by four vertices -/
structure Trapezoid where
  E : Point
  F : Point
  G : Point
  H : Point

/-- Calculates the area of a trapezoid given its vertices -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ :=
  let height := t.G.x - t.E.x
  let base1 := t.F.y - t.E.y
  let base2 := t.G.y - t.H.y
  (base1 + base2) * height / 2

/-- Theorem: The area of trapezoid EFGH with given vertices is 30 square units -/
theorem trapezoid_EFGH_area :
  let t : Trapezoid := {
    E := { x := -2, y := -3 },
    F := { x := -2, y := 2 },
    G := { x := 4, y := 5 },
    H := { x := 4, y := 0 }
  }
  trapezoidArea t = 30 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_EFGH_area_l749_74971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_end_pattern_factorial_start_pattern_l749_74916

-- Define a function to check if a number ends with 2004 followed by 0s or 4s
def endsWithPattern (n : ℕ) : Prop :=
  ∃ k : ℕ, n % (10^(k+4)) = 2004 * 10^k ∧ 
    (∀ i : ℕ, i < k → (n / 10^i) % 10 = 0 ∨ (n / 10^i) % 10 = 4)

-- Define a function to check if a number starts with 2004
def startsWithPattern (n : ℕ) : Prop :=
  ∃ k : ℕ, 2004 * 10^k ≤ n ∧ n < 2005 * 10^k

theorem factorial_end_pattern :
  ∀ n : ℕ, n > 0 → ¬ endsWithPattern (n!) := by
  sorry

theorem factorial_start_pattern :
  ∃ n : ℕ, n > 0 ∧ startsWithPattern (n!) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_end_pattern_factorial_start_pattern_l749_74916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_vectors_l749_74923

/-- Given two unit vectors e₁ and e₂ with an angle α between them where cos α = 1/3,
    and vectors a = 3e₁ - 2e₂ and b = 3e₁ - e₂, prove that the cosine of the angle β
    between a and b is 2√2/3 -/
theorem cosine_of_angle_between_vectors (e₁ e₂ : EuclideanSpace ℝ (Fin 2)) (α β : ℝ) (a b : EuclideanSpace ℝ (Fin 2)) :
  ‖e₁‖ = 1 →
  ‖e₂‖ = 1 →
  inner e₁ e₂ = Real.cos α →
  Real.cos α = 1/3 →
  a = 3 • e₁ - 2 • e₂ →
  b = 3 • e₁ - e₂ →
  Real.cos β = (inner a b) / (‖a‖ * ‖b‖) →
  Real.cos β = 2 * Real.sqrt 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_vectors_l749_74923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_with_prime_absolute_difference_l749_74924

theorem unique_prime_with_prime_absolute_difference : ∃! p : ℤ, Nat.Prime p.natAbs ∧ Nat.Prime (|p^4 - 86|).natAbs := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_with_prime_absolute_difference_l749_74924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_l749_74936

-- Auxiliary definitions
def Sphere : Type := Real → Real → Real → Prop

def SphereTangentToBase (s : Sphere) (base_radius : Real) : Prop := sorry

def SphereTangentToLateralSurface (s : Sphere) (apex_angle : Real) : Prop := sorry

def SphereTangentToSphere (s1 s2 : Sphere) : Prop := sorry

theorem cone_base_radius (apex_angle : Real) (sphere_radius : Real) :
  apex_angle = π / 3 →
  sphere_radius = 1 →
  ∃ (base_radius : Real),
    base_radius = (5 : Real) / Real.sqrt 3 ∧
    (∃ (sphere1 sphere2 sphere3 : Sphere),
      (∀ x y z, sphere1 x y z ↔ (x^2 + y^2 + z^2 = sphere_radius^2)) ∧
      (∀ x y z, sphere2 x y z ↔ (x^2 + y^2 + z^2 = sphere_radius^2)) ∧
      (∀ x y z, sphere3 x y z ↔ (x^2 + y^2 + z^2 = sphere_radius^2)) ∧
      SphereTangentToBase sphere1 base_radius ∧
      SphereTangentToBase sphere2 base_radius ∧
      SphereTangentToBase sphere3 base_radius ∧
      SphereTangentToLateralSurface sphere1 apex_angle ∧
      SphereTangentToLateralSurface sphere2 apex_angle ∧
      SphereTangentToLateralSurface sphere3 apex_angle ∧
      SphereTangentToSphere sphere1 sphere2 ∧
      SphereTangentToSphere sphere2 sphere3 ∧
      SphereTangentToSphere sphere3 sphere1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_l749_74936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_11_not_in_set_ge_4_l749_74981

theorem sqrt_11_not_in_set_ge_4 : Real.sqrt 11 ∉ {x : ℝ | x ≥ 4} := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_11_not_in_set_ge_4_l749_74981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucca_bread_problem_l749_74969

/-- The fraction of remaining bread Lucca ate on the third day -/
theorem lucca_bread_problem (initial_bread : ℕ) 
  (first_day_fraction : ℚ) (second_day_fraction : ℚ) (remaining_bread : ℕ)
  (h1 : initial_bread = 200)
  (h2 : first_day_fraction = 1 / 4)
  (h3 : second_day_fraction = 2 / 5)
  (h4 : remaining_bread = 45) : 
  let first_day_remaining := initial_bread - (first_day_fraction * initial_bread).floor
  let second_day_remaining := first_day_remaining - (second_day_fraction * first_day_remaining).floor
  (↑(second_day_remaining - remaining_bread) : ℚ) / second_day_remaining = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucca_bread_problem_l749_74969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l749_74945

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists with sides a, b, c opposite to angles A, B, C
  (0 < A ∧ A < Real.pi) ∧ (0 < B ∧ B < Real.pi) ∧ (0 < C ∧ C < Real.pi) ∧ 
  (A + B + C = Real.pi) ∧
  (a > 0 ∧ b > 0 ∧ c > 0) ∧
  -- Vectors m and n are parallel
  (1 * (1 + Real.cos A) = 2 * Real.sin A * Real.sin A) ∧
  -- Condition on side lengths
  (b + c = Real.sqrt 3 * a) →
  -- Conclusions
  (A = Real.pi / 3) ∧ 
  (Real.sin (B + Real.pi / 6) = Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l749_74945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lower_bound_l749_74982

noncomputable def a : ℕ → ℝ
| 0 => 1
| n + 1 => a n / (a n ^ 3 + 1)

theorem a_lower_bound (n : ℕ) : 
  a n > 1 / (3 * n + Real.log (n + 1) + 14/9) ^ (1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lower_bound_l749_74982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_color_packing_l749_74965

/-- Packing objects of different colors into boxes -/
theorem color_packing (n k : ℕ) (hn : n > 0) (hk : k > 0) :
  ∃ (packing : Fin k → Fin (n*k) → Fin k),
    (∀ (box : Fin k), ∃ (c1 c2 : Fin k),
      ∀ (obj : Fin (n*k)),
        packing box obj = c1 ∨ packing box obj = c2) ∧
    (∀ (obj : Fin (n*k)), ∃! (box : Fin k), packing box obj = obj.val % k) ∧
    (∀ (box : Fin k), (Finset.filter (λ obj => packing box obj = obj.val % k) (Finset.univ : Finset (Fin (n*k)))).card ≤ n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_color_packing_l749_74965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_l749_74944

/-- A primitive fifth root of unity -/
noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

/-- The statement to be proven -/
theorem no_integer_solution :
  ¬ ∃ (a b c d k : ℤ), k > 1 ∧ (a + b * ω + c * ω^2 + d * ω^3)^k = 1 + ω := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_l749_74944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_2_power_2010_l749_74989

def last_digit_of_2_power (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 2
  | 2 => 4
  | _ => 8

theorem last_digit_2_power_2010 :
  last_digit_of_2_power 2010 = 4 := by
  rfl

#eval last_digit_of_2_power 2010

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_2_power_2010_l749_74989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l749_74995

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the line l
def line_l (x y m : ℝ) : Prop := y = x - m

-- Define the distance between two points
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2)

-- Theorem statement
theorem intersection_distance (m : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ 
    line_l x₁ y₁ m ∧ line_l x₂ y₂ m ∧
    distance x₁ y₁ x₂ y₂ = Real.sqrt 14) → 
  m = 1 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l749_74995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_richter_magnitude_example_l749_74940

/-- The Richter magnitude formula -/
noncomputable def richterMagnitude (A : ℝ) (A₀ : ℝ) : ℝ := Real.log A / Real.log 10 - Real.log A₀ / Real.log 10

/-- Theorem: The Richter magnitude for A = 1000 and A₀ = 0.001 is 6 -/
theorem richter_magnitude_example : richterMagnitude 1000 0.001 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_richter_magnitude_example_l749_74940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_solutions_l749_74959

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then -x else (x - 1)^2

-- State the theorem
theorem f_equals_one_solutions (a : ℝ) :
  f a = 1 ↔ a = 2 ∨ a = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_solutions_l749_74959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_calculations_correctness_l749_74913

noncomputable section

-- Define the functions
def f1 (x : ℝ) := (1/2) * Real.sin (2*x)
def f2 (x : ℝ) := Real.cos (1/x)
def f3 (x : ℝ) := x^2 + Real.exp 2
def f4 (x : ℝ) := Real.log x - 1/x^2

-- Define their derivatives as given in the problem
def f1_deriv (x : ℝ) := Real.cos (2*x)
def f2_deriv (x : ℝ) := -(1/x) * Real.sin (1/x)
def f3_deriv (x : ℝ) := 2*x + Real.exp 2
def f4_deriv (x : ℝ) := 1/x + 2/x^3

-- Theorem statement
theorem derivative_calculations_correctness :
  (∀ x, deriv f1 x = f1_deriv x) ∧
  (∃ x, deriv f2 x ≠ f2_deriv x) ∧
  (∃ x, deriv f3 x ≠ f3_deriv x) ∧
  (∀ x, deriv f4 x = f4_deriv x) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_calculations_correctness_l749_74913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fuel_is_70_gallons_l749_74934

/-- Represents a car with its fuel efficiency and distance to travel -/
structure Car where
  efficiency : ℚ  -- miles per gallon (using rational numbers)
  distance : ℚ    -- miles to travel

/-- Calculates the fuel needed for a given car -/
def fuelNeeded (car : Car) : ℚ :=
  car.distance / car.efficiency

/-- The total fuel needed for two cars -/
def totalFuelNeeded (car1 car2 : Car) : ℚ :=
  fuelNeeded car1 + fuelNeeded car2

/-- Theorem stating that the total fuel needed for Car B and Car C is 70 gallons -/
theorem total_fuel_is_70_gallons :
  let carB : Car := { efficiency := 30, distance := 750 }
  let carC : Car := { efficiency := 20, distance := 900 }
  totalFuelNeeded carB carC = 70 := by
  -- Unfold definitions
  unfold totalFuelNeeded fuelNeeded
  -- Simplify rational number arithmetic
  simp [Car.efficiency, Car.distance]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fuel_is_70_gallons_l749_74934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l749_74970

open Real

noncomputable def f (x : ℝ) : ℝ := tan x ^ 2 + 2 * tan x + 6 * (1 / tan x) + 9 * (1 / tan x) ^ 2 + 4

theorem min_value_of_f :
  ∃ (x : ℝ), 0 < x ∧ x < π / 2 ∧
  (∀ (y : ℝ), 0 < y ∧ y < π / 2 → f y ≥ f x) ∧
  f x = 10 + 4 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l749_74970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l749_74985

/-- The curve C in the Cartesian plane -/
def curve_C (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- The line l in the Cartesian plane -/
def line_l (x y : ℝ) : Prop := x + y = 2

/-- The distance from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x + y - 2| / Real.sqrt 2

/-- The maximum distance from any point on curve C to line l is 2√2 -/
theorem max_distance_curve_to_line :
  ∃ (x y : ℝ), curve_C x y ∧
    (∀ (x' y' : ℝ), curve_C x' y' →
      distance_to_line x y ≥ distance_to_line x' y') ∧
    distance_to_line x y = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l749_74985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_special_case_l749_74986

open Real

theorem cosine_sum_special_case : 
  ∀ α : ℝ, cos (α - 35 * π / 180) * cos (25 * π / 180 + α) + 
           sin (α - 35 * π / 180) * sin (25 * π / 180 + α) = 1/2 := by
  intro α
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_special_case_l749_74986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reservoir_fullness_l749_74907

/-- Represents the capacity and contents of a reservoir before and after a storm --/
structure ReservoirState where
  capacity : ℚ
  initialContents : ℚ
  stormDeposit : ℚ
  cityConsumption : ℚ
  evaporationLoss : ℚ

/-- Calculates the percentage of the reservoir that is full --/
noncomputable def percentageFull (total : ℚ) (capacity : ℚ) : ℚ :=
  (total / capacity) * 100

/-- Theorem about reservoir fullness before and after a storm --/
theorem reservoir_fullness (state : ReservoirState)
  (h1 : state.capacity > 0)
  (h2 : state.stormDeposit = 115)
  (h3 : percentageFull (state.initialContents + state.stormDeposit) state.capacity = 80)
  (h4 : state.initialContents = 245)
  (h5 : state.cityConsumption = 15)
  (h6 : state.evaporationLoss = 5) :
  (∃ (ε₁ ε₂ : ℚ), 
    abs ε₁ < 1/100 ∧ abs ε₂ < 1/100 ∧
    percentageFull state.initialContents state.capacity = 42.61 + ε₁ ∧
    percentageFull (state.initialContents - state.cityConsumption - state.evaporationLoss) state.capacity = 23.48 + ε₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reservoir_fullness_l749_74907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l749_74975

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := y = x + 5 * Real.sqrt 7

/-- The distance function from a point (x, y) to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (y - x - 5 * Real.sqrt 7) / Real.sqrt 2

/-- The maximum distance from any point on the ellipse to the line -/
theorem max_distance_ellipse_to_line :
  ∃ (x y : ℝ), ellipse x y ∧ ∀ (x' y' : ℝ), ellipse x' y' →
    distance_to_line x y ≥ distance_to_line x' y' ∧
    distance_to_line x y = 3 * Real.sqrt 14 := by
  sorry

#check max_distance_ellipse_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l749_74975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_properties_l749_74984

/-- The sum of interior angles of a polygon -/
def sum_interior_angles : ℝ := 3510

/-- The number of sides in the polygon -/
def num_sides : ℕ := 22

/-- The measure of each interior angle if the polygon is regular -/
def interior_angle : ℝ := 159.55

theorem polygon_properties :
  (sum_interior_angles = 180 * (num_sides - 2 : ℝ)) ∧
  (abs (interior_angle - sum_interior_angles / num_sides) < 0.01) := by
  sorry

#check polygon_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_properties_l749_74984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_set_not_line_l749_74946

-- Define the two lines
noncomputable def line1 (x y : ℝ) : ℝ := 4 * x + 3 * y - 5
noncomputable def line2 (x y : ℝ) : ℝ := 4 * x + 3 * y + 10

-- Define the distance from a point to a line
noncomputable def distance_to_line (x y : ℝ) (line : ℝ → ℝ → ℝ) : ℝ :=
  |line x y| / Real.sqrt (4^2 + 3^2)

-- Define the set of points M
noncomputable def point_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | distance_to_line p.1 p.2 line1 + distance_to_line p.1 p.2 line2 = 3}

-- Theorem statement
theorem point_set_not_line : ¬ ∃ (a b c : ℝ), ∀ (p : ℝ × ℝ), p ∈ point_set → a * p.1 + b * p.2 = c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_set_not_line_l749_74946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_minus_alpha_l749_74935

theorem tan_pi_4_minus_alpha (α : Real) 
  (h1 : Real.cos α = -4/5) 
  (h2 : α ∈ Set.Ioo (Real.pi/2) Real.pi) : 
  Real.tan (Real.pi/4 - α) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_minus_alpha_l749_74935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l749_74951

noncomputable def f (x a : ℝ) : ℝ := -Real.pi/x + Real.sin x + a^2 * Real.sin (x + Real.pi/4)

theorem max_value_implies_a (a : ℝ) : 
  (∃ M, M = 3 ∧ ∀ x, f x a ≤ M) → a = Real.sqrt 3 ∨ a = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l749_74951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opponent_total_score_l749_74937

def team_scores : List ℕ := [2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16]

def lost_by_one (score : ℕ) : Bool := score % 2 = 0

def opponent_score_lost_by_one (score : ℕ) : ℕ := score + 1

def opponent_score_won (score : ℕ) : ℚ := score / 2

theorem opponent_total_score : 
  ((team_scores.filter lost_by_one).map opponent_score_lost_by_one).sum +
  ((team_scores.filter (fun x => ¬(lost_by_one x))).map opponent_score_won).sum = 86 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opponent_total_score_l749_74937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_satisfy_conditions_l749_74993

/-- A line in the xy-plane is represented by its x and y intercepts -/
structure Line where
  x_intercept : ℝ
  y_intercept : ℝ

/-- Check if a number is a positive cube less than 30 -/
def is_positive_cube_lt_30 (n : ℝ) : Prop :=
  ∃ k : ℕ, (k : ℝ)^3 = n ∧ 0 < n ∧ n < 30

/-- Check if a line passes through the point (6,4) -/
def passes_through_6_4 (l : Line) : Prop :=
  6 / l.x_intercept + 4 / l.y_intercept = 1

/-- The main theorem -/
theorem two_lines_satisfy_conditions :
  ∃! (s : Finset Line),
    s.card = 2 ∧
    (∀ l ∈ s,
      passes_through_6_4 l ∧
      is_positive_cube_lt_30 l.x_intercept ∧
      l.y_intercept > 0 ∧ Int.floor l.y_intercept = l.y_intercept) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_satisfy_conditions_l749_74993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_points_l749_74979

def point_A : ℝ × ℝ := (1, -1)
def point_B : ℝ × ℝ := (1, 4)
def point_C : ℝ × ℝ := (4, 2)

def circle_equation (x y : ℝ) : Prop :=
  (x - 3/2)^2 + (y - 3/2)^2 = 13/4

noncomputable def circle_center : ℝ × ℝ := (3/2, 3/2)
noncomputable def circle_radius : ℝ := Real.sqrt (13/4)

theorem circle_through_points :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  circle_equation point_C.1 point_C.2 ∧
  (∀ (x y : ℝ), circle_equation x y ↔ ((x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_points_l749_74979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_C_chord_length_l749_74912

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, 1 + 2 * Real.sin α)

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (1 + t * Real.cos (45 * Real.pi / 180), t * Real.sin (45 * Real.pi / 180))

-- Theorem for the polar equation of curve C
theorem polar_equation_C :
  ∀ ρ θ : ℝ,
  (ρ * Real.cos θ)^2 + (ρ * Real.sin θ - 1)^2 = 4 ↔
  ρ^2 - 2*ρ*Real.sin θ - 3 = 0 := by
  sorry

-- Theorem for the length of the chord
theorem chord_length :
  ∃ t₁ t₂ : ℝ,
  (1 + t₁ * Real.cos (45 * Real.pi / 180))^2 + (t₁ * Real.sin (45 * Real.pi / 180) - 1)^2 = 4 ∧
  (1 + t₂ * Real.cos (45 * Real.pi / 180))^2 + (t₂ * Real.sin (45 * Real.pi / 180) - 1)^2 = 4 ∧
  |t₂ - t₁| = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_C_chord_length_l749_74912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_meet_time_l749_74987

/-- Represents the properties of two cars moving towards each other -/
structure TwoCars where
  initial_distance : ℝ
  speed_ratio : ℝ
  speed_b : ℝ

/-- Calculates the time taken for two cars to meet -/
noncomputable def time_to_meet (cars : TwoCars) : ℝ :=
  let speed_a := (cars.speed_ratio * cars.speed_b) / (cars.speed_ratio + 1)
  let relative_speed := speed_a + cars.speed_b
  (cars.initial_distance / relative_speed) * 60 -- Convert to minutes

/-- Theorem stating that the time taken for the cars to meet is approximately 32 minutes -/
theorem cars_meet_time (cars : TwoCars) 
    (h1 : cars.initial_distance = 88)
    (h2 : cars.speed_ratio = 5/6)
    (h3 : cars.speed_b = 90) : 
  ∃ ε > 0, |time_to_meet cars - 32| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_meet_time_l749_74987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_size_lower_bound_l749_74931

/-- Represents a tournament where each pair of teams played a match -/
structure Tournament where
  teams : Type
  won_against : teams → teams → Prop

/-- Theorem stating the lower bounds for tournament sizes under given conditions -/
theorem tournament_size_lower_bound 
  {t : Tournament} 
  [Fintype t.teams]
  (h1 : ∀ a b : t.teams, ∃ c : t.teams, t.won_against c a ∧ t.won_against c b) 
  (h2 : ∀ a b c : t.teams, ∃ d : t.teams, t.won_against d a ∧ t.won_against d b ∧ t.won_against d c) :
  (7 : ℕ) ≤ Fintype.card t.teams ∧ (15 : ℕ) ≤ Fintype.card t.teams :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_size_lower_bound_l749_74931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_two_max_l749_74953

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (Real.pi * x / 6) * Real.cos (Real.pi * x / 6) - Real.sqrt 3 * (Real.sin (Real.pi * x / 6))^2

def has_at_least_two_max (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x₁ x₂, -1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ a ∧
    ∀ x, -1 ≤ x ∧ x ≤ a → f x ≤ f x₁ ∧ f x ≤ f x₂

theorem min_a_for_two_max :
  ∀ a : ℕ, (has_at_least_two_max f a → a ≥ 8) ∧
           (a ≥ 8 → has_at_least_two_max f a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_two_max_l749_74953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_min_at_neg_one_l749_74998

-- Define the function f(x) = xe^x
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

-- State the theorem
theorem f_local_min_at_neg_one : 
  ∃ δ > 0, ∀ x ∈ Set.Ioo (-1 - δ) (-1 + δ), f (-1) ≤ f x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_min_at_neg_one_l749_74998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_x_cot_2x_is_half_l749_74904

-- Define the limit we want to prove
noncomputable def limit_x_cot_2x : ℝ := 1/2

-- State the theorem
theorem limit_x_cot_2x_is_half :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → 
    |x * (Real.cos (2*x) / Real.sin (2*x)) - limit_x_cot_2x| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_x_cot_2x_is_half_l749_74904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_correct_l749_74932

noncomputable def f (x : ℝ) : ℝ := 1 / (x + 3) + Real.sqrt (4 - x)

def f_domain : Set ℝ := { x | x ≤ 4 ∧ x ≠ -3 }

theorem f_domain_correct :
  ∀ x : ℝ, x ∈ f_domain ↔ (∃ y : ℝ, f x = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_correct_l749_74932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_contains_perfect_square_sum_l749_74956

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem subset_contains_perfect_square_sum 
  (n : ℕ) (hn : n ≥ 15) 
  (M : Finset ℕ) (hM : M = Finset.range n) 
  (A B : Finset ℕ) 
  (hA : A ⊂ M) (hB : B ⊂ M) 
  (hAB_disjoint : A ∩ B = ∅) 
  (hAB_cover : A ∪ B = M) : 
  (∃ (x y : ℕ), x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ is_perfect_square (x + y)) ∨
  (∃ (x y : ℕ), x ∈ B ∧ y ∈ B ∧ x ≠ y ∧ is_perfect_square (x + y)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_contains_perfect_square_sum_l749_74956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_projection_ratio_l749_74972

/-- Represents a tetrahedron with two opposite edges -/
structure Tetrahedron where
  a : ℝ  -- Length of one opposite edge
  b : ℝ  -- Length of the other opposite edge
  h : ℝ  -- Distance between parallel planes

/-- Represents the area of a projection of the tetrahedron onto a plane -/
noncomputable def ProjectionArea (t : Tetrahedron) (angle : ℝ) : ℝ :=
  (t.a * Real.cos angle + t.b * Real.cos angle) * t.h / 2

/-- Theorem stating that there exist two perpendicular planes such that 
    the ratio of the areas of the projections of the tetrahedron onto 
    these planes is at least √2 -/
theorem tetrahedron_projection_ratio (t : Tetrahedron) :
  ∃ (angle : ℝ), 
    (ProjectionArea t angle) / (ProjectionArea t (angle + Real.pi/2)) ≥ Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_projection_ratio_l749_74972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corner_difference_divisible_by_six_l749_74922

/-- A type representing a 9x9 grid of natural numbers -/
def Grid := Fin 9 → Fin 9 → ℕ

/-- Predicate to check if two cells are adjacent in the grid -/
def adjacent (i j k l : Fin 9) : Prop :=
  (i = k ∧ (j.val + 1 = l.val ∨ l.val + 1 = j.val)) ∨
  (j = l ∧ (i.val + 1 = k.val ∨ k.val + 1 = i.val))

/-- Predicate to check if a cell is a corner cell -/
def isCorner (i j : Fin 9) : Prop :=
  (i.val = 0 ∨ i.val = 8) ∧ (j.val = 0 ∨ j.val = 8)

/-- The main theorem -/
theorem corner_difference_divisible_by_six (g : Grid) : 
  (∀ n ∈ Finset.range 81, ∃! i j, g i j = n + 1) →
  (∀ i j k l, |Int.ofNat (g i j) - Int.ofNat (g k l)| = 3 → adjacent i j k l) →
  ∃ i j k l, isCorner i j ∧ isCorner k l ∧ 6 ∣ |Int.ofNat (g i j) - Int.ofNat (g k l)| :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_corner_difference_divisible_by_six_l749_74922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_count_permutations_theorem_l749_74919

/-- 
Given a positive integer n > 1, count_permutations n returns the number of permutations 
(a₁, a₂, ..., aₙ) of 1, 2, ..., n such that there is exactly one i ∈ {1, 2, ..., n-1} 
where aᵢ > aᵢ₊₁.
-/
def count_permutations (n : ℕ) : ℕ :=
  2^n - n - 1

/-- 
Assume a function that returns the actual number of permutations satisfying the given condition.
-/
axiom number_of_valid_permutations : ℕ → ℕ

/-- 
Theorem stating that for any positive integer n > 1, the number of permutations 
(a₁, a₂, ..., aₙ) of 1, 2, ..., n with exactly one i ∈ {1, 2, ..., n-1} where 
aᵢ > aᵢ₊₁ is equal to 2ⁿ - n - 1.
-/
theorem count_permutations_theorem (n : ℕ) (h : n > 1) :
  count_permutations n = number_of_valid_permutations n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_count_permutations_theorem_l749_74919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l749_74974

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + (x - 2) ^ 0

-- Define the domain of f
def domain_f : Set ℝ := {x | x > -1 ∧ x ≠ 2}

-- Theorem statement
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = domain_f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l749_74974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_cube_root_expression_l749_74938

/-- ω is a nonreal root of x^3 = 1 -/
noncomputable def ω : ℂ := sorry

axiom ω_cube : ω^3 = 1
axiom ω_nonreal : ω ≠ 0 ∧ ω ≠ 1 ∧ ω ≠ -1

/-- The main theorem -/
theorem complex_cube_root_expression :
  (1 - 2*ω + 3*ω^2)^4 + (1 + 3*ω - 2*ω^2)^4 = 9375*ω + 2722 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_cube_root_expression_l749_74938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l749_74930

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / (x^2 + 2)

-- State the theorem
theorem function_properties :
  ∃ (a b : ℝ),
    (∀ x, x ∈ Set.Ioo (-Real.sqrt 2) (Real.sqrt 2) → f a b x = (a * x) / (x^2 + 2)) ∧
    f a b 0 = 0 ∧
    f a b (1/3) = 3/19 ∧
    a = 1 ∧
    b = 0 ∧
    (∀ x₁ x₂, x₁ ∈ Set.Ioo (-Real.sqrt 2) (Real.sqrt 2) → 
               x₂ ∈ Set.Ioo (-Real.sqrt 2) (Real.sqrt 2) → 
               x₁ < x₂ → f a b x₁ < f a b x₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l749_74930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_axis_l749_74910

-- Define the necessary structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def power (p : ℝ × ℝ) (c : Circle) : ℝ := sorry

def Line (a b : ℝ × ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • a + t • b}

-- State the theorem
theorem radical_axis (Γ₁ Γ₂ : Circle) (A B : ℝ × ℝ) 
  (h_intersect : (A.1 - Γ₁.center.1)^2 + (A.2 - Γ₁.center.2)^2 = Γ₁.radius^2 ∧ 
                 (A.1 - Γ₂.center.1)^2 + (A.2 - Γ₂.center.2)^2 = Γ₂.radius^2 ∧
                 (B.1 - Γ₁.center.1)^2 + (B.2 - Γ₁.center.2)^2 = Γ₁.radius^2 ∧ 
                 (B.1 - Γ₂.center.1)^2 + (B.2 - Γ₂.center.2)^2 = Γ₂.radius^2) :
  {P : ℝ × ℝ | power P Γ₁ = power P Γ₂} = Line A B := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_axis_l749_74910
