import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_minus_three_x_equals_fourteen_l761_76194

theorem cube_minus_three_x_equals_fourteen :
  let x : ℝ := (7 + 4 * Real.sqrt 3) ^ (1/3 : ℝ) + (7 + 4 * Real.sqrt 3) ^ (-1/3 : ℝ)
  x^3 - 3*x = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_minus_three_x_equals_fourteen_l761_76194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_computer_price_proof_l761_76132

/-- The price of the basic computer and printer -/
noncomputable def total_price : ℝ := 2500

/-- The price difference between enhanced and basic computer -/
noncomputable def price_difference : ℝ := 500

/-- The fraction of the total price that the printer would be with the enhanced computer -/
noncomputable def printer_fraction : ℝ := 1/3

/-- The price of the basic computer -/
noncomputable def basic_computer_price : ℝ := 1500

theorem basic_computer_price_proof :
  ∃ (printer_price : ℝ),
    -- The sum of basic computer and printer prices equals the total price
    basic_computer_price + printer_price = total_price ∧
    -- The printer price is 1/3 of the total price with the enhanced computer
    printer_price = printer_fraction * (basic_computer_price + price_difference + printer_price) :=
by
  -- We'll use 1000 as the printer price
  let printer_price := 1000
  
  -- Prove the existence of such a printer price
  use printer_price
  
  constructor
  
  -- Prove the first condition
  · simp [basic_computer_price, total_price, printer_price]
    norm_num
  
  -- Prove the second condition
  · simp [printer_fraction, basic_computer_price, price_difference, printer_price]
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_computer_price_proof_l761_76132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizon_ratio_approx_sqrt_two_l761_76159

/-- The radius of the Earth in meters -/
def earth_radius : ℝ := 6000000

/-- Calculate the distance to the horizon for a given height above sea level -/
noncomputable def horizon_distance (height : ℝ) : ℝ := Real.sqrt (2 * earth_radius * height)

/-- The ratio of horizon distances for heights of 2 meters and 1 meter -/
noncomputable def horizon_ratio : ℝ := horizon_distance 2 / horizon_distance 1

/-- Theorem stating that the horizon ratio is approximately √2 -/
theorem horizon_ratio_approx_sqrt_two :
  |horizon_ratio - Real.sqrt 2| < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizon_ratio_approx_sqrt_two_l761_76159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l761_76117

def attendance_data : List (Nat × Nat) := [(0, 2), (1, 3), (2, 8), (3, 4), (4, 3)]

def total_students : Nat := 20

def mean (data : List (Nat × Nat)) : Rat :=
  (data.map (λ (days, count) => days * count) |>.sum : Nat) / total_students

def median (data : List (Nat × Nat)) : Nat :=
  let cumulative := data.scanl (λ acc (_, count) => acc + count) 0
  (data.find? (λ (_, count) => 
    let cum := cumulative.head!
    cum ≥ total_students / 2 && cum < (total_students + 1) / 2
  )).map Prod.fst |>.getD 0

theorem mean_median_difference :
  mean attendance_data - (median attendance_data : Rat) = 3 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l761_76117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_inverse_f_inv_correct_composition_result_l761_76104

def f : Fin 5 → Fin 6 := fun x =>
  match x with
  | ⟨0, _⟩ => ⟨3, by norm_num⟩
  | ⟨1, _⟩ => ⟨5, by norm_num⟩
  | ⟨2, _⟩ => ⟨1, by norm_num⟩
  | ⟨3, _⟩ => ⟨4, by norm_num⟩
  | ⟨4, _⟩ => ⟨2, by norm_num⟩

theorem f_has_inverse : Function.Bijective f := by sorry

noncomputable def f_inv : Fin 6 → Fin 5 := Function.invFun f

theorem f_inv_correct : Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f := by sorry

theorem composition_result : f_inv (f_inv (f_inv ⟨5, by norm_num⟩)) = ⟨4, by norm_num⟩ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_inverse_f_inv_correct_composition_result_l761_76104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jay_bought_three_cups_l761_76164

/-- Represents the movie outing expenses for three friends -/
structure MovieOuting where
  ticket_price : ℚ
  popcorn_price : ℚ
  milktea_price : ℚ
  popcorn_boxes : ℕ
  contribution_per_person : ℚ

/-- Calculates the number of cups of milk tea bought -/
def milktea_cups (mo : MovieOuting) : ℕ :=
  let total_contribution := 3 * mo.contribution_per_person
  let ticket_cost := 3 * mo.ticket_price
  let popcorn_cost := mo.popcorn_boxes * mo.popcorn_price
  let milktea_cost := total_contribution - ticket_cost - popcorn_cost
  (milktea_cost / mo.milktea_price).floor.toNat

/-- The main theorem stating that Jay bought 3 cups of milk tea -/
theorem jay_bought_three_cups (mo : MovieOuting) 
  (h1 : mo.ticket_price = 7)
  (h2 : mo.popcorn_price = 3/2)
  (h3 : mo.milktea_price = 3)
  (h4 : mo.popcorn_boxes = 2)
  (h5 : mo.contribution_per_person = 11) :
  milktea_cups mo = 3 := by
  sorry

#eval milktea_cups { ticket_price := 7, popcorn_price := 3/2, milktea_price := 3, popcorn_boxes := 2, contribution_per_person := 11 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jay_bought_three_cups_l761_76164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_specific_l761_76139

/-- The slope of the angle bisector of two lines -/
noncomputable def angle_bisector_slope (m₁ m₂ : ℝ) : ℝ :=
  (m₁ + m₂ - Real.sqrt (1 + m₁^2 + m₂^2)) / (1 - m₁ * m₂)

/-- Theorem: The slope of the angle bisector of y = 2x and y = 4x is (√21 - 6) / 7 -/
theorem angle_bisector_slope_specific : 
  angle_bisector_slope 2 4 = (Real.sqrt 21 - 6) / 7 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval angle_bisector_slope 2 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_specific_l761_76139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_29_l761_76123

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (λ x => x ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_29 : sum_of_divisors 29 = 30 := by
  -- The proof goes here
  sorry

#eval sum_of_divisors 29  -- This will evaluate the function for n = 29

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_29_l761_76123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_ratio_l761_76154

/-- A square in the 2D plane -/
structure Square (a : ℝ) where
  vertices : List (ℝ × ℝ) := [(-a, -a), (a, -a), (-a, a), (a, a)]

/-- A line in the 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The perimeter of the quadrilateral formed by cutting the square with the line -/
def perimeter_of_quadrilateral (a : ℝ) (s : Square a) (l : Line) : ℝ :=
  sorry

/-- Theorem: The perimeter of a quadrilateral formed by cutting a square with a line, divided by the side length of the square -/
theorem quadrilateral_perimeter_ratio (a : ℝ) (s : Square a) (l : Line) 
    (h1 : a > 0)
    (h2 : l.slope = 1/2)
    (h3 : l.intercept = 0) : 
  ∃ p : ℝ, p = 4 + Real.sqrt 5 ∧ 
    p * a = perimeter_of_quadrilateral a s l := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_ratio_l761_76154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_textbook_sale_price_l761_76195

/-- Proves that the price of each textbook on sale is $10 given the conditions of Marta's textbook purchases --/
theorem textbook_sale_price (sale_books : ℕ) (online_books : ℕ) (bookstore_books : ℕ) 
  (online_total : ℚ) (total_spent : ℚ) (sale_price : ℚ) :
  sale_books = 5 →
  online_books = 2 →
  bookstore_books = 3 →
  online_total = 40 →
  total_spent = 210 →
  sale_books * sale_price + online_total + 3 * online_total = total_spent →
  sale_price = 10 := by
  intros h1 h2 h3 h4 h5 h6
  -- The proof steps would go here
  sorry

#check textbook_sale_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_textbook_sale_price_l761_76195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l761_76196

/-- A geometric sequence with first term a₁ and common ratio q -/
noncomputable def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ :=
  λ n => a₁ * q^(n - 1)

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric sequence with a₁ = -2 and S₃ = -7/2, 
    the common ratio q is either 1/2 or -3/2 -/
theorem geometric_sequence_ratio : 
  ∀ q : ℝ, 
  geometric_sequence (-2) q 1 = -2 ∧ 
  geometric_sum (-2) q 3 = -7/2 → 
  q = 1/2 ∨ q = -3/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l761_76196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_inequality_solutions_l761_76133

def τ (n : ℕ+) : ℕ := (Nat.divisors n.val).card

noncomputable def σ (n : ℕ+) : ℕ := (Nat.divisors n.val).sum id

theorem divisor_inequality_solutions :
  {n : ℕ+ | (n : ℝ) * Real.sqrt (τ n : ℝ) ≤ (σ n : ℝ)} = {1, 2, 4, 6, 12} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_inequality_solutions_l761_76133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_product_squared_l761_76147

/-- Given a circle with unit radius and five points dividing its circumference into equal parts,
    the product of the lengths of two specific chords, when squared, equals 5. -/
theorem chord_product_squared (A₁ A₂ A₃ A₄ A₅ : ℝ × ℝ) : 
  let circle := {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 = 1}
  (A₁ ∈ circle) ∧ (A₂ ∈ circle) ∧ (A₃ ∈ circle) ∧ (A₄ ∈ circle) ∧ (A₅ ∈ circle) →
  (∃ θ : ℝ, 0 < θ ∧ θ < 2 * Real.pi ∧
    A₂ = (Real.cos θ, Real.sin θ) ∧
    A₃ = (Real.cos (2*θ), Real.sin (2*θ)) ∧
    A₄ = (Real.cos (3*θ), Real.sin (3*θ)) ∧
    A₅ = (Real.cos (4*θ), Real.sin (4*θ))) →
  (dist A₁ A₂ * dist A₁ A₃)^2 = 5 := by
  sorry

/-- Helper function to calculate the Euclidean distance between two points -/
noncomputable def dist (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_product_squared_l761_76147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_A_statement_D_l761_76150

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (line_perpendicular : Line → Line → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)
variable (skew : Line → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Statement A
theorem statement_A (α β : Plane) (l m : Line) :
  perpendicular α β →
  contains β l →
  intersect α β m →
  line_perpendicular l m →
  line_perpendicular_plane l α :=
by sorry

-- Statement D
theorem statement_D (α β : Plane) (l m : Line) :
  skew l m →
  contains α l →
  line_parallel_plane l β →
  contains β m →
  line_parallel_plane m α →
  plane_parallel α β :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_A_statement_D_l761_76150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_who_plays_hide_and_seek_l761_76126

/-- Represents whether a person goes to play hide and seek or not -/
def Person := Bool

/-- The conditions for the hide and seek game -/
structure HideAndSeekConditions where
  andrew : Person
  boris : Person
  vasya : Person
  gena : Person
  denis : Person
  condition1 : andrew = true → boris = true ∧ vasya = false
  condition2 : boris = true → gena = true ∨ denis = true
  condition3 : vasya = false → boris = false ∧ denis = false
  condition4 : andrew = false → boris = true ∧ gena = false

/-- The theorem stating who plays hide and seek given the conditions -/
theorem who_plays_hide_and_seek (c : HideAndSeekConditions) :
  c.boris = true ∧ c.vasya = true ∧ c.denis = true ∧ c.andrew = false ∧ c.gena = false :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_who_plays_hide_and_seek_l761_76126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_two_l761_76143

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 3^(-x) - 1 else Real.sqrt x

-- State the theorem
theorem f_greater_than_two (m : ℝ) : f m > 2 ↔ m < -1 ∨ m > 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_two_l761_76143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_two_divisor_l761_76190

theorem largest_power_of_two_divisor : ∃ k : ℕ, k = 3 ∧ 
  2^k = Nat.gcd (17^4 - 9^4 + 8 * 17^2) (2^(k + 1)) ∧
  ∀ m : ℕ, m > k → ¬(2^m ∣ (17^4 - 9^4 + 8 * 17^2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_two_divisor_l761_76190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_area_difference_l761_76119

noncomputable def rectangle_width : ℝ := 8
noncomputable def rectangle_height : ℝ := 12

noncomputable def area_semicircle (diameter : ℝ) : ℝ := (Real.pi / 2) * (diameter / 2)^2

noncomputable def area_large_semicircles : ℝ := 2 * area_semicircle rectangle_height
noncomputable def area_small_semicircles : ℝ := 2 * area_semicircle rectangle_width

theorem semicircle_area_difference : 
  (area_large_semicircles - area_small_semicircles) / area_small_semicircles * 100 = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_area_difference_l761_76119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_adjacent_girls_l761_76146

/-- Represents a child in the circle -/
inductive Child
| Boy
| Girl
deriving BEq, Repr

/-- Represents the circle of children -/
def Circle := List Child

/-- Function to count the number of boys in the circle -/
def countBoys (circle : Circle) : Nat :=
  circle.filter (· == Child.Boy) |>.length

/-- Function to count the number of girls in the circle -/
def countGirls (circle : Circle) : Nat :=
  circle.filter (· == Child.Girl) |>.length

/-- Check if any two girls are adjacent in the circle -/
def hasAdjacentGirls (circle : Circle) : Bool :=
  circle.zip (circle.rotate 1) |>.any (fun (a, b) => a == Child.Girl && b == Child.Girl)

/-- The main theorem to prove -/
theorem no_adjacent_girls (circle : Circle) 
  (h1 : circle.length = 104)
  (h2 : countBoys circle = 101)
  (h3 : countGirls circle = 3)
  (h4 : ∀ (i : Nat), i < countBoys circle → ∃ (initial final : Nat), initial = 50 ∧ final = 49) :
  ¬hasAdjacentGirls circle := by
  sorry

#check no_adjacent_girls

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_adjacent_girls_l761_76146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l761_76168

theorem inequality_solution_set :
  {x : ℝ | 3 ≤ |5 - 2*x| ∧ |5 - 2*x| < 9} = Set.Ioc (-2) 1 ∪ Set.Ico 4 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l761_76168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_standing_together_l761_76144

-- Define the number of students
def n : ℕ := 3

-- Define the number of ways two specific students can stand together
def ways_together : ℕ := 2 * 2

-- Define the total number of possible arrangements
def total_arrangements : ℕ := Nat.factorial n

-- Theorem to prove
theorem probability_standing_together :
  (ways_together : ℚ) / total_arrangements = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_standing_together_l761_76144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_equals_half_compound_interest_l761_76134

/-- Calculate compound interest amount -/
noncomputable def compoundInterestAmount (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate / 100) ^ time

/-- Calculate compound interest -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  compoundInterestAmount principal rate time - principal

/-- Calculate simple interest -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * rate * (time : ℝ) / 100

theorem simple_interest_equals_half_compound_interest : 
  ∃ (P : ℝ), 
    simpleInterest P 8 3 = (1/2) * compoundInterest 4000 10 2 ∧ 
    P = 1750 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_equals_half_compound_interest_l761_76134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheat_cleaning_problem_l761_76121

theorem wheat_cleaning_problem (total_acres planned_per_day additional_per_day last_day_acres : ℝ) 
  (h1 : total_acres > 0)
  (h2 : planned_per_day > 0)
  (h3 : additional_per_day ≥ 0)
  (h4 : last_day_acres > 0)
  (h5 : total_acres > last_day_acres) :
  ∃ (days : ℝ), 
    days > 0 ∧ 
    (planned_per_day + additional_per_day) * (days - 1) + last_day_acres = total_acres :=
by
  -- We need to prove the existence of a positive number of days
  -- that satisfies the equation
  sorry  -- The actual proof would go here

#check wheat_cleaning_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheat_cleaning_problem_l761_76121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_ratios_l761_76183

/-- Given squares A, B, and C with side lengths a, b, and c respectively -/
structure SquareSideLengths (a b c : ℝ) : Prop where
  b_eq : b = 4 * a
  c_eq : c = 3.5 * a

/-- The area ratio of two squares -/
noncomputable def areaRatio (side1 side2 : ℝ) : ℝ :=
  (side1 * side1) / (side2 * side2)

/-- Theorem stating the area ratios of squares B and C to square A -/
theorem square_area_ratios {a b c : ℝ} (h : SquareSideLengths a b c) :
  areaRatio b a = 16 ∧ areaRatio c a = 12.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_ratios_l761_76183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_y_l761_76109

-- Define the sequence
noncomputable def y : ℕ → ℝ
  | 0 => 1  -- Add a case for 0 to cover all natural numbers
  | 1 => (4 : ℝ) ^ (1/4)
  | 2 => ((4 : ℝ) ^ (1/4)) ^ ((4 : ℝ) ^ (1/3))
  | n+3 => (y (n+2)) ^ ((4 : ℝ) ^ (1/3))

-- Define IsInt for real numbers
def IsInt (x : ℝ) : Prop := ∃ n : ℤ, x = n

-- Theorem statement
theorem smallest_integer_y : (∀ k < 4, ¬ IsInt (y k)) ∧ IsInt (y 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_y_l761_76109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l761_76158

/-- Line l in parametric form -/
def line_l (s : ℝ) : ℝ × ℝ := (1 + s, 1 - s)

/-- Curve C in parametric form -/
def curve_C (t : ℝ) : ℝ × ℝ := (t + 2, t^2)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem intersection_distance :
  ∃ (s₁ s₂ t₁ t₂ : ℝ),
    s₁ ≠ s₂ ∧
    line_l s₁ = curve_C t₁ ∧
    line_l s₂ = curve_C t₂ ∧
    distance (line_l s₁) (line_l s₂) = Real.sqrt 2 := by
  sorry

#check intersection_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l761_76158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l761_76155

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State the conditions
variable (h1 : f 1 = 3)
variable (h2 : ∀ x, HasDerivAt f (f' x) x)
variable (h3 : ∀ x, f' x < 2)

-- State the theorem
theorem inequality_solution_set :
  (∀ x, f x < 2 * x + 1 ↔ x > 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l761_76155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_equality_l761_76138

noncomputable section

-- Define the circle
variable (ω : Set (EuclideanSpace ℝ (Fin 2)))

-- Define the points
variable (P A A' B B' X : EuclideanSpace ℝ (Fin 2))

-- Define the lines
def d (P A : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {x | ∃ t : ℝ, x = P + t • (A - P)}

def d' (P B : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {x | ∃ t : ℝ, x = P + t • (B - P)}

-- Define the conditions
variable (h_P_outside : P ∉ ω)
variable (h_A_in_circle : A ∈ ω)
variable (h_A'_in_circle : A' ∈ ω)
variable (h_B_in_circle : B ∈ ω)
variable (h_B'_in_circle : B' ∈ ω)
variable (h_A_in_d : A ∈ d P A)
variable (h_A'_in_d : A' ∈ d P A)
variable (h_B_in_d' : B ∈ d' P B)
variable (h_B'_in_d' : B' ∈ d' P B)
variable (h_X_intersection : X ∈ {x | ∃ t s : ℝ, x = A + t • (B' - A) ∧ x = A' + s • (B - A')})

-- State the theorem
theorem intersection_product_equality : 
  ‖X - A‖ * ‖X - B'‖ = ‖X - B‖ * ‖X - A'‖ := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_equality_l761_76138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l761_76120

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

noncomputable def m (C : ℝ) : Vector2D := ⟨2 * Real.cos (C / 2), -Real.sin C⟩
noncomputable def n (C : ℝ) : Vector2D := ⟨Real.cos (C / 2), 2 * Real.sin C⟩

/-- Dot product of two 2D vectors -/
def dotProduct (v w : Vector2D) : ℝ := v.x * w.x + v.y * w.y

/-- Two vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : Vector2D) : Prop := dotProduct v w = 0

theorem triangle_properties (t : Triangle) 
  (h1 : perpendicular (m t.C) (n t.C))
  (h2 : t.a^2 = 2*t.b^2 + t.c^2) : 
  t.C = π/3 ∧ Real.tan t.A = -3 * Real.sqrt 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l761_76120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_P_in_U_l761_76116

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set P
def P : Set ℝ := {y | ∃ x : ℝ, y = 1 / x ∧ 0 < x ∧ x < 1}

-- State the theorem
theorem complement_of_P_in_U : 
  Set.compl P = Set.Iic 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_P_in_U_l761_76116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_and_minimal_coefficients_l761_76180

/-- Parabola P₁ -/
def P₁ (x y : ℝ) : Prop := y = x^2 + 9/4

/-- Parabola P₂ -/
def P₂ (x y : ℝ) : Prop := x = y^2 + 25/16

/-- Common tangent line -/
def TangentLine (x y : ℝ) : Prop := 4*x + 2*y = 1

theorem common_tangent_and_minimal_coefficients :
  (∀ x y : ℝ, TangentLine x y → (P₁ x y ∨ P₂ x y → ∃ ε > 0, ∀ δ ∈ Set.Ioo (-ε) ε, ¬(P₁ (x+δ) (y + 4/2*δ) ∧ P₂ (x+δ) (y + 4/2*δ)))) ∧
  (∀ a b c : ℤ, (∀ x y : ℝ, a*x + b*y = c ↔ TangentLine x y) → Int.gcd a b = 1 → Int.gcd a c = 1 → a ≥ 4 ∧ b ≥ 2 ∧ c ≥ 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_and_minimal_coefficients_l761_76180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_proportion_m_eq_two_l761_76131

/-- A function f(x) is a direct proportion function if there exists a non-zero constant k such that f(x) = k * x for all x. -/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- The function y = (m+2)x^(m-1) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m + 2) * (x ^ (m - 1))

theorem direct_proportion_m_eq_two :
  ∃! m : ℝ, is_direct_proportion (f m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_proportion_m_eq_two_l761_76131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_speed_is_35_l761_76114

/-- Calculates the total distance travelled by a car with increasing speed -/
noncomputable def totalDistance (initialSpeed : ℝ) (hourlyIncrease : ℝ) (hours : ℕ) : ℝ :=
  (hours : ℝ) / 2 * (2 * initialSpeed + (hours - 1 : ℝ) * hourlyIncrease)

/-- Proves that given the conditions, the initial speed (distance travelled in the first hour) is 35 km/h -/
theorem initial_speed_is_35 :
  ∃ (initialSpeed : ℝ),
    totalDistance initialSpeed 2 12 = 552 ∧ initialSpeed = 35 := by
  use 35
  constructor
  · -- Prove that totalDistance 35 2 12 = 552
    simp [totalDistance]
    norm_num
  · -- Prove that initialSpeed = 35
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_speed_is_35_l761_76114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_and_minimum_value_l761_76162

-- Define the inequality and its solution set
noncomputable def inequality (a : ℝ) (x : ℝ) : Prop := a * x^2 - 3 * x + 2 < 0
noncomputable def solution_set (a b : ℝ) : Set ℝ := {x | 1 < x ∧ x < b ∧ inequality a x}

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := (2 * a + b) * x - 9 / ((a - b) * x)

-- State the theorem
theorem inequality_solution_and_minimum_value :
  ∃ (a b : ℝ),
    (∀ x, x ∈ solution_set a b ↔ 1 < x ∧ x < b) ∧
    a = 1 ∧ b = 2 ∧
    (∀ x ∈ solution_set a b, f a b x ≥ 12) ∧
    (∃ x ∈ solution_set a b, f a b x = 12) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_and_minimum_value_l761_76162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_matrix_values_l761_76186

noncomputable def rotation_matrix (θ : Real) : Matrix (Fin 2) (Fin 2) Real :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ, Real.cos θ]]

theorem rotation_matrix_values :
  ∀ (a b : Real),
  (rotation_matrix (Real.arccos (4/5)) = ![![a, b], ![3/5, 4/5]])
  → (a = 4/5 ∧ b = -3/5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_matrix_values_l761_76186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_period_pi_f_odd_l761_76142

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi)

-- State the theorem
theorem f_properties :
  (∀ x, f (x + Real.pi) = f x) ∧  -- Periodic with period π
  (∀ x, f (-x) = -f x)            -- Odd function
  := by
    sorry

-- Additional properties
theorem f_period_pi : ∀ x, f (x + Real.pi) = f x := by
  sorry

theorem f_odd : ∀ x, f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_period_pi_f_odd_l761_76142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l761_76125

/-- Represents a right triangle divided into a square and two smaller right triangles -/
structure DividedRightTriangle where
  /-- Side length of the square -/
  a : ℝ
  /-- Ratio of the area of one smaller right triangle to the area of the square -/
  k : ℝ
  /-- Assumption that a and k are positive -/
  a_pos : 0 < a
  k_pos : 0 < k

/-- The ratio of the area of the other smaller right triangle to the area of the square -/
noncomputable def areaRatio (t : DividedRightTriangle) : ℝ := 1 / (4 * t.k)

theorem area_ratio_theorem (t : DividedRightTriangle) : 
  areaRatio t = 1 / (4 * t.k) := by
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l761_76125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_triangle_area_l761_76141

/-- Given three squares with areas 256, 36, and 225, where two of these squares form 
    the legs of a right triangle, prove that the area of this triangle is 45. -/
theorem interior_triangle_area (a b c : ℝ) (ha : a = 256) (hb : b = 36) (hc : c = 225) :
  (1 / 2 : ℝ) * Real.sqrt b * Real.sqrt c = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_triangle_area_l761_76141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_count_l761_76145

theorem binomial_coefficient_count : 
  let S : Finset ℕ := Finset.filter (λ x => x ≤ 2014 ∧ Nat.choose 2014 x ≥ Nat.choose 2014 999) (Finset.range 2015)
  Finset.card S = 17 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_count_l761_76145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l761_76136

noncomputable def f₁ (x : ℝ) : ℝ := 3 * x + 3
noncomputable def f₂ (x : ℝ) : ℝ := (1/3) * x + 2
noncomputable def f₃ (x : ℝ) : ℝ := -(1/2) * x + 8

noncomputable def g (x : ℝ) : ℝ := min (f₁ x) (min (f₂ x) (f₃ x))

theorem max_value_of_g :
  ∃ (M : ℝ), M = 22/5 ∧ ∀ (x : ℝ), g x ≤ M ∧ ∃ (x₀ : ℝ), g x₀ = M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l761_76136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangles_l761_76193

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A set of 9 points in 3D space -/
def NinePoints : Set Point3D := sorry

/-- Any four points in the set are not coplanar -/
axiom not_coplanar (p1 p2 p3 p4 : Point3D) :
  p1 ∈ NinePoints → p2 ∈ NinePoints → p3 ∈ NinePoints → p4 ∈ NinePoints →
  p1 ≠ p2 → p1 ≠ p3 → p1 ≠ p4 → p2 ≠ p3 → p2 ≠ p4 → p3 ≠ p4 →
  ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0 ∧
    a * p1.x + b * p1.y + c * p1.z + d ≠ 0 ∨
    a * p2.x + b * p2.y + c * p2.z + d ≠ 0 ∨
    a * p3.x + b * p3.y + c * p3.z + d ≠ 0 ∨
    a * p4.x + b * p4.y + c * p4.z + d ≠ 0

/-- A graph formed by connecting some of the 9 points -/
def Graph : Set (Point3D × Point3D) := sorry

/-- No tetrahedron exists in the graph -/
axiom no_tetrahedron :
  ∀ (p1 p2 p3 p4 : Point3D),
    p1 ∈ NinePoints → p2 ∈ NinePoints → p3 ∈ NinePoints → p4 ∈ NinePoints →
    p1 ≠ p2 → p1 ≠ p3 → p1 ≠ p4 → p2 ≠ p3 → p2 ≠ p4 → p3 ≠ p4 →
    ¬((p1, p2) ∈ Graph ∧ (p1, p3) ∈ Graph ∧ (p1, p4) ∈ Graph ∧
      (p2, p3) ∈ Graph ∧ (p2, p4) ∈ Graph ∧ (p3, p4) ∈ Graph)

/-- Count of triangles in the graph -/
def triangle_count : ℕ := sorry

/-- The maximum number of triangles is 27 -/
theorem max_triangles : triangle_count ≤ 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangles_l761_76193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l761_76187

-- Define the function f(x)
noncomputable def f (x : Real) : Real := 4 * Real.sin x * Real.cos (x - Real.pi / 3) - Real.sqrt 3

-- State the theorem
theorem f_properties :
  -- Smallest positive period is π
  (∃ (T : Real), T > 0 ∧ T = Real.pi ∧ ∀ (x : Real), f x = f (x + T)) ∧
  -- Zeros of the function
  (∀ (x : Real), f x = 0 ↔ ∃ (k : Int), x = Real.pi / 6 + k * Real.pi / 2) ∧
  -- Maximum value in the interval [π/24, 3π/4]
  (∀ (x : Real), Real.pi / 24 ≤ x ∧ x ≤ 3 * Real.pi / 4 → f x ≤ 2) ∧
  (∃ (x : Real), Real.pi / 24 ≤ x ∧ x ≤ 3 * Real.pi / 4 ∧ f x = 2) ∧
  -- Minimum value in the interval [π/24, 3π/4]
  (∀ (x : Real), Real.pi / 24 ≤ x ∧ x ≤ 3 * Real.pi / 4 → f x ≥ -Real.sqrt 2) ∧
  (∃ (x : Real), Real.pi / 24 ≤ x ∧ x ≤ 3 * Real.pi / 4 ∧ f x = -Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l761_76187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l761_76148

def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 + 1

def g (m : ℝ) (x : ℝ) : ℝ := x^4 - 5*x^3 + (2-m)*x^2 + 1

theorem problem_solution :
  (∀ a : ℝ, (deriv (f a)) (2/3) = 0 → a = 1) ∧
  (∀ a : ℝ, (∃ x y : ℝ, x ∈ Set.Ioo (-2) 3 ∧ y ∈ Set.Ioo (-2) 3 ∧ x ≠ y ∧
    (deriv (f a)) x = 0 ∧ (deriv (f a)) y = 0) →
    a ∈ Set.Ioo (-3) 0 ∪ Set.Ioo 0 (9/2)) ∧
  (∀ m : ℝ, (∃! s : Finset ℝ, s.card = 3 ∧ ∀ x ∈ s, f 1 x = g m x) ↔
    m ∈ Set.Ioo (-3) 1 ∪ Set.Ioi 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l761_76148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l761_76199

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  2 * Real.sqrt a + 3 * (b ^ (1/3)) ≥ 5 * ((a * b) ^ (1/5)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l761_76199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_angles_are_75_and_45_l761_76113

/-- A quadrilateral with three equal sides and two known angles -/
structure SpecialQuadrilateral where
  -- Three equal sides
  a : ℝ
  -- Two known angles (in degrees)
  angle1 : ℝ
  angle2 : ℝ
  -- Side equality condition
  eq_sides : a > 0
  -- Angle conditions
  angle1_eq_90 : angle1 = 90
  angle2_eq_150 : angle2 = 150

/-- The theorem stating that the other two angles are 75° and 45° -/
theorem other_angles_are_75_and_45 (q : SpecialQuadrilateral) :
  ∃ (angle3 angle4 : ℝ), angle3 = 75 ∧ angle4 = 45 := by
  sorry

#check other_angles_are_75_and_45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_angles_are_75_and_45_l761_76113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_not_nonnegative_reals_l761_76108

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (a * x^2 + 4 * x + 1)

theorem range_not_nonnegative_reals (a : ℝ) (h : a = 6) :
  ¬ (∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, f a x = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_not_nonnegative_reals_l761_76108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_is_correct_l761_76176

/-- The line about which the point is symmetric to the origin -/
def symmetric_line (x y : ℝ) : Prop := x - 2*y + 2 = 0

/-- The point that is symmetric to the origin -/
noncomputable def symmetric_point : ℝ × ℝ := (-4/5, 8/5)

/-- The origin -/
def origin : ℝ × ℝ := (0, 0)

/-- Proposition: The symmetric_point is indeed symmetric to the origin about the symmetric_line -/
theorem symmetric_point_is_correct : 
  let (x₀, y₀) := symmetric_point
  let (x₁, y₁) := origin
  -- The midpoint of the line segment connecting the two points lies on the symmetric line
  symmetric_line ((x₀ + x₁)/2) ((y₀ + y₁)/2) ∧ 
  -- The line connecting the two points is perpendicular to the symmetric line
  (y₀ - y₁) = 2*(x₀ - x₁) := by
  sorry

#check symmetric_point_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_is_correct_l761_76176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_polynomials_h_3_l761_76174

/-- Represents a polynomial with integer coefficients -/
structure IntPolynomial where
  coeffs : List Int
  leading_coeff_pos : coeffs.head? = some (coeffs.head?.getD 0) → coeffs.head?.getD 0 > 0

/-- Calculates the h-value for a given polynomial -/
def h_value (p : IntPolynomial) : Nat :=
  p.coeffs.length - 1 + p.coeffs.foldl (fun acc x => acc + Int.natAbs x) 0

/-- The set of all polynomials satisfying h = 3 -/
def polynomials_with_h_3 : Set IntPolynomial :=
  {p | h_value p = 3}

/-- The main theorem stating that there are exactly 5 polynomials with h = 3 -/
theorem count_polynomials_h_3 :
    ∃ (s : Finset IntPolynomial), s.card = 5 ∧ ∀ p, p ∈ s ↔ p ∈ polynomials_with_h_3 := by
  sorry

#check count_polynomials_h_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_polynomials_h_3_l761_76174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_four_zeros_l761_76112

noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then |2^x - 1|
  else 3 / (x - 1)

noncomputable def g (x : ℝ) : ℝ := f (f x) - 2

theorem g_has_four_zeros :
  ∃ (a b c d : ℝ), a < b ∧ b < c ∧ c < d ∧
    (∀ x : ℝ, g x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) :=
by
  sorry

#check g_has_four_zeros

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_four_zeros_l761_76112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_woman_upstream_distance_l761_76130

/-- Represents the distance traveled upstream by a swimmer -/
noncomputable def distance_upstream (still_speed : ℝ) (downstream_distance : ℝ) (time : ℝ) : ℝ :=
  let stream_speed := downstream_distance / time - still_speed
  (still_speed - stream_speed) * time

/-- The theorem stating the upstream distance swam by the woman -/
theorem woman_upstream_distance :
  distance_upstream 10 45 3 = 15 := by
  -- Unfold the definition of distance_upstream
  unfold distance_upstream
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_woman_upstream_distance_l761_76130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisectors_concurrent_l761_76160

-- Define the triangle and points
variable (A B C D E F : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (h1 : dist A D = dist D B)
variable (h2 : dist B E = dist E C)
variable (h3 : dist C F = dist F A)

-- Define the angle bisectors
noncomputable def angle_bisector_ADB : Set (EuclideanSpace ℝ (Fin 2)) := sorry
noncomputable def angle_bisector_BEC : Set (EuclideanSpace ℝ (Fin 2)) := sorry
noncomputable def angle_bisector_CFA : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- State the theorem
theorem angle_bisectors_concurrent :
  ∃ P, P ∈ angle_bisector_ADB ∧ P ∈ angle_bisector_BEC ∧ P ∈ angle_bisector_CFA := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisectors_concurrent_l761_76160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sine_inequality_l761_76103

theorem negation_of_sine_inequality :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sine_inequality_l761_76103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_equalize_seven_boxes_l761_76173

/-- Represents the state of the boxes with their coin counts. -/
def BoxState := List Nat

/-- Checks if all elements in a list are equal. -/
def allEqual (l : List Nat) : Prop :=
  l.all (· = l.head!)

/-- Represents a valid move between adjacent boxes. -/
def isValidMove (fromBox toBox : Nat) (n : Nat) : Prop :=
  (fromBox + 1) % n = toBox ∨ (toBox + 1) % n = fromBox

/-- The minimum number of moves required to equalize the boxes. -/
noncomputable def minMovesToEqualize (initial : BoxState) : Nat :=
  sorry -- Implementation details omitted

/-- The main theorem stating the minimum number of moves required. -/
theorem min_moves_to_equalize_seven_boxes :
  let initial : BoxState := [2, 3, 5, 10, 15, 17, 20]
  minMovesToEqualize initial = 22 := by
  sorry

#check min_moves_to_equalize_seven_boxes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_equalize_seven_boxes_l761_76173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_l761_76152

theorem percentage_difference : 
  (12 / 100 * 24.2) - (10 / 100 * 14.2) = 1.484 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_l761_76152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_line_relations_l761_76135

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (planeparallel : Plane → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (skew : Line → Line → Prop)
variable (planeParallelLine : Plane → Line → Prop)  -- New relation

-- Define the lines and planes
variable (l m n : Line)
variable (α β γ : Plane)

-- State the theorem
theorem plane_line_relations :
  (∀ (l m : Line) (α β : Plane),
    skew l m ∧ subset l α ∧ subset m β → ¬(planeparallel α β)) ∧
  (∀ (l m : Line) (α β : Plane),
    planeparallel α β ∧ subset l α ∧ subset m β → ¬(parallel l m ∧ ¬(skew l m))) ∧
  (∀ (l m n : Line) (α β γ : Plane),
    intersect α β l ∧ intersect β γ m ∧ intersect γ α n ∧ planeParallelLine γ l → parallel m n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_line_relations_l761_76135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_a_b_l761_76177

theorem smallest_sum_a_b : ∃ (a b : ℕ), 
  a > 0 ∧ b > 0 ∧
  (2^6 * 7^3 : ℕ) = a^b ∧ 
  (∀ (c d : ℕ), c > 0 → d > 0 → (2^6 * 7^3 : ℕ) = c^d → a + b ≤ c + d) ∧
  a + b = 2746 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_a_b_l761_76177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_profit_is_50_percent_l761_76175

/-- Represents the trader's buying and selling weights -/
structure TraderWeights where
  buyIndicated : ℝ
  buyCheated : ℝ
  sellActual : ℝ
  sellClaimed : ℝ

/-- Defines the trader's cheating behavior -/
def cheatWeights (w : TraderWeights) : Prop :=
  w.buyCheated = w.buyIndicated * 1.1 ∧
  w.sellClaimed = w.sellActual * 1.5 ∧
  w.buyIndicated = w.sellClaimed

/-- Calculates the profit percentage -/
noncomputable def profitPercentage (w : TraderWeights) : ℝ :=
  (w.sellClaimed - w.sellActual) / w.sellActual * 100

/-- Theorem stating that the trader's profit percentage is 50% -/
theorem trader_profit_is_50_percent (w : TraderWeights) 
  (h : cheatWeights w) : profitPercentage w = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_profit_is_50_percent_l761_76175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_calculate_expression_l761_76182

-- Problem 1
theorem simplify_expression (a b : ℝ) : (5*a - 3*b) + 5*(a - 2*b) = 10*a - 13*b := by
  sorry

-- Problem 2
theorem calculate_expression : -2^2 + (Real.pi - 3.14)^(0 : ℕ) + (1/2)^(-2 : ℤ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_calculate_expression_l761_76182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l761_76165

-- Define a, b, and c as noncomputable
noncomputable def a : ℝ := 2^(1/3)
noncomputable def b : ℝ := Real.log 3 / Real.log 4
noncomputable def c : ℝ := Real.log 5 / Real.log 8

-- State the theorem
theorem order_of_abc : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l761_76165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l761_76110

theorem fraction_simplification (b y : ℝ) (h : b^2 ≠ y^2) :
  (Real.sqrt (b^2 + y^2) - (y^2 - b^2) / Real.sqrt (b^2 + y^2)) / (b^2 - y^2) = 
  2 * b^2 / (b^2 - y^2)^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l761_76110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l761_76179

noncomputable def f (x : ℝ) : ℝ := (Real.sin x * Real.cos x) / (1 + Real.sin x - Real.cos x)

theorem f_range : 
  {y : ℝ | ∃ x, f x = y ∧ 1 + Real.sin x - Real.cos x ≠ 0} = 
  Set.Icc (-(Real.sqrt 2 + 1) / 2) (-1) ∪ Set.Ioc (-1) ((Real.sqrt 2 - 1) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l761_76179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_f_f_x_eq_4_l761_76181

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if -5 ≤ x ∧ x ≤ -1 then -1/2 * x^2 - 2*x + 1
  else if -1 < x ∧ x ≤ 2 then x + 1
  else if 2 < x ∧ x ≤ 5 then 1/3 * x^2 - 2*x + 4
  else 0  -- Default value for x outside [-5, 5]

-- Theorem statement
theorem no_solutions_for_f_f_x_eq_4 :
  ∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → f (f x) ≠ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_f_f_x_eq_4_l761_76181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_container_capacities_l761_76127

/-- The capacity of the first container -/
def C₁ : ℝ := sorry

/-- The capacity of the second container -/
def C₂ : ℝ := sorry

/-- The amount of water initially in the first container -/
def W₁ : ℝ := 49

/-- The amount of water initially in the second container -/
def W₂ : ℝ := 56

/-- Theorem stating the capacities of the containers given the conditions -/
theorem container_capacities :
  (C₁ - W₁ = W₂ - C₂/2) ∧
  (C₂ - W₂ = W₁ - C₁/3) →
  C₁ = 63 ∧ C₂ = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_container_capacities_l761_76127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_probability_l761_76140

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of an event -/
def probability (favorable_outcomes total_outcomes : ℕ) : ℚ :=
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem committee_probability :
  let total_members : ℕ := 32
  let boys : ℕ := 14
  let girls : ℕ := 18
  let committee_size : ℕ := 6
  let total_ways : ℕ := choose total_members committee_size
  let ways_with_0_boys : ℕ := choose girls committee_size
  let ways_with_1_boy : ℕ := boys * choose girls (committee_size - 1)
  let ways_with_less_than_2_boys : ℕ := ways_with_0_boys + ways_with_1_boy
  probability (total_ways - ways_with_less_than_2_boys) total_ways = 767676 / 906192 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_probability_l761_76140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l761_76166

noncomputable def f (x : ℝ) : ℝ := (1/4)^x - 3*(1/2)^x + 2

theorem f_range : 
  ∀ y ∈ Set.range f, -1/4 ≤ y ∧ y ≤ 6 ∧
  ∃ x₁ x₂ : ℝ, -2 ≤ x₁ ∧ x₁ ≤ 2 ∧ -2 ≤ x₂ ∧ x₂ ≤ 2 ∧ 
  f x₁ = -1/4 ∧ f x₂ = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l761_76166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_inverse_17_mod_1200_l761_76169

theorem modular_inverse_17_mod_1200 :
  ∃ (x : ℕ), x < 1200 ∧ (17 * x) % 1200 = 1 :=
by
  -- We claim that 353 is the modular inverse
  use 353
  constructor
  -- First, prove that 353 < 1200
  · norm_num
  -- Then, prove that (17 * 353) % 1200 = 1
  · norm_num
  -- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_inverse_17_mod_1200_l761_76169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sprint_team_size_l761_76171

/-- The number of people on a sprint team, given the total distance run by the team and the distance run by each person. -/
noncomputable def sprintTeamSize (totalDistance : ℝ) (individualDistance : ℝ) : ℝ :=
  totalDistance / individualDistance

/-- Theorem: The sprint team has 150 people if they run 750 miles in total when each person runs 5.0 miles. -/
theorem sprint_team_size :
  sprintTeamSize 750 5 = 150 := by
  -- Unfold the definition of sprintTeamSize
  unfold sprintTeamSize
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sprint_team_size_l761_76171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_fold_probability_l761_76149

/-- A square sheet of paper --/
structure Square where
  side : ℝ
  area : ℝ := side * side

/-- A point on the square sheet --/
structure Point where
  x : ℝ
  y : ℝ

/-- A beautiful fold on the square --/
def is_beautiful_fold (s : Square) (p : Point) : Prop :=
  ∃ (line : Set Point), 
    line.Nonempty ∧ 
    p ∈ line ∧
    (∃ (center : Point), center ∈ line ∧ 
      center.x = s.side / 2 ∧ center.y = s.side / 2)

/-- The probability of a beautiful fold --/
noncomputable def probability_beautiful_fold (s : Square) : ℝ :=
  (s.side * s.side / 2) / s.area

/-- Theorem: The probability of a beautiful fold is 1/2 --/
theorem beautiful_fold_probability (s : Square) : 
  probability_beautiful_fold s = 1 / 2 := by
  sorry

#check beautiful_fold_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_fold_probability_l761_76149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_auto_finance_credit_calculation_l761_76163

noncomputable def total_consumer_credit : ℝ := 475

noncomputable def auto_credit_percentage : ℝ := 0.36

noncomputable def auto_finance_fraction : ℝ := 1/3

theorem auto_finance_credit_calculation :
  auto_finance_fraction * (auto_credit_percentage * total_consumer_credit) = 57 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_auto_finance_credit_calculation_l761_76163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l761_76102

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x + Real.pi / 3) + 2 * (Real.cos (x + Real.pi / 6))^2

theorem f_properties :
  (∀ x, f x = f (-x)) ∧
  (∀ x, f (-Real.pi/2 - x) = f (-Real.pi/2 + x)) ∧
  (∀ x, f (-Real.pi/4 + x) + f (-Real.pi/4 - x) = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l761_76102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_specific_matrix_l761_76188

theorem det_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 0, -4; 3, -1, 5; -6, 2, 0]
  Matrix.det A = -20 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_specific_matrix_l761_76188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_condition_l761_76153

-- Define the property that "a > b" is necessary but not sufficient for "log a > log b"
def necessary_but_not_sufficient (a b : ℝ) : Prop :=
  (∀ x y : ℝ, (x > 0 ∧ y > 0 ∧ Real.log x > Real.log y) → x > y) ∧
  ¬(∀ x y : ℝ, x > y → (x > 0 ∧ y > 0 ∧ Real.log x > Real.log y))

-- Theorem statement
theorem log_inequality_condition :
  ∀ a b : ℝ, necessary_but_not_sufficient a b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_condition_l761_76153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_value_l761_76118

theorem sin_2theta_value (θ : ℝ) (h : Real.cos θ + Real.sin θ = 7/5) : 
  Real.sin (2 * θ) = 24/25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_value_l761_76118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_5_range_of_m_l761_76191

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| + |x - 1|

-- Theorem for the solution set of f(x) ≥ 5
theorem solution_set_f_geq_5 :
  {x : ℝ | f x ≥ 5} = Set.Iic (-3) ∪ Set.Ici 2 :=
sorry

-- Theorem for the range of m when f(x) ≥ m^2 - 2m for all x
theorem range_of_m (m : ℝ) :
  (∀ x, f x ≥ m^2 - 2*m) ↔ m ∈ Set.Icc (-1) 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_5_range_of_m_l761_76191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_both_classes_represented_l761_76170

-- Define the total number of students
def total_students : ℕ := 30

-- Define the number of students in German class
def german_students : ℕ := 22

-- Define the number of students in Chinese class
def chinese_students : ℕ := 19

-- Define the probability of selecting two students from different classes
def prob_different_classes : ℚ := 352 / 435

-- Theorem statement
theorem prob_both_classes_represented :
  (1 : ℚ) - (Nat.choose (german_students + chinese_students - total_students) 2 + 
             Nat.choose (german_students - (german_students + chinese_students - total_students)) 2 + 
             Nat.choose (chinese_students - (german_students + chinese_students - total_students)) 2) / 
            (Nat.choose total_students 2 : ℚ) = prob_different_classes := by
  sorry

#eval prob_different_classes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_both_classes_represented_l761_76170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_support_preferences_l761_76178

def total_audience : ℕ := 1500

def team_a_support : ℚ := 40 / 100
def team_b_support : ℚ := 30 / 100
def team_c_support : ℚ := 20 / 100
def team_d_support : ℚ := 10 / 100
def team_e_support : ℚ := 5 / 100

def team_ab_overlap : ℚ := 12 / 100
def team_bc_overlap : ℚ := 15 / 100
def team_cd_overlap : ℚ := 10 / 100
def team_de_overlap : ℚ := 5 / 100

def non_supporters : ℚ := 4 / 100

theorem unknown_support_preferences :
  let team_a := (team_a_support * total_audience : ℚ).floor
  let team_b := (team_b_support * total_audience : ℚ).floor
  let team_c := (team_c_support * total_audience : ℚ).floor
  let team_d := (team_d_support * total_audience : ℚ).floor
  let team_e := (team_e_support * total_audience : ℚ).floor
  let ab_overlap := (team_ab_overlap * team_a : ℚ).floor
  let bc_overlap := (team_bc_overlap * team_b : ℚ).floor
  let cd_overlap := (team_cd_overlap * team_c : ℚ).floor
  let de_overlap := (team_de_overlap * team_d : ℚ).floor
  let non_support := (non_supporters * total_audience : ℚ).floor
  let total_supporters := team_a + team_b + team_c + team_d + team_e
  let total_overlaps := ab_overlap + bc_overlap + cd_overlap + de_overlap
  let accounted_for := total_supporters - total_overlaps + non_support
  total_audience - accounted_for = 43 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_support_preferences_l761_76178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maze_path_probability_l761_76137

/-- Represents a maze with randomly colored segments -/
structure Maze :=
  (is_symmetric : Bool)
  (random_coloring : Bool)

/-- Represents a path in the maze -/
inductive MazePath
  | white_ab : MazePath  -- White path from A to B
  | black_cd : MazePath  -- Black path from C to D

/-- The probability of a given path existing in the maze -/
noncomputable def path_probability (m : Maze) (p : MazePath) : ℝ :=
  sorry

/-- Theorem stating the probability of a white path from A to B is 1/2 -/
theorem maze_path_probability (m : Maze) 
  (h_sym : m.is_symmetric = true) 
  (h_rand : m.random_coloring = true) 
  (h_exclusive : ∀ (p q : MazePath), p ≠ q → path_probability m p + path_probability m q = 1) 
  (h_equal_prob : path_probability m MazePath.white_ab = path_probability m MazePath.black_cd) :
  path_probability m MazePath.white_ab = 1/2 := by
  sorry

#check maze_path_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maze_path_probability_l761_76137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_in_class_l761_76129

theorem boys_in_class (total_students : ℕ) (girls_fraction : ℚ) (boys : ℕ) : 
  total_students = 160 →
  girls_fraction = 5 / 8 →
  boys = total_students - (girls_fraction * ↑total_students).floor →
  boys = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_in_class_l761_76129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_motion_is_simple_harmonic_stone_performs_simple_harmonic_motion_l761_76161

/-- Represents the Earth with uniform density -/
structure Earth where
  radius : ℝ
  density : ℝ

/-- Represents a stone in the Earth's tunnel -/
structure Stone where
  mass : ℝ
  position : ℝ → ℝ  -- Position as a function of time
  velocity : ℝ → ℝ  -- Velocity as a function of time

/-- Gravitational acceleration as a function of distance from Earth's center -/
noncomputable def gravitational_acceleration (earth : Earth) (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * earth.density * r

/-- The motion of a stone in the Earth's tunnel is simple harmonic -/
theorem stone_motion_is_simple_harmonic (earth : Earth) (stone : Stone) :
  ∃ (A ω : ℝ), ∀ t, stone.position t = A * Real.cos (ω * t) := by
  sorry

/-- The stone performs simple harmonic motion in the Earth's tunnel -/
theorem stone_performs_simple_harmonic_motion (earth : Earth) (stone : Stone) :
  ∃ (period : ℝ), period > 0 ∧ 
    ∀ t, stone.position (t + period) = stone.position t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_motion_is_simple_harmonic_stone_performs_simple_harmonic_motion_l761_76161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decrease_intervals_triangle_side_c_l761_76192

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (x / 2) * Real.cos (x / 2) - Real.cos (x / 2) ^ 2 + 1 / 2

-- Theorem for the intervals of monotonic decrease
theorem monotonic_decrease_intervals (k : ℤ) :
  ∀ x ∈ Set.Icc (2 * Real.pi / 3 + 2 * ↑k * Real.pi) (5 * Real.pi / 3 + 2 * ↑k * Real.pi),
    ∀ y ∈ Set.Icc (2 * Real.pi / 3 + 2 * ↑k * Real.pi) (5 * Real.pi / 3 + 2 * ↑k * Real.pi),
      x ≤ y → f x ≥ f y :=
by
  sorry

-- Theorem for the value of c in the triangle
theorem triangle_side_c (A B C : ℝ) (a b c : ℝ) :
  f A = 1 / 2 →
  a = Real.sqrt 3 →
  Real.sin B = 2 * Real.sin C →
  c = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decrease_intervals_triangle_side_c_l761_76192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_identity_l761_76172

theorem tan_identity (θ : Real) (h : Real.tan θ = 3/4) :
  (1 + Real.cos θ) / Real.sin θ - Real.sin θ / (1 - Real.cos θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_identity_l761_76172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excenter_not_on_circumcircle_l761_76106

/-- Triangle ABC with its circumscribed circle and an excircle -/
structure TriangleWithCircles where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  circumcenter : ℝ × ℝ
  excenter : ℝ × ℝ

/-- The center of the excircle does not lie on the circumscribed circle -/
theorem excenter_not_on_circumcircle (t : TriangleWithCircles) : 
  t.excenter ≠ t.circumcenter ∧ 
  (t.excenter.1 - t.circumcenter.1)^2 + (t.excenter.2 - t.circumcenter.2)^2 ≠ 
  (t.A.1 - t.circumcenter.1)^2 + (t.A.2 - t.circumcenter.2)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_excenter_not_on_circumcircle_l761_76106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_distance_to_line_l761_76115

/-- A circle passing through (2,1) and tangent to both coordinate axes -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : (center.1 - 2)^2 + (center.2 - 1)^2 = radius^2
  tangent_to_axes : center.1 = center.2 ∧ center.1 = radius

/-- The distance from a point to a line ax + by + c = 0 -/
noncomputable def distance_point_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  |a * p.1 + b * p.2 + c| / Real.sqrt (a^2 + b^2)

theorem tangent_circle_distance_to_line :
  ∀ (circle : TangentCircle),
  distance_point_to_line circle.center 2 (-1) (-3) = 2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_distance_to_line_l761_76115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extrema_g_l761_76167

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 3| + |2*x - 8| - |x - 5|

-- Define the interval [1, 10]
def I : Set ℝ := Set.Icc 1 10

-- State the theorem
theorem sum_of_extrema_g : 
  ∃ (min_val max_val : ℝ), 
    (∀ x ∈ I, g x ≥ min_val) ∧ 
    (∃ x ∈ I, g x = min_val) ∧
    (∀ x ∈ I, g x ≤ max_val) ∧ 
    (∃ x ∈ I, g x = max_val) ∧
    min_val + max_val = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extrema_g_l761_76167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_properties_l761_76151

-- Define the conic curve
noncomputable def conic_curve (θ : Real) : Real × Real := (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

-- Define point A
noncomputable def point_A : Real × Real := (0, Real.sqrt 3)

-- Define focal points
def focal_points : (Real × Real) × (Real × Real) := 
  ((-1, 0), (1, 0))

-- Define line L
noncomputable def line_L (t : Real) : Real × Real :=
  (-1 + (Real.sqrt 3 / 2) * t, (1 / 2) * t)

-- Define polar equation of line AF₂
def line_AF₂_polar (ρ φ : Real) : Prop :=
  Real.sqrt 3 * ρ * Real.cos φ + ρ * Real.sin φ - Real.sqrt 3 = 0

theorem conic_properties :
  let (F₁, F₂) := focal_points
  ∀ t φ ρ : Real,
    (line_L t = (F₁.1 + (Real.sqrt 3 / 2) * t, F₁.2 + (1 / 2) * t)) ∧
    (line_AF₂_polar ρ φ ↔ Real.sqrt 3 * ρ * Real.cos φ + ρ * Real.sin φ - Real.sqrt 3 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_properties_l761_76151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_properties_l761_76197

/-- A cone with surface area 3π and lateral surface that unfolds to a semicircle -/
structure Cone where
  /-- The surface area of the cone is 3π -/
  surface_area : ℝ
  surface_area_eq : surface_area = 3 * Real.pi
  /-- The lateral surface unfolds to a semicircle -/
  lateral_surface_is_semicircle : Prop

/-- The slant height of the cone -/
noncomputable def slant_height (c : Cone) : ℝ := 2

/-- The volume of the cone -/
noncomputable def volume (c : Cone) : ℝ := (Real.sqrt 3 / 3) * Real.pi

theorem cone_properties (c : Cone) : 
  slant_height c = 2 ∧ volume c = (Real.sqrt 3 / 3) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_properties_l761_76197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_calculation_l761_76111

theorem income_calculation (total_income : ℝ) : 
  (let children_share := 0.15 * 3 * total_income
   let wife_share := 0.30 * total_income
   let remaining_after_family := total_income - children_share - wife_share
   let orphan_donation := 0.10 * remaining_after_family
   let final_amount := remaining_after_family - orphan_donation
   final_amount = 40000) →
  abs (total_income - 177777.78) < 0.01 := by
sorry

#eval Float.abs (177777.78 - (40000 / (1 - 0.45 - 0.30 - 0.1 * 0.25))) < 0.01

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_calculation_l761_76111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_theorem_l761_76189

/-- The locus of point Q -/
def locus_of_Q (x y : ℝ) : Prop :=
  y^2 = -2*x ∧ x ≤ 1 - Real.sqrt 2

/-- Parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 2*x

/-- Unit circle equation -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Tangent line to parabola at point P(a, b) -/
def tangent_to_parabola (a b x y : ℝ) : Prop :=
  parabola a b → y*b = x + a

/-- Point Q is the intersection of tangents to circle at M and N -/
def Q_is_intersection_of_tangents (qx qy mx my nx ny : ℝ) : Prop :=
  unit_circle mx my ∧ unit_circle nx ny ∧
  ∃ (px py : ℝ), parabola px py ∧
    tangent_to_parabola px py mx my ∧
    tangent_to_parabola px py nx ny

/-- Main theorem: The locus of point Q satisfies the given equation -/
theorem locus_of_Q_theorem :
  ∀ (qx qy : ℝ),
    (∀ (mx my nx ny : ℝ), Q_is_intersection_of_tangents qx qy mx my nx ny) →
    locus_of_Q qx qy :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_theorem_l761_76189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_statement_true_l761_76128

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin (2 * x)

theorem exactly_one_statement_true : 
  ∃! n : Fin 4, 
    (n = 0 ∧ (∃ p > 0, ∀ x, f (x + p) = f x) ∧ (∀ p > 0, (∀ x, f (x + p) = f x) → p ≥ 2 * Real.pi)) ∨
    (n = 1 ∧ (∀ x y, -Real.pi/4 ≤ x ∧ x < y ∧ y ≤ Real.pi/4 → f x < f y)) ∨
    (n = 2 ∧ (∀ x, -Real.pi/6 ≤ x ∧ x ≤ Real.pi/3 → -Real.sqrt 3/4 ≤ f x ∧ f x ≤ Real.sqrt 3/4) ∧
             (∃ x y, -Real.pi/6 ≤ x ∧ x ≤ Real.pi/3 ∧ -Real.pi/6 ≤ y ∧ y ≤ Real.pi/3 ∧ 
                     f x = -Real.sqrt 3/4 ∧ f y = Real.sqrt 3/4)) ∨
    (n = 3 ∧ (∀ x, f x = (1/2) * Real.sin (2 * (x + Real.pi/8) + Real.pi/4))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_statement_true_l761_76128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_ratio_l761_76105

theorem square_area_ratio (R : ℝ) : 
  R > 0 →
  (4 : ℝ) = 2 * R →
  let inscribed_square_side := R / Real.sqrt 2
  let circumscribed_square_area := 4^2
  let inscribed_square_area := inscribed_square_side^2
  inscribed_square_area / circumscribed_square_area = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_ratio_l761_76105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_years_l761_76100

/-- Jim's current age -/
def j : ℕ := sorry

/-- Eliza's current age -/
def e : ℕ := sorry

/-- The number of years until the ratio of their ages is 3:2 -/
def x : ℕ := sorry

/-- Jim's age was twice Eliza's age 4 years ago -/
axiom age_relation_1 : j - 4 = 2 * (e - 4)

/-- Jim's age was three times Eliza's age 10 years ago -/
axiom age_relation_2 : j - 10 = 3 * (e - 10)

/-- The ratio of their ages will be 3:2 after x years -/
axiom future_age_ratio : (j + x) * 2 = (e + x) * 3

theorem age_ratio_years : x = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_years_l761_76100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_common_intersection_l761_76122

/-- Represents a triangular pyramid -/
structure TriangularPyramid where
  baseArea : ℝ
  height : ℝ
  height_pos : height > 0

/-- The area of the intersection of a horizontal plane with a pyramid at height x -/
noncomputable def intersectionArea (p : TriangularPyramid) (x : ℝ) : ℝ :=
  p.baseArea * (1 - x / p.height)^2

/-- States that for any three pyramids, there exists a plane intersecting them with equal area -/
axiom three_equal_intersections (pyramids : Fin 7 → TriangularPyramid) :
  ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ∃ x, 0 ≤ x ∧ x ≤ min (pyramids i).height (min (pyramids j).height (pyramids k).height) ∧
      intersectionArea (pyramids i) x = intersectionArea (pyramids j) x ∧
      intersectionArea (pyramids j) x = intersectionArea (pyramids k) x

/-- The main theorem to be proved -/
theorem exists_common_intersection (pyramids : Fin 7 → TriangularPyramid) :
  ∃ x, ∀ i j, intersectionArea (pyramids i) x = intersectionArea (pyramids j) x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_common_intersection_l761_76122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equals_five_l761_76156

theorem complex_fraction_equals_five : ∀ i : ℂ, i * i = -1 →
  (2 + i) * (3 - 4*i) / (2 - i) = 5 := by
  intro i hi
  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equals_five_l761_76156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_digits_of_2048_l761_76198

/-- The number of digits in the base-7 representation of a positive integer -/
noncomputable def numDigitsBase7 (n : ℕ+) : ℕ :=
  Nat.floor (Real.log (n : ℝ) / Real.log 7) + 1

/-- Theorem: The number of digits in the base-7 representation of 2048 is 4 -/
theorem base7_digits_of_2048 : numDigitsBase7 2048 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_digits_of_2048_l761_76198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_increasing_sum_l761_76185

def arithmeticSequence (lambda : ℝ) (n : ℕ) : ℝ := 2 * n + lambda

def partialSum (lambda : ℝ) (n : ℕ) : ℝ := n^2 + (lambda + 1) * n

theorem arithmetic_sequence_increasing_sum (lambda : ℝ) :
  (∀ n ≥ 5, partialSum lambda (n + 1) > partialSum lambda n) → lambda > -12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_increasing_sum_l761_76185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_quadruplet_l761_76107

def I : Set ℕ := {n | 1 ≤ n ∧ n ≤ 999}

def hasQuadruplet (S : Set ℕ) : Prop :=
  ∃ a b c d, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + 2*b + 3*c = d

theorem smallest_n_with_quadruplet :
  ∀ n : ℕ, n ≥ 835 →
    (∀ S : Finset ℕ, ↑S ⊆ I → S.card = n → hasQuadruplet ↑S) ∧
    (∃ T : Finset ℕ, ↑T ⊆ I ∧ T.card = 834 ∧ ¬hasQuadruplet ↑T) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_quadruplet_l761_76107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_maintainable_l761_76124

/-- A type representing a balance scale with weights -/
structure BalanceScale (n : ℕ) where
  weights : Fin n → ℕ
  sum_left : ℕ
  sum_right : ℕ
  is_balanced : sum_left = sum_right

/-- Proposition: For any n > 6, there exists a set of three weights that can be removed from a balanced scale while maintaining equilibrium -/
theorem balance_maintainable (n : ℕ) (h : n > 6) :
  ∀ (scale : BalanceScale n), ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (∃ (new_scale : BalanceScale (n - 3)),
      new_scale.sum_left = scale.sum_left - (scale.weights i + scale.weights j + scale.weights k) ∧
      new_scale.sum_right = scale.sum_right - (scale.weights i + scale.weights j + scale.weights k)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_maintainable_l761_76124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_ratio_theorem_l761_76101

open Real

-- Define the set of positive real numbers
def PositiveReals := {x : ℝ | x > 0}

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_ratio_theorem 
  (h1 : ∀ x > 0, HasDerivAt f ((3 / x) * f x) x)
  (h2 : f (2^2016) ≠ 0) :
  f (2^2017) / f (2^2016) = 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_ratio_theorem_l761_76101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_max_height_l761_76157

/-- The elevation function for a ball thrown vertically upward -/
noncomputable def elevation (t : ℝ) : ℝ := 30 + 180 * t - 20 * t^2

/-- The time at which the maximum height is reached -/
noncomputable def t_max : ℝ := 180 / (2 * 20)

/-- The maximum height reached by the ball -/
noncomputable def max_height : ℝ := elevation t_max

theorem ball_max_height :
  max_height = 435 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_max_height_l761_76157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2023_eq_half_l761_76184

/-- The reciprocal difference of a rational number -/
def reciprocal_diff (a : ℚ) : ℚ := 1 / (1 - a)

/-- The sequence of reciprocal differences -/
def a : ℕ → ℚ
  | 0 => 1/2  -- Add this case for 0
  | 1 => 1/2
  | n+2 => reciprocal_diff (a (n+1))

/-- The 2023rd term of the sequence is 1/2 -/
theorem a_2023_eq_half : a 2023 = 1/2 := by
  sorry

#eval a 2023  -- This line is optional, for testing purposes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2023_eq_half_l761_76184
