import Mathlib

namespace intersection_points_form_parallelogram_l945_94556

-- Define the circle type
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the intersection points
def M : ℝ × ℝ := sorry
def N : ℝ × ℝ := sorry
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := sorry
def D : ℝ × ℝ := sorry

-- Define the four circles
def circle1 : Circle := sorry
def circle2 : Circle := sorry
def circle3 : Circle := sorry
def circle4 : Circle := sorry

-- Define the properties of the circles' intersections
def three_circles_intersect (c1 c2 c3 : Circle) : Prop :=
  ∃ p : ℝ × ℝ, (p = M ∨ p = N) ∧ 
    (‖p - c1.center‖ = c1.radius) ∧
    (‖p - c2.center‖ = c2.radius) ∧
    (‖p - c3.center‖ = c3.radius)

def two_circles_intersect (c1 c2 : Circle) : Prop :=
  ∃ p : ℝ × ℝ, (p = A ∨ p = B ∨ p = C ∨ p = D) ∧
    (‖p - c1.center‖ = c1.radius) ∧
    (‖p - c2.center‖ = c2.radius)

-- Theorem statement
theorem intersection_points_form_parallelogram
  (h1 : circle1.radius = circle2.radius ∧ circle2.radius = circle3.radius ∧ circle3.radius = circle4.radius)
  (h2 : three_circles_intersect circle1 circle2 circle3 ∧
        three_circles_intersect circle1 circle2 circle4 ∧
        three_circles_intersect circle1 circle3 circle4 ∧
        three_circles_intersect circle2 circle3 circle4)
  (h3 : two_circles_intersect circle1 circle2 ∧
        two_circles_intersect circle1 circle3 ∧
        two_circles_intersect circle1 circle4 ∧
        two_circles_intersect circle2 circle3 ∧
        two_circles_intersect circle2 circle4 ∧
        two_circles_intersect circle3 circle4) :
  C - D = B - A :=
by sorry

end intersection_points_form_parallelogram_l945_94556


namespace c_monthly_income_l945_94523

/-- Proves that C's monthly income is 17000, given the conditions from the problem -/
theorem c_monthly_income (a_annual_income : ℕ) (a_b_ratio : ℚ) (b_c_percentage : ℚ) :
  a_annual_income = 571200 →
  a_b_ratio = 5 / 2 →
  b_c_percentage = 112 / 100 →
  (a_annual_income / 12 : ℚ) * (2 / 5) / b_c_percentage = 17000 :=
by sorry

end c_monthly_income_l945_94523


namespace complex_number_in_second_quadrant_l945_94515

theorem complex_number_in_second_quadrant :
  let i : ℂ := Complex.I
  let z : ℂ := 1 + 2 * i + 3 * i^2
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end complex_number_in_second_quadrant_l945_94515


namespace bookstore_shipment_calculation_bookstore_shipment_proof_l945_94591

/-- Calculates the number of books received in a shipment given initial inventory, sales data, and final inventory. -/
theorem bookstore_shipment_calculation 
  (initial_inventory : ℕ) 
  (saturday_in_store : ℕ) 
  (saturday_online : ℕ) 
  (sunday_in_store_multiplier : ℕ) 
  (sunday_online_increase : ℕ) 
  (final_inventory : ℕ) : ℕ :=
  let total_saturday_sales := saturday_in_store + saturday_online
  let sunday_in_store := sunday_in_store_multiplier * saturday_in_store
  let sunday_online := saturday_online + sunday_online_increase
  let total_sunday_sales := sunday_in_store + sunday_online
  let total_sales := total_saturday_sales + total_sunday_sales
  let inventory_after_sales := initial_inventory - total_sales
  final_inventory - inventory_after_sales

/-- Proves that the bookstore received 160 books in the shipment. -/
theorem bookstore_shipment_proof : 
  bookstore_shipment_calculation 743 37 128 2 34 502 = 160 := by
  sorry

end bookstore_shipment_calculation_bookstore_shipment_proof_l945_94591


namespace expression_simplification_l945_94581

theorem expression_simplification (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) :
  (p / q + q / r + r / p - 1) * (p + q + r) +
  (p / q + q / r - r / p + 1) * (p + q - r) +
  (p / q - q / r + r / p + 1) * (p - q + r) +
  (-p / q + q / r + r / p + 1) * (-p + q + r) =
  4 * (p^2 / q + q^2 / r + r^2 / p) :=
by sorry

end expression_simplification_l945_94581


namespace divisor_problem_l945_94592

theorem divisor_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 166 → quotient = 9 → remainder = 4 → 
  dividend = divisor * quotient + remainder →
  divisor = 18 := by sorry

end divisor_problem_l945_94592


namespace point_symmetric_to_origin_l945_94510

theorem point_symmetric_to_origin (a : ℝ) : 
  let P : ℝ × ℝ := (2 - a, 3 * a + 6)
  (|2 - a| = |3 * a + 6|) → 
  (∃ (x y : ℝ), (x = -3 ∧ y = -3) ∨ (x = -6 ∧ y = 6)) ∧ 
  ((-(2 - a), -(3 * a + 6)) = (x, y)) :=
by sorry

end point_symmetric_to_origin_l945_94510


namespace only_translation_preserves_pattern_l945_94571

/-- Represents a shape in the pattern -/
inductive Shape
| Triangle
| Circle

/-- Represents the infinite alternating pattern -/
def Pattern := ℕ → Shape

/-- The alternating pattern of triangles and circles -/
def alternatingPattern : Pattern :=
  fun n => if n % 2 = 0 then Shape.Triangle else Shape.Circle

/-- Represents a transformation on the pattern -/
structure Transformation :=
  (apply : Pattern → Pattern)

/-- Rotation around a point on line ℓ under a triangle apex -/
def rotationTransformation : Transformation :=
  { apply := fun _ => alternatingPattern }

/-- Translation parallel to line ℓ -/
def translationTransformation : Transformation :=
  { apply := fun p n => p (n + 2) }

/-- Reflection across a line perpendicular to line ℓ -/
def reflectionTransformation : Transformation :=
  { apply := fun p n => p (n + 1) }

/-- Checks if a transformation preserves the pattern -/
def preservesPattern (t : Transformation) : Prop :=
  ∀ n, t.apply alternatingPattern n = alternatingPattern n

theorem only_translation_preserves_pattern :
  preservesPattern translationTransformation ∧
  ¬preservesPattern rotationTransformation ∧
  ¬preservesPattern reflectionTransformation :=
sorry

end only_translation_preserves_pattern_l945_94571


namespace sum_of_factorials_perfect_square_l945_94514

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem sum_of_factorials_perfect_square :
  ∀ n : ℕ, is_perfect_square (sum_of_factorials n) ↔ n = 1 ∨ n = 3 :=
sorry

end sum_of_factorials_perfect_square_l945_94514


namespace rental_cost_equation_l945_94536

/-- The monthly cost of renting a car. -/
def R : ℝ := sorry

/-- The monthly cost of the new car. -/
def new_car_cost : ℝ := 30

/-- The number of months in a year. -/
def months_in_year : ℕ := 12

/-- The difference in total cost over a year. -/
def cost_difference : ℝ := 120

/-- Theorem stating the relationship between rental cost and new car cost. -/
theorem rental_cost_equation : 
  months_in_year * R - months_in_year * new_car_cost = cost_difference := by
  sorry

end rental_cost_equation_l945_94536


namespace highlighter_difference_l945_94511

/-- Proves that the difference between blue and pink highlighters is 5 --/
theorem highlighter_difference (yellow pink blue : ℕ) : 
  yellow = 7 →
  pink = yellow + 7 →
  yellow + pink + blue = 40 →
  blue - pink = 5 := by
sorry


end highlighter_difference_l945_94511


namespace lcm_18_24_l945_94509

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l945_94509


namespace complex_square_simplification_l945_94544

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (5 - 3 * i)^2 = 16 - 30 * i :=
by sorry

end complex_square_simplification_l945_94544


namespace binary_arithmetic_equality_l945_94547

/-- Converts a list of bits (0s and 1s) to a natural number -/
def binaryToNat (bits : List Nat) : Nat :=
  bits.foldl (fun acc bit => 2 * acc + bit) 0

/-- The theorem to be proved -/
theorem binary_arithmetic_equality : 
  let a := binaryToNat [1, 1, 0, 1, 1]
  let b := binaryToNat [1, 0, 1, 0]
  let c := binaryToNat [1, 0, 0, 0, 1]
  let d := binaryToNat [1, 0, 1, 1]
  let e := binaryToNat [1, 1, 1, 0]
  let result := binaryToNat [0, 0, 1, 0, 0, 1]
  a + b - c + d - e = result := by
  sorry

end binary_arithmetic_equality_l945_94547


namespace simplest_common_denominator_l945_94590

-- Define the fractions
def fraction1 (x y : ℚ) : ℚ := 1 / (2 * x^2 * y)
def fraction2 (x y : ℚ) : ℚ := 1 / (6 * x * y^3)

-- Define the common denominator
def common_denominator (x y : ℚ) : ℚ := 6 * x^2 * y^3

-- Theorem statement
theorem simplest_common_denominator (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (a b : ℚ), 
    fraction1 x y = a / common_denominator x y ∧
    fraction2 x y = b / common_denominator x y ∧
    (∀ (c : ℚ), c > 0 → 
      (∃ (d e : ℚ), fraction1 x y = d / c ∧ fraction2 x y = e / c) →
      c ≥ common_denominator x y) :=
sorry

end simplest_common_denominator_l945_94590


namespace inscribed_rectangle_area_l945_94552

/-- A rectangle inscribed in a semicircle -/
structure InscribedRectangle where
  /-- Length of side MO of the rectangle -/
  mo : ℝ
  /-- Length of MG (equal to KO) -/
  mg : ℝ
  /-- The rectangle is inscribed in a semicircle -/
  inscribed : mo > 0 ∧ mg > 0

/-- The area of the inscribed rectangle is 240 -/
theorem inscribed_rectangle_area
  (rect : InscribedRectangle)
  (h1 : rect.mo = 20)
  (h2 : rect.mg = 12) :
  rect.mo * (rect.mg * rect.mg / rect.mo) = 240 :=
sorry

end inscribed_rectangle_area_l945_94552


namespace intersection_line_proof_l945_94501

/-- Given two lines in the plane and a slope, prove that a certain line passes through their intersection point with the given slope. -/
theorem intersection_line_proof (x y : ℝ) : 
  (3 * x + 4 * y = 5) →  -- First line equation
  (3 * x - 4 * y = 13) →  -- Second line equation
  (∃ (x₀ y₀ : ℝ), (3 * x₀ + 4 * y₀ = 5) ∧ (3 * x₀ - 4 * y₀ = 13) ∧ (2 * x₀ - y₀ = 7)) ∧  -- Intersection point exists and satisfies all equations
  (∀ (x₁ y₁ : ℝ), (2 * x₁ - y₁ = 7) → (y₁ - y) / (x₁ - x) = 2 ∨ x₁ = x)  -- Slope of the line 2x - y - 7 = 0 is 2
  := by sorry

end intersection_line_proof_l945_94501


namespace expected_coffee_days_expected_tea_days_expected_more_coffee_days_l945_94572

/-- Represents the outcome of rolling a die -/
inductive DieOutcome
| Prime
| Composite
| RollAgain

/-- Represents a fair eight-sided die with the given rules -/
def fairDie : Fin 8 → DieOutcome
| 1 => DieOutcome.RollAgain
| 2 => DieOutcome.Prime
| 3 => DieOutcome.Prime
| 4 => DieOutcome.Composite
| 5 => DieOutcome.Prime
| 6 => DieOutcome.Composite
| 7 => DieOutcome.Prime
| 8 => DieOutcome.Composite

/-- The probability of getting a prime number -/
def primeProbability : ℚ := 4 / 7

/-- The probability of getting a composite number -/
def compositeProbability : ℚ := 3 / 7

/-- The number of days in a non-leap year -/
def daysInYear : ℕ := 365

theorem expected_coffee_days (p : ℚ) (d : ℕ) (h : p = primeProbability) : 
  ⌊p * d⌋ = 209 :=
sorry

theorem expected_tea_days (p : ℚ) (d : ℕ) (h : p = compositeProbability) : 
  ⌊p * d⌋ = 156 :=
sorry

theorem expected_more_coffee_days : 
  ⌊primeProbability * daysInYear⌋ - ⌊compositeProbability * daysInYear⌋ = 53 :=
sorry

end expected_coffee_days_expected_tea_days_expected_more_coffee_days_l945_94572


namespace lewis_money_at_end_of_harvest_l945_94577

/-- Calculates the money Lewis will have at the end of the harvest season -/
def money_at_end_of_harvest (weekly_earnings : ℕ) (weekly_rent : ℕ) (num_weeks : ℕ) : ℕ :=
  (weekly_earnings - weekly_rent) * num_weeks

/-- Proves that Lewis will have $325175 at the end of the harvest season -/
theorem lewis_money_at_end_of_harvest :
  money_at_end_of_harvest 491 216 1181 = 325175 := by
  sorry

end lewis_money_at_end_of_harvest_l945_94577


namespace tenth_prime_is_29_l945_94562

/-- Definition of natural numbers -/
def NaturalNumber (n : ℕ) : Prop := n ≥ 0

/-- Definition of prime numbers -/
def PrimeNumber (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 0 ∧ m < p → (p % m = 0 → m = 1)

/-- Function to get the nth prime number -/
def nthPrime (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The 10th prime number is 29 -/
theorem tenth_prime_is_29 : nthPrime 10 = 29 := by
  sorry

end tenth_prime_is_29_l945_94562


namespace optimal_k_value_l945_94528

theorem optimal_k_value : ∃! k : ℝ, 
  (∀ a b c d : ℝ, a ≥ -1 ∧ b ≥ -1 ∧ c ≥ -1 ∧ d ≥ -1 → 
    a^3 + b^3 + c^3 + d^3 + 1 ≥ k * (a + b + c + d)) ∧ 
  k = 3/4 := by
  sorry

end optimal_k_value_l945_94528


namespace function_property_l945_94598

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * ((f x)^2 - f x * f y + (f y)^2)

/-- The main theorem to be proved -/
theorem function_property (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x :=
by sorry

end function_property_l945_94598


namespace triangle_max_area_l945_94569

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the maximum area of the triangle is (3√3)/4 when b = √3 and (2a-c)cos B = √3 cos C -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  b = Real.sqrt 3 →
  (2 * a - c) * Real.cos B = Real.sqrt 3 * Real.cos C →
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧ 
    ∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ S) →
  (3 * Real.sqrt 3) / 4 = (1/2) * b * c * Real.sin A :=
by sorry

end triangle_max_area_l945_94569


namespace jakes_birdhouse_height_l945_94543

/-- Represents the dimensions of a birdhouse in inches -/
structure BirdhouseDimensions where
  width : ℕ
  height : ℕ
  depth : ℕ

/-- Calculates the volume of a birdhouse given its dimensions -/
def birdhouse_volume (d : BirdhouseDimensions) : ℕ :=
  d.width * d.height * d.depth

theorem jakes_birdhouse_height :
  let sara_birdhouse : BirdhouseDimensions := {
    width := 12,  -- 1 foot = 12 inches
    height := 24, -- 2 feet = 24 inches
    depth := 24   -- 2 feet = 24 inches
  }
  let jake_birdhouse : BirdhouseDimensions := {
    width := 16,
    height := 20, -- We'll prove this is correct
    depth := 18
  }
  birdhouse_volume sara_birdhouse - birdhouse_volume jake_birdhouse = 1152 :=
by sorry


end jakes_birdhouse_height_l945_94543


namespace square_area_not_possible_l945_94576

-- Define the points
def P : ℝ × ℝ := (1, 0)
def Q : ℝ × ℝ := (2, 0)
def R : ℝ × ℝ := (4, 0)
def S : ℝ × ℝ := (8, 0)

-- Define a predicate for four lines forming a square
def forms_square (l₁ l₂ l₃ l₄ : Set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ) (side : ℝ),
    side > 0 ∧
    (∀ p ∈ l₁ ∪ l₂ ∪ l₃ ∪ l₄, ∃ i j : ℤ, (p.1 - center.1)^2 + (p.2 - center.2)^2 = 2 * side^2 * (i^2 + j^2)) ∧
    (P ∈ l₁ ∪ l₂ ∪ l₃ ∪ l₄) ∧
    (Q ∈ l₁ ∪ l₂ ∪ l₃ ∪ l₄) ∧
    (R ∈ l₁ ∪ l₂ ∪ l₃ ∪ l₄) ∧
    (S ∈ l₁ ∪ l₂ ∪ l₃ ∪ l₄)

-- The theorem to prove
theorem square_area_not_possible :
  ∀ l₁ l₂ l₃ l₄ : Set (ℝ × ℝ),
  forms_square l₁ l₂ l₃ l₄ →
  ∀ side : ℝ, side^2 ≠ 26/5 :=
by sorry

end square_area_not_possible_l945_94576


namespace simplify_fraction_l945_94555

theorem simplify_fraction : (84 : ℚ) / 144 = 7 / 12 := by sorry

end simplify_fraction_l945_94555


namespace inequality_properties_l945_94553

theorem inequality_properties (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) :
  (a + b < a * b) ∧
  (abs a ≤ abs b) ∧
  (a ≥ b) ∧
  (b/a + a/b > 2) := by
sorry

end inequality_properties_l945_94553


namespace QED_product_l945_94550

theorem QED_product (Q E D : ℂ) : 
  Q = 5 + 2*I ∧ E = I ∧ D = 5 - 2*I → Q * E * D = 29 * I :=
by sorry

end QED_product_l945_94550


namespace inequality_proof_l945_94540

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + 8 / (x * y) + y^2 ≥ 8 := by
  sorry

end inequality_proof_l945_94540


namespace inverse_proportion_relationship_l945_94527

/-- Given points A(-1, y₁), B(2, y₂), and C(3, y₃) on the graph of y = -6/x,
    prove that y₁ > y₃ > y₂ -/
theorem inverse_proportion_relationship (y₁ y₂ y₃ : ℝ) 
  (h₁ : y₁ = 6 / (-1))
  (h₂ : y₂ = -6 / 2)
  (h₃ : y₃ = -6 / 3) :
  y₁ > y₃ ∧ y₃ > y₂ := by
  sorry

end inverse_proportion_relationship_l945_94527


namespace additional_nails_l945_94583

/-- Calculates the number of additional nails used in a house wall construction. -/
theorem additional_nails (total_nails : ℕ) (nails_per_plank : ℕ) (planks_needed : ℕ) :
  total_nails = 11 →
  nails_per_plank = 3 →
  planks_needed = 1 →
  total_nails - (nails_per_plank * planks_needed) = 8 := by
  sorry

#check additional_nails

end additional_nails_l945_94583


namespace certain_amount_calculation_l945_94597

theorem certain_amount_calculation (x A : ℝ) (h1 : x = 230) (h2 : 0.65 * x = 0.20 * A) : A = 747.5 := by
  sorry

end certain_amount_calculation_l945_94597


namespace z_in_third_quadrant_l945_94542

theorem z_in_third_quadrant (z : ℂ) (h : Complex.I * z = (4 + 3 * Complex.I) / (1 + 2 * Complex.I)) : 
  z.re < 0 ∧ z.im < 0 := by
  sorry

end z_in_third_quadrant_l945_94542


namespace player_positions_satisfy_distances_l945_94554

/-- Represents the positions of four soccer players on a number line -/
def PlayerPositions : Fin 4 → ℝ
  | 0 => 0
  | 1 => 1
  | 2 => 4
  | 3 => 6

/-- Calculates the distance between two players -/
def distance (i j : Fin 4) : ℝ :=
  |PlayerPositions i - PlayerPositions j|

/-- The set of required pairwise distances -/
def RequiredDistances : Set ℝ := {1, 2, 3, 4, 5, 6}

/-- Theorem stating that the player positions satisfy the required distances -/
theorem player_positions_satisfy_distances :
  ∀ i j : Fin 4, i ≠ j → distance i j ∈ RequiredDistances :=
sorry

end player_positions_satisfy_distances_l945_94554


namespace inequality_proof_l945_94512

theorem inequality_proof (a b c : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0)
  (sum_one : a + b + c = 1) :
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 := by
  sorry

end inequality_proof_l945_94512


namespace amy_video_files_amy_video_files_proof_l945_94565

theorem amy_video_files : ℕ → Prop :=
  fun initial_video_files =>
    let initial_music_files : ℕ := 4
    let deleted_files : ℕ := 23
    let remaining_files : ℕ := 2
    initial_music_files + initial_video_files - deleted_files = remaining_files →
    initial_video_files = 21

-- Proof
theorem amy_video_files_proof : amy_video_files 21 := by
  sorry

end amy_video_files_amy_video_files_proof_l945_94565


namespace daniel_initial_noodles_l945_94531

/-- The number of noodles Daniel had initially -/
def initial_noodles : ℕ := sorry

/-- The number of noodles Daniel gave to William -/
def noodles_to_william : ℕ := 15

/-- The number of noodles Daniel gave to Emily -/
def noodles_to_emily : ℕ := 20

/-- The number of noodles Daniel has left -/
def noodles_left : ℕ := 40

/-- Theorem stating that Daniel started with 75 noodles -/
theorem daniel_initial_noodles : initial_noodles = 75 := by sorry

end daniel_initial_noodles_l945_94531


namespace soccer_team_age_mode_l945_94582

def player_ages : List ℕ := [17, 17, 18, 18, 16, 18, 17, 15, 18, 18, 17, 16, 18, 17, 18, 14]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem soccer_team_age_mode :
  mode player_ages = 18 := by
  sorry

end soccer_team_age_mode_l945_94582


namespace remainder_problem_l945_94529

theorem remainder_problem (k : ℤ) : ∃ (x : ℤ), x = 8 * k + 1 ∧ 71 * x % 8 = 7 := by
  sorry

end remainder_problem_l945_94529


namespace cuboid_missing_edge_l945_94558

/-- Proves that for a cuboid with given dimensions and volume, the unknown edge length is 5 cm -/
theorem cuboid_missing_edge :
  let edge1 : ℝ := 2
  let edge3 : ℝ := 8
  let volume : ℝ := 80
  ∃ edge2 : ℝ, edge1 * edge2 * edge3 = volume ∧ edge2 = 5
  := by sorry

end cuboid_missing_edge_l945_94558


namespace class_election_votes_l945_94567

theorem class_election_votes (total_votes : ℕ) (fiona_votes : ℕ) : 
  fiona_votes = 48 → 
  (fiona_votes : ℚ) / total_votes = 2 / 5 → 
  total_votes = 120 := by
sorry

end class_election_votes_l945_94567


namespace kid_tickets_sold_l945_94502

/-- Prove that the number of kid tickets sold is 75 -/
theorem kid_tickets_sold (total_tickets : ℕ) (total_profit : ℕ) 
  (adult_price kid_price : ℕ) (h1 : total_tickets = 175) 
  (h2 : total_profit = 750) (h3 : adult_price = 6) (h4 : kid_price = 2) : 
  ∃ (adult_tickets kid_tickets : ℕ), 
    adult_tickets + kid_tickets = total_tickets ∧ 
    adult_price * adult_tickets + kid_price * kid_tickets = total_profit ∧
    kid_tickets = 75 :=
sorry

end kid_tickets_sold_l945_94502


namespace min_a_for_four_integer_solutions_l945_94530

theorem min_a_for_four_integer_solutions : 
  let has_four_solutions (a : ℤ) := 
    (∃ x₁ x₂ x₃ x₄ : ℤ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧ 
      (x₁ - a < 0) ∧ (2 * x₁ + 3 > 0) ∧
      (x₂ - a < 0) ∧ (2 * x₂ + 3 > 0) ∧
      (x₃ - a < 0) ∧ (2 * x₃ + 3 > 0) ∧
      (x₄ - a < 0) ∧ (2 * x₄ + 3 > 0))
  ∀ a : ℤ, has_four_solutions a → a ≥ 3 ∧ has_four_solutions 3 :=
by sorry

end min_a_for_four_integer_solutions_l945_94530


namespace polynomial_simplification_l945_94534

theorem polynomial_simplification (s : ℝ) : (2 * s^2 + 5 * s - 3) - (2 * s^2 + 9 * s - 4) = -4 * s + 1 := by
  sorry

end polynomial_simplification_l945_94534


namespace bowling_team_weight_problem_l945_94588

theorem bowling_team_weight_problem (original_players : ℕ) 
                                    (original_avg_weight : ℝ) 
                                    (new_players : ℕ) 
                                    (known_new_player_weight : ℝ) 
                                    (new_avg_weight : ℝ) :
  original_players = 7 →
  original_avg_weight = 76 →
  new_players = 2 →
  known_new_player_weight = 60 →
  new_avg_weight = 78 →
  ∃ (unknown_new_player_weight : ℝ),
    (original_players * original_avg_weight + known_new_player_weight + unknown_new_player_weight) / 
    (original_players + new_players) = new_avg_weight ∧
    unknown_new_player_weight = 110 :=
by sorry

end bowling_team_weight_problem_l945_94588


namespace star_3_5_l945_94538

-- Define the star operation
def star (a b : ℝ) : ℝ := (a + b)^2 + (a - b)^2

-- Theorem statement
theorem star_3_5 : star 3 5 = 68 := by sorry

end star_3_5_l945_94538


namespace fixed_point_on_line_l945_94519

/-- The line equation is satisfied by the point (2, 3) for all values of k -/
theorem fixed_point_on_line (k : ℝ) : (2*k - 1) * 2 - (k - 2) * 3 - (k + 4) = 0 := by
  sorry

end fixed_point_on_line_l945_94519


namespace jacoby_trip_cost_l945_94521

def trip_cost (hourly_rate job_hours cookie_price cookies_sold 
               lottery_ticket_cost lottery_winnings sister_gift sister_count
               additional_needed : ℕ) : ℕ :=
  let job_earnings := hourly_rate * job_hours
  let cookie_earnings := cookie_price * cookies_sold
  let sister_gifts := sister_gift * sister_count
  let total_earned := job_earnings + cookie_earnings + lottery_winnings + sister_gifts
  let total_after_ticket := total_earned - lottery_ticket_cost
  total_after_ticket + additional_needed

theorem jacoby_trip_cost : 
  trip_cost 20 10 4 24 10 500 500 2 3214 = 5000 := by
  sorry

end jacoby_trip_cost_l945_94521


namespace saras_baking_days_l945_94524

/-- Proves the number of weekdays Sara makes cakes given the problem conditions -/
theorem saras_baking_days (cakes_per_day : ℕ) (price_per_cake : ℕ) (total_collected : ℕ) 
  (h1 : cakes_per_day = 4)
  (h2 : price_per_cake = 8)
  (h3 : total_collected = 640) :
  total_collected / price_per_cake / cakes_per_day = 20 := by
  sorry

end saras_baking_days_l945_94524


namespace log_8_x_equals_3_75_l945_94579

theorem log_8_x_equals_3_75 (x : ℝ) :
  Real.log x / Real.log 8 = 3.75 → x = 1024 * Real.sqrt 2 := by
  sorry

end log_8_x_equals_3_75_l945_94579


namespace line_through_circle_center_l945_94533

theorem line_through_circle_center (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y = 0 ∧ 
                 3*x + y + a = 0 ∧ 
                 x = -1 ∧ y = 2) → 
  a = 1 := by sorry

end line_through_circle_center_l945_94533


namespace age_difference_l945_94551

theorem age_difference (a b c : ℕ) : 
  b = 20 →
  c = b / 2 →
  a + b + c = 52 →
  a = b + 2 := by
sorry

end age_difference_l945_94551


namespace defective_smartphones_l945_94573

theorem defective_smartphones (total : ℕ) (prob : ℝ) (defective : ℕ) : 
  total = 220 → 
  prob = 0.14470734744707348 →
  (defective : ℝ) / total * ((defective : ℝ) - 1) / (total - 1) = prob →
  defective = 84 :=
by sorry

end defective_smartphones_l945_94573


namespace solution_set_when_a_is_1_range_of_a_given_condition_l945_94566

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |3*x + a|

-- Part 1
theorem solution_set_when_a_is_1 :
  {x : ℝ | f 1 x ≥ 5} = {x : ℝ | x ≤ -1 ∨ x ≥ 1} := by sorry

-- Part 2
theorem range_of_a_given_condition (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ + 2*|x₀ - 2| < 3) → -9 < a ∧ a < -3 := by sorry

end solution_set_when_a_is_1_range_of_a_given_condition_l945_94566


namespace equal_roots_quadratic_l945_94508

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y + k = 0 → y = x) → 
  k = 4 := by sorry

end equal_roots_quadratic_l945_94508


namespace two_apples_per_slice_l945_94518

/-- The number of apples in each slice of pie -/
def apples_per_slice (total_apples : ℕ) (num_pies : ℕ) (slices_per_pie : ℕ) : ℚ :=
  (total_apples : ℚ) / (num_pies * slices_per_pie)

/-- Theorem stating that there are 2 apples per slice given the problem conditions -/
theorem two_apples_per_slice :
  apples_per_slice (4 * 12) 4 6 = 2 := by
  sorry

end two_apples_per_slice_l945_94518


namespace celestia_badges_l945_94568

theorem celestia_badges (total : ℕ) (hermione : ℕ) (luna : ℕ) (celestia : ℕ)
  (h_total : total = 83)
  (h_hermione : hermione = 14)
  (h_luna : luna = 17)
  (h_sum : total = hermione + luna + celestia) :
  celestia = 52 := by
sorry

end celestia_badges_l945_94568


namespace canoe_weight_proof_l945_94546

def canoe_capacity : ℕ := 6
def person_weight : ℕ := 140

def total_weight_with_dog : ℕ :=
  let people_with_dog := (2 * canoe_capacity) / 3
  let total_people_weight := people_with_dog * person_weight
  let dog_weight := person_weight / 4
  total_people_weight + dog_weight

theorem canoe_weight_proof :
  total_weight_with_dog = 595 := by
  sorry

end canoe_weight_proof_l945_94546


namespace f_is_even_l945_94520

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Theorem statement
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end f_is_even_l945_94520


namespace fermats_little_theorem_l945_94584

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (h : Prime p) :
  a^p ≡ a [MOD p] := by
  sorry

end fermats_little_theorem_l945_94584


namespace line_circle_intersection_range_l945_94506

theorem line_circle_intersection_range (a : ℝ) :
  (∃ x y : ℝ, x - y - a = 0 ∧ (x - 1)^2 + y^2 = 2) →
  -1 ≤ a ∧ a ≤ 3 :=
by sorry

end line_circle_intersection_range_l945_94506


namespace quadratic_minimum_l945_94505

theorem quadratic_minimum (k : ℝ) : 
  (∃ x₀ ∈ Set.Icc 0 2, ∀ x ∈ Set.Icc 0 2, 
    (x^2 - 4*k*x + 4*k^2 + 2*k - 1) ≥ (x₀^2 - 4*k*x₀ + 4*k^2 + 2*k - 1)) → 
  k = 1 := by
sorry

end quadratic_minimum_l945_94505


namespace unique_solution_l945_94500

/-- The set of digits used in the equation -/
def Digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- The sum of all digits from 0 to 9 -/
def DigitsSum : Nat := Finset.sum Digits id

/-- The set of digits used on the left side of the equation -/
def LeftDigits : Finset Nat := {0, 1, 2, 4, 5, 7, 8, 9}

/-- The sum of digits on the left side of the equation -/
def LeftSum : Nat := Finset.sum LeftDigits id

/-- The two-digit number on the right side of the equation -/
def RightNumber : Nat := 36

/-- The statement that the equation is a valid solution -/
theorem unique_solution :
  (LeftSum = RightNumber) ∧
  (Digits \ LeftDigits).card = 2 ∧
  (∀ (s : Finset Nat), s ⊂ Digits → s.card = 8 → Finset.sum s id ≠ RightNumber) :=
by sorry

end unique_solution_l945_94500


namespace scalar_projection_implies_k_l945_94563

/-- Given vectors a and b in ℝ², prove that if the scalar projection of b onto a is 1, then the first component of b is 3. -/
theorem scalar_projection_implies_k (a b : ℝ × ℝ) :
  a = (3, 4) →
  b.2 = -1 →
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (a.1^2 + a.2^2) = 1 →
  b.1 = 3 := by
  sorry

end scalar_projection_implies_k_l945_94563


namespace third_islander_statement_l945_94535

-- Define the types of islanders
inductive IslanderType
| Knight
| Liar

-- Define the islanders
def A : IslanderType := IslanderType.Liar
def B : IslanderType := IslanderType.Knight
def C : IslanderType := IslanderType.Knight

-- Define the statements made by the islanders
def statement_A : Prop := ∀ x, x ≠ A → IslanderType.Liar = x
def statement_B : Prop := ∃! x, x ≠ B ∧ IslanderType.Knight = x

-- Theorem to prove
theorem third_islander_statement :
  (A = IslanderType.Liar) →
  (B = IslanderType.Knight) →
  (C = IslanderType.Knight) →
  statement_A →
  statement_B →
  (∃! x, x ≠ C ∧ IslanderType.Knight = x) :=
by sorry

end third_islander_statement_l945_94535


namespace remainder_proof_l945_94593

theorem remainder_proof (n : ℕ) (h1 : n = 88) (h2 : (3815 - 31) % n = 0) (h3 : ∃ r, (4521 - r) % n = 0) :
  4521 % n = 33 := by
  sorry

end remainder_proof_l945_94593


namespace erasers_problem_l945_94596

theorem erasers_problem (initial_erasers bought_erasers final_erasers : ℕ) : 
  bought_erasers = 42 ∧ final_erasers = 137 → initial_erasers = 95 :=
by sorry

end erasers_problem_l945_94596


namespace max_gcd_consecutive_terms_l945_94578

def a (n : ℕ) : ℕ := n.factorial + n

theorem max_gcd_consecutive_terms :
  ∃ (k : ℕ), k ≥ 2 ∧ 
  (∀ (n : ℕ), Nat.gcd (a n) (a (n + 1)) ≤ k) ∧
  (∃ (m : ℕ), Nat.gcd (a m) (a (m + 1)) = k) :=
sorry

end max_gcd_consecutive_terms_l945_94578


namespace quadratic_root_k_value_l945_94561

theorem quadratic_root_k_value (k : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + 3 * x - k = 0 ∧ x = 7) → k = 119 := by
  sorry

end quadratic_root_k_value_l945_94561


namespace range_of_m_l945_94595

/-- The proposition p: "The equation x^2 + 2mx + 1 = 0 has two distinct positive roots" -/
def p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ x₁^2 + 2*m*x₁ + 1 = 0 ∧ x₂^2 + 2*m*x₂ + 1 = 0

/-- The proposition q: "The equation x^2 + 2(m-2)x - 3m + 10 = 0 has no real roots" -/
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2*(m-2)*x - 3*m + 10 ≠ 0

/-- The set representing the range of m -/
def S : Set ℝ := {m | m ≤ -2 ∨ (-1 ≤ m ∧ m < 3)}

theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ∈ S := by sorry

end range_of_m_l945_94595


namespace client_cost_is_3400_l945_94539

def ladders_cost (num_ladders1 : ℕ) (rungs_per_ladder1 : ℕ) 
                 (num_ladders2 : ℕ) (rungs_per_ladder2 : ℕ) 
                 (cost_per_rung : ℕ) : ℕ :=
  (num_ladders1 * rungs_per_ladder1 + num_ladders2 * rungs_per_ladder2) * cost_per_rung

theorem client_cost_is_3400 :
  ladders_cost 10 50 20 60 2 = 3400 := by
  sorry

end client_cost_is_3400_l945_94539


namespace correct_factorization_l945_94594

theorem correct_factorization (x : ℝ) : 2 * x^2 + 3 * x + 1 = (2 * x + 1) * (x + 1) := by
  sorry

end correct_factorization_l945_94594


namespace initially_tagged_fish_l945_94525

/-- The number of fish initially caught and tagged -/
def T : ℕ := 70

/-- The number of fish caught in the second catch -/
def second_catch : ℕ := 50

/-- The number of tagged fish in the second catch -/
def tagged_in_second_catch : ℕ := 2

/-- The total number of fish in the pond -/
def total_fish : ℕ := 1750

/-- Theorem stating that T is the correct number of initially tagged fish -/
theorem initially_tagged_fish :
  (T : ℚ) / total_fish = tagged_in_second_catch / second_catch :=
by sorry

end initially_tagged_fish_l945_94525


namespace sin_cos_tan_order_l945_94580

theorem sin_cos_tan_order :
  ∃ (a b c : ℝ), a = Real.sin 2 ∧ b = Real.cos 2 ∧ c = Real.tan 2 ∧ c < b ∧ b < a := by
  sorry

end sin_cos_tan_order_l945_94580


namespace cellphone_survey_rate_increase_is_30_percent_l945_94557

/-- Calculates the percentage increase in pay rate for cellphone surveys -/
def cellphone_survey_rate_increase (regular_rate : ℚ) (total_surveys : ℕ) 
  (cellphone_surveys : ℕ) (total_earnings : ℚ) : ℚ :=
  let regular_earnings := regular_rate * total_surveys
  let additional_earnings := total_earnings - regular_earnings
  let additional_rate := additional_earnings / cellphone_surveys
  let cellphone_rate := regular_rate + additional_rate
  (cellphone_rate - regular_rate) / regular_rate * 100

/-- Theorem stating the percentage increase in pay rate for cellphone surveys -/
theorem cellphone_survey_rate_increase_is_30_percent :
  cellphone_survey_rate_increase 10 100 60 1180 = 30 := by
  sorry

end cellphone_survey_rate_increase_is_30_percent_l945_94557


namespace equation_solutions_sum_of_squares_complex_equation_l945_94564

theorem equation_solutions (x : ℝ) :
  (x^2 + 2) / x = 5 + 2/5 → x = 5 ∨ x = 2/5 := by sorry

theorem sum_of_squares (a b : ℝ) :
  a + 3/a = 7 ∧ b + 3/b = 7 → a^2 + b^2 = 43 := by sorry

theorem complex_equation (t k : ℝ) :
  (∃ x₁ x₂ : ℝ, 6/(x₁ - 1) = k - x₁ ∧ 6/(x₂ - 1) = k - x₂ ∧ x₁ = t + 1 ∧ x₂ = t^2 + 2) →
  k^2 - 4*k + 4*t^3 = 32 := by sorry

end equation_solutions_sum_of_squares_complex_equation_l945_94564


namespace parabola_vertex_l945_94537

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 5 * (x - 2)^2 + 6

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 6)

/-- Theorem: The vertex of the parabola y = 5(x-2)^2 + 6 is at the point (2, 6) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola vertex.1) ∧ 
  parabola vertex.1 = vertex.2 := by
sorry

end parabola_vertex_l945_94537


namespace unique_score_100_l945_94585

/-- Represents a competition score -/
structure CompetitionScore where
  total : Nat
  correct : Nat
  wrong : Nat
  score : Nat
  h1 : total = 25
  h2 : score = 25 + 5 * correct - wrong
  h3 : total = correct + wrong

/-- Checks if a given score uniquely determines the number of correct and wrong answers -/
def isUniquelyDetermined (s : Nat) : Prop :=
  ∃! cs : CompetitionScore, cs.score = s

theorem unique_score_100 :
  isUniquelyDetermined 100 ∧
  ∀ s, 95 < s ∧ s < 100 → ¬isUniquelyDetermined s :=
sorry

end unique_score_100_l945_94585


namespace circle_ratio_after_radius_increase_l945_94517

/-- 
For any circle with radius r, if the radius is increased by 2 units, 
the ratio of the new circumference to the new diameter is equal to π.
-/
theorem circle_ratio_after_radius_increase (r : ℝ) : 
  let new_radius : ℝ := r + 2
  let new_circumference : ℝ := 2 * Real.pi * new_radius
  let new_diameter : ℝ := 2 * new_radius
  new_circumference / new_diameter = Real.pi :=
by sorry

end circle_ratio_after_radius_increase_l945_94517


namespace sum_of_roots_l945_94513

theorem sum_of_roots (a b c d k : ℝ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  k > 0 →
  a + b = k →
  c + d = k^2 →
  c^2 - 4*a*c - 5*b = 0 →
  d^2 - 4*a*d - 5*b = 0 →
  a^2 - 4*c*a - 5*d = 0 →
  b^2 - 4*c*b - 5*d = 0 →
  a + b + c + d = k + k^2 :=
by
  sorry

end sum_of_roots_l945_94513


namespace sqrt_fifth_power_l945_94507

theorem sqrt_fifth_power : (Real.sqrt ((Real.sqrt 5)^4))^5 = 3125 := by
  sorry

end sqrt_fifth_power_l945_94507


namespace initial_pineapples_l945_94599

/-- Proves that the initial number of pineapples in the store was 86 -/
theorem initial_pineapples (sold : ℕ) (rotten : ℕ) (fresh : ℕ) 
  (h1 : sold = 48) 
  (h2 : rotten = 9) 
  (h3 : fresh = 29) : 
  sold + rotten + fresh = 86 := by
  sorry

end initial_pineapples_l945_94599


namespace smallest_x_value_l945_94548

theorem smallest_x_value (x : ℝ) : 
  ((5 * x - 20) / (4 * x - 5))^2 + ((5 * x - 20) / (4 * x - 5)) = 20 → x ≥ 0 :=
by
  sorry

end smallest_x_value_l945_94548


namespace intersection_length_l945_94532

/-- The length of the line segment formed by the intersection of y = x + 1 and x²/4 + y²/3 = 1 is 24/7 -/
theorem intersection_length :
  let l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 1}
  let C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2/4 + p.2^2/3 = 1}
  let A : Set (ℝ × ℝ) := l ∩ C
  ∃ p q : ℝ × ℝ, p ∈ A ∧ q ∈ A ∧ p ≠ q ∧ ‖p - q‖ = 24/7 := by
  sorry

end intersection_length_l945_94532


namespace tan_120_degrees_l945_94574

theorem tan_120_degrees : Real.tan (2 * Real.pi / 3) = -Real.sqrt 3 := by
  sorry

end tan_120_degrees_l945_94574


namespace x_minus_y_values_l945_94516

theorem x_minus_y_values (x y : ℝ) (hx : |x| = 4) (hy : |y| = 7) (hsum : x + y > 0) :
  x - y = -3 ∨ x - y = -11 := by
  sorry

end x_minus_y_values_l945_94516


namespace garden_area_l945_94522

/-- A rectangular garden with width one-third of its length and perimeter 72 meters has an area of 243 square meters. -/
theorem garden_area (width length : ℝ) : 
  width > 0 ∧ 
  length > 0 ∧ 
  width = length / 3 ∧ 
  2 * (width + length) = 72 →
  width * length = 243 := by
sorry

end garden_area_l945_94522


namespace pizza_delivery_problem_l945_94589

theorem pizza_delivery_problem (total_time : ℕ) (avg_time_per_stop : ℕ) 
  (two_pizza_stops : ℕ) (pizzas_per_two_pizza_stop : ℕ) :
  total_time = 40 →
  avg_time_per_stop = 4 →
  two_pizza_stops = 2 →
  pizzas_per_two_pizza_stop = 2 →
  ∃ (single_pizza_stops : ℕ),
    (single_pizza_stops + two_pizza_stops) * avg_time_per_stop = total_time ∧
    single_pizza_stops + two_pizza_stops * pizzas_per_two_pizza_stop = 12 :=
by sorry

end pizza_delivery_problem_l945_94589


namespace infinite_pairs_exist_l945_94570

theorem infinite_pairs_exist (m : ℕ+) :
  ∃ f : ℕ → ℕ × ℕ,
    Function.Injective f ∧
    ∀ n : ℕ,
      let (x, y) := f n
      Nat.gcd x y = 1 ∧
      y ∣ (x^2 + m) ∧
      x ∣ (y^2 + m) :=
by
  sorry

end infinite_pairs_exist_l945_94570


namespace largest_integer_inequality_l945_94587

theorem largest_integer_inequality : 
  (∀ x : ℤ, x ≤ 3 → (x : ℚ) / 5 + 6 / 7 < 8 / 5) ∧ 
  (4 : ℚ) / 5 + 6 / 7 ≥ 8 / 5 :=
by sorry

end largest_integer_inequality_l945_94587


namespace negation_of_all_men_honest_l945_94504

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for being a man and being honest
variable (man : U → Prop)
variable (honest : U → Prop)

-- State the theorem
theorem negation_of_all_men_honest :
  (¬ ∀ x, man x → honest x) ↔ (∃ x, man x ∧ ¬ honest x) :=
by sorry

end negation_of_all_men_honest_l945_94504


namespace set_cardinality_lower_bound_l945_94586

theorem set_cardinality_lower_bound
  (m : ℕ)
  (A : Finset ℤ)
  (B : Fin m → Finset ℤ)
  (h_m : m ≥ 2)
  (h_subset : ∀ k : Fin m, B k ⊆ A)
  (h_sum : ∀ k : Fin m, (B k).sum id = m ^ (k : ℕ).succ) :
  A.card ≥ m / 2 :=
sorry

end set_cardinality_lower_bound_l945_94586


namespace even_function_m_value_l945_94575

def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 2) * x + (m^2 - 7*m + 12)

theorem even_function_m_value (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) → m = 2 := by
  sorry

end even_function_m_value_l945_94575


namespace sphere_radius_calculation_l945_94526

-- Define the radius of the hemisphere
def hemisphere_radius : ℝ := 2

-- Define the number of smaller spheres
def num_spheres : ℕ := 8

-- State the theorem
theorem sphere_radius_calculation :
  ∃ (r : ℝ), 
    (2 / 3 * Real.pi * hemisphere_radius ^ 3 = num_spheres * (4 / 3 * Real.pi * r ^ 3)) ∧
    (r = (Real.sqrt 2) / 2) := by
  sorry

end sphere_radius_calculation_l945_94526


namespace point_transformation_l945_94560

/-- Rotation of a point (x, y) by 180° around (h, k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ := (2*h - x, 2*k - y)

/-- Reflection of a point (x, y) about y = x -/
def reflectYEqualX (x y : ℝ) : ℝ × ℝ := (y, x)

/-- The main theorem -/
theorem point_transformation (a b : ℝ) :
  let Q : ℝ × ℝ := (a, b)
  let rotated := rotate180 a b 2 3
  let final := reflectYEqualX rotated.1 rotated.2
  final = (3, -7) → a - b = 8 := by
sorry

end point_transformation_l945_94560


namespace smallest_a_for_distinct_roots_in_unit_interval_l945_94503

theorem smallest_a_for_distinct_roots_in_unit_interval :
  ∃ (b c : ℤ), 
    (∃ (x y : ℝ), 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≠ y ∧ 
      5 * x^2 - b * x + c = 0 ∧ 5 * y^2 - b * y + c = 0) ∧
    (∀ (a : ℕ), a < 5 → 
      ¬∃ (b c : ℤ), ∃ (x y : ℝ), 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≠ y ∧ 
        a * x^2 - b * x + c = 0 ∧ a * y^2 - b * y + c = 0) :=
by sorry

end smallest_a_for_distinct_roots_in_unit_interval_l945_94503


namespace trigonometric_identity_l945_94541

theorem trigonometric_identity (α : Real) :
  (Real.tan (π/4 - α/2) * (1 - Real.cos (3*π/2 - α)) / Real.cos α - 2 * Real.cos (2*α)) /
  (Real.tan (π/4 - α/2) * (1 + Real.sin (4*π + α)) / Real.cos α + 2 * Real.cos (2*α)) =
  Real.tan (π/6 + α) * Real.tan (α - π/6) := by
  sorry

end trigonometric_identity_l945_94541


namespace symmetric_roots_iff_b_eq_two_or_four_l945_94549

/-- The polynomial in question -/
def P (b : ℝ) (z : ℂ) : ℂ :=
  z^5 - 8*z^4 + 12*b*z^3 - 4*(3*b^2 + 4*b - 4)*z^2 + 2*z + 2

/-- The roots of the polynomial form a symmetric pattern around the origin -/
def symmetric_roots (b : ℝ) : Prop :=
  ∃ (r : Finset ℂ), Finset.card r = 5 ∧ 
    (∀ z ∈ r, P b z = 0) ∧
    (∀ z ∈ r, -z ∈ r)

/-- The main theorem stating the condition for symmetric roots -/
theorem symmetric_roots_iff_b_eq_two_or_four :
  ∀ b : ℝ, symmetric_roots b ↔ b = 2 ∨ b = 4 := by sorry

end symmetric_roots_iff_b_eq_two_or_four_l945_94549


namespace product_one_to_five_l945_94559

theorem product_one_to_five : (List.range 5).foldl (·*·) 1 = 120 := by
  sorry

end product_one_to_five_l945_94559


namespace quadratic_equation_root_difference_l945_94545

theorem quadratic_equation_root_difference (k : ℝ) : 
  (∃ x₁ x₂ : ℂ, x₁ ≠ x₂ ∧ 2 * x₁^2 + k * x₁ + 26 = 0 ∧ 2 * x₂^2 + k * x₂ + 26 = 0) →
  Complex.abs (x₁ - x₂) = 6 →
  k = 4 * Real.sqrt 22 :=
by sorry

end quadratic_equation_root_difference_l945_94545
