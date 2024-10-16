import Mathlib

namespace NUMINAMATH_CALUDE_twentyfour_game_solution_l3957_395734

/-- A type representing the allowed arithmetic operations -/
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

/-- A type representing an arithmetic expression -/
inductive Expr
  | Const (n : Int)
  | BinOp (op : Operation) (e1 e2 : Expr)

/-- Evaluate an expression -/
def eval : Expr → Int
  | Expr.Const n => n
  | Expr.BinOp Operation.Add e1 e2 => eval e1 + eval e2
  | Expr.BinOp Operation.Sub e1 e2 => eval e1 - eval e2
  | Expr.BinOp Operation.Mul e1 e2 => eval e1 * eval e2
  | Expr.BinOp Operation.Div e1 e2 => eval e1 / eval e2

/-- Check if an expression uses all given numbers exactly once -/
def usesAllNumbers (e : Expr) (nums : List Int) : Bool :=
  match e with
  | Expr.Const n => nums == [n]
  | Expr.BinOp _ e1 e2 =>
    let nums1 := nums.filter (λ n => n ∉ collectNumbers e2)
    let nums2 := nums.filter (λ n => n ∉ collectNumbers e1)
    usesAllNumbers e1 nums1 && usesAllNumbers e2 nums2
where
  collectNumbers : Expr → List Int
    | Expr.Const n => [n]
    | Expr.BinOp _ e1 e2 => collectNumbers e1 ++ collectNumbers e2

theorem twentyfour_game_solution :
  ∃ (e : Expr), usesAllNumbers e [3, -5, 6, -8] ∧ eval e = 24 := by
  sorry

end NUMINAMATH_CALUDE_twentyfour_game_solution_l3957_395734


namespace NUMINAMATH_CALUDE_coloring_books_distribution_l3957_395758

def books_per_shelf (initial_stock : ℕ) (books_sold : ℕ) (num_shelves : ℕ) : ℕ :=
  (initial_stock - books_sold) / num_shelves

theorem coloring_books_distribution 
  (initial_stock : ℕ) 
  (books_sold : ℕ) 
  (num_shelves : ℕ) 
  (h1 : initial_stock = 40) 
  (h2 : books_sold = 20) 
  (h3 : num_shelves = 5) :
  books_per_shelf initial_stock books_sold num_shelves = 4 := by
  sorry

end NUMINAMATH_CALUDE_coloring_books_distribution_l3957_395758


namespace NUMINAMATH_CALUDE_irrational_between_3_and_4_l3957_395702

theorem irrational_between_3_and_4 : ∃! x : ℝ, (x = Real.sqrt 7 ∨ x = 2 * Real.sqrt 2 ∨ x = Real.sqrt 13 ∨ x = Real.sqrt 17) ∧ 3 < x ∧ x < 4 :=
by sorry

end NUMINAMATH_CALUDE_irrational_between_3_and_4_l3957_395702


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3957_395787

-- Define the sets A and B
def A : Set ℝ := {x | 2 / x > 1}
def B : Set ℝ := {x | Real.log x < 0}

-- Define the union of A and B
def AunionB : Set ℝ := A ∪ B

-- Theorem statement
theorem union_of_A_and_B : AunionB = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3957_395787


namespace NUMINAMATH_CALUDE_classroom_pictures_l3957_395799

/-- The number of oil paintings on the walls of the classroom -/
def oil_paintings : ℕ := 9

/-- The number of watercolor paintings on the walls of the classroom -/
def watercolor_paintings : ℕ := 7

/-- The total number of pictures on the walls of the classroom -/
def total_pictures : ℕ := oil_paintings + watercolor_paintings

theorem classroom_pictures : total_pictures = 16 := by
  sorry

end NUMINAMATH_CALUDE_classroom_pictures_l3957_395799


namespace NUMINAMATH_CALUDE_triangle_area_l3957_395770

def a : Fin 2 → ℝ := ![4, -1]
def b : Fin 2 → ℝ := ![3, 5]

theorem triangle_area : 
  (1/2 : ℝ) * |a 0 * b 1 - a 1 * b 0| = 23/2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3957_395770


namespace NUMINAMATH_CALUDE_target_average_income_l3957_395777

def past_incomes : List ℝ := [406, 413, 420, 436, 395]
def next_two_weeks_avg : ℝ := 365
def total_weeks : ℕ := 7

theorem target_average_income :
  let total_past_income := past_incomes.sum
  let total_next_two_weeks := 2 * next_two_weeks_avg
  let total_income := total_past_income + total_next_two_weeks
  total_income / total_weeks = 400 := by
  sorry

end NUMINAMATH_CALUDE_target_average_income_l3957_395777


namespace NUMINAMATH_CALUDE_wrapping_paper_area_l3957_395766

/-- The area of wrapping paper needed to cover a rectangular box with a small cube on top -/
theorem wrapping_paper_area (w h : ℝ) (w_pos : 0 < w) (h_pos : 0 < h) :
  let box_width := 2 * w
  let box_length := w
  let box_height := h
  let cube_side := w / 2
  let total_height := box_height + cube_side
  let paper_width := box_width + 2 * total_height
  let paper_length := box_length + 2 * total_height
  paper_width * paper_length = (3 * w + 2 * h) * (2 * w + 2 * h) :=
by sorry


end NUMINAMATH_CALUDE_wrapping_paper_area_l3957_395766


namespace NUMINAMATH_CALUDE_multiple_of_smaller_number_l3957_395773

theorem multiple_of_smaller_number (s l : ℝ) (h1 : s + l = 24) (h2 : s = 10) : ∃ m : ℝ, m * s = 5 * l ∧ m = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_smaller_number_l3957_395773


namespace NUMINAMATH_CALUDE_travel_options_count_l3957_395731

/-- The number of travel options from A to C given the number of trains from A to B and ferries from B to C -/
def travelOptions (trains : ℕ) (ferries : ℕ) : ℕ :=
  trains * ferries

/-- Theorem stating that the number of travel options from A to C is 6 -/
theorem travel_options_count :
  travelOptions 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_travel_options_count_l3957_395731


namespace NUMINAMATH_CALUDE_tripled_room_painting_cost_l3957_395715

/-- Represents the cost of painting a room -/
structure PaintingCost where
  original : ℝ
  scaled : ℝ

/-- Represents the dimensions of a room -/
structure RoomDimensions where
  length : ℝ
  breadth : ℝ
  height : ℝ

/-- Calculates the wall area of a room given its dimensions -/
def wallArea (d : RoomDimensions) : ℝ :=
  2 * (d.length + d.breadth) * d.height

/-- Scales the dimensions of a room by a factor -/
def scaleDimensions (d : RoomDimensions) (factor : ℝ) : RoomDimensions :=
  { length := d.length * factor
  , breadth := d.breadth * factor
  , height := d.height * factor }

/-- Theorem: The cost of painting a room with tripled dimensions is Rs. 3150 
    given that the original cost is Rs. 350 -/
theorem tripled_room_painting_cost 
  (d : RoomDimensions) 
  (c : PaintingCost) 
  (h1 : c.original = 350) 
  (h2 : c.original / wallArea d = c.scaled / wallArea (scaleDimensions d 3)) : 
  c.scaled = 3150 := by
  sorry

end NUMINAMATH_CALUDE_tripled_room_painting_cost_l3957_395715


namespace NUMINAMATH_CALUDE_special_ellipse_properties_l3957_395790

/-- An ellipse with a vertex at (0,1) and focal length 2√3 -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h0 : a > b
  h1 : b > 0
  h2 : b = 1
  h3 : a^2 = b^2 + 3

/-- The intersection points of a line with the ellipse -/
def LineEllipseIntersection (E : SpecialEllipse) (k : ℝ) :=
  {x : ℝ × ℝ | ∃ t, x.1 = -2 + t ∧ x.2 = 1 + k*t ∧ (x.1^2 / E.a^2 + x.2^2 / E.b^2 = 1)}

/-- The x-intercepts of lines connecting (0,1) to the intersection points -/
def XIntercepts (E : SpecialEllipse) (k : ℝ) (B C : ℝ × ℝ) :=
  {x : ℝ | ∃ t, (t*B.1 = x ∧ t*B.2 = 1) ∨ (t*C.1 = x ∧ t*C.2 = 1)}

theorem special_ellipse_properties (E : SpecialEllipse) :
  (∀ x y, x^2/4 + y^2 = 1 ↔ x^2/E.a^2 + y^2/E.b^2 = 1) ∧
  (∀ k : ℝ, k ≠ 0 →
    ∀ B C : ℝ × ℝ, B ∈ LineEllipseIntersection E k → C ∈ LineEllipseIntersection E k → B ≠ C →
    ∀ M N : ℝ, M ∈ XIntercepts E k B C → N ∈ XIntercepts E k B C → M ≠ N →
    (M - N)^2 * |k| = 16) :=
sorry

end NUMINAMATH_CALUDE_special_ellipse_properties_l3957_395790


namespace NUMINAMATH_CALUDE_twelfth_even_multiple_of_4_l3957_395791

/-- The nth term in the sequence of positive integers that are both even and multiples of 4 -/
def evenMultipleOf4 (n : ℕ) : ℕ := 4 * n

/-- Theorem stating that the 12th term in the sequence of positive integers 
    that are both even and multiples of 4 is equal to 48 -/
theorem twelfth_even_multiple_of_4 : evenMultipleOf4 12 = 48 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_even_multiple_of_4_l3957_395791


namespace NUMINAMATH_CALUDE_tuesday_rainfall_correct_l3957_395708

/-- Represents the rainfall data for three days -/
structure RainfallData where
  total : Float
  monday : Float
  wednesday : Float

/-- Calculates the rainfall on Tuesday given the rainfall data for three days -/
def tuesdayRainfall (data : RainfallData) : Float :=
  data.total - (data.monday + data.wednesday)

/-- Theorem stating that the rainfall on Tuesday is correctly calculated -/
theorem tuesday_rainfall_correct (data : RainfallData) 
  (h1 : data.total = 0.6666666666666666)
  (h2 : data.monday = 0.16666666666666666)
  (h3 : data.wednesday = 0.08333333333333333) :
  tuesdayRainfall data = 0.41666666666666663 := by
  sorry

#eval tuesdayRainfall { 
  total := 0.6666666666666666, 
  monday := 0.16666666666666666, 
  wednesday := 0.08333333333333333 
}

end NUMINAMATH_CALUDE_tuesday_rainfall_correct_l3957_395708


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3957_395720

theorem complex_equation_solution :
  ∀ (z : ℂ), (2 + Complex.I) * z = 5 * Complex.I → z = 1 + 2 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3957_395720


namespace NUMINAMATH_CALUDE_square_removal_domino_tiling_l3957_395700

theorem square_removal_domino_tiling (n m : ℕ) (hn : n = 2011) (hm : m = 11) :
  (∃ (k : ℕ), k = (n - m + 1)^2 / 2 + ((n - m + 1)^2 % 2)) ∧
  (∀ (k : ℕ), k = (n - m + 1)^2 / 2 + ((n - m + 1)^2 % 2) → k = 2002001) :=
by sorry

end NUMINAMATH_CALUDE_square_removal_domino_tiling_l3957_395700


namespace NUMINAMATH_CALUDE_vector_equality_l3957_395742

/-- Given four non-overlapping points P, A, B, C on a plane, 
    if PA + PB + PC = 0 and AB + AC = m * AP, then m = 3 -/
theorem vector_equality (P A B C : ℝ × ℝ) (m : ℝ) 
  (h1 : (A.1 - P.1, A.2 - P.2) + (B.1 - P.1, B.2 - P.2) + (C.1 - P.1, C.2 - P.2) = (0, 0))
  (h2 : (B.1 - A.1, B.2 - A.2) + (C.1 - A.1, C.2 - A.2) = (m * (A.1 - P.1), m * (A.2 - P.2)))
  (h3 : P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C) :
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_l3957_395742


namespace NUMINAMATH_CALUDE_lloyds_hourly_rate_l3957_395767

def regular_hours : ℝ := 7.5
def overtime_rate : ℝ := 1.5
def total_hours : ℝ := 10.5
def total_earnings : ℝ := 66

def hourly_rate : ℝ := 5.5

theorem lloyds_hourly_rate : 
  regular_hours * hourly_rate + 
  (total_hours - regular_hours) * overtime_rate * hourly_rate = 
  total_earnings := by sorry

end NUMINAMATH_CALUDE_lloyds_hourly_rate_l3957_395767


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3957_395711

theorem fraction_to_decimal (numerator denominator : ℕ) (decimal : ℚ) : 
  numerator = 16 → denominator = 50 → decimal = 0.32 → 
  (numerator : ℚ) / (denominator : ℚ) = decimal :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3957_395711


namespace NUMINAMATH_CALUDE_mary_screws_problem_l3957_395712

theorem mary_screws_problem (initial_screws : Nat) (buy_multiplier : Nat) (num_sections : Nat) : 
  initial_screws = 8 → buy_multiplier = 2 → num_sections = 4 → 
  (initial_screws + initial_screws * buy_multiplier) / num_sections = 6 := by
sorry

end NUMINAMATH_CALUDE_mary_screws_problem_l3957_395712


namespace NUMINAMATH_CALUDE_vacation_pictures_count_l3957_395755

def zoo_pictures : ℕ := 150
def aquarium_pictures : ℕ := 210
def museum_pictures : ℕ := 90
def amusement_park_pictures : ℕ := 120

def zoo_deletion_percentage : ℚ := 25 / 100
def aquarium_deletion_percentage : ℚ := 15 / 100
def amusement_park_deletion : ℕ := 20
def museum_addition : ℕ := 30

theorem vacation_pictures_count :
  ⌊(zoo_pictures : ℚ) * (1 - zoo_deletion_percentage)⌋ +
  ⌊(aquarium_pictures : ℚ) * (1 - aquarium_deletion_percentage)⌋ +
  (museum_pictures + museum_addition) +
  (amusement_park_pictures - amusement_park_deletion) = 512 := by
  sorry

end NUMINAMATH_CALUDE_vacation_pictures_count_l3957_395755


namespace NUMINAMATH_CALUDE_cousin_distribution_count_l3957_395744

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 5 cousins -/
def num_cousins : ℕ := 5

/-- There are 5 rooms -/
def num_rooms : ℕ := 5

/-- The number of ways to distribute the cousins among the rooms -/
def num_distributions : ℕ := distribute num_cousins num_rooms

theorem cousin_distribution_count : num_distributions = 137 := by sorry

end NUMINAMATH_CALUDE_cousin_distribution_count_l3957_395744


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3957_395763

theorem imaginary_part_of_z (z : ℂ) (h : z * Complex.I = 2 + Complex.I) :
  Complex.im z = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3957_395763


namespace NUMINAMATH_CALUDE_todd_initial_gum_value_l3957_395748

/-- The number of gum pieces Steve gave to Todd -/
def steve_gum : ℕ := 16

/-- The total number of gum pieces Todd has after receiving gum from Steve -/
def todd_total_gum : ℕ := 54

/-- The initial number of gum pieces Todd had -/
def todd_initial_gum : ℕ := todd_total_gum - steve_gum

theorem todd_initial_gum_value : todd_initial_gum = 38 := by
  sorry

end NUMINAMATH_CALUDE_todd_initial_gum_value_l3957_395748


namespace NUMINAMATH_CALUDE_rates_sum_of_squares_l3957_395779

/-- Represents the rates of biking, jogging, and swimming in km/h -/
structure Rates where
  biking : ℕ
  jogging : ℕ
  swimming : ℕ

/-- The sum of the squares of the rates -/
def sumOfSquares (r : Rates) : ℕ :=
  r.biking ^ 2 + r.jogging ^ 2 + r.swimming ^ 2

theorem rates_sum_of_squares : ∃ r : Rates,
  (3 * r.biking + 2 * r.jogging + 2 * r.swimming = 112) ∧
  (2 * r.biking + 3 * r.jogging + 4 * r.swimming = 129) ∧
  (sumOfSquares r = 1218) := by
  sorry

end NUMINAMATH_CALUDE_rates_sum_of_squares_l3957_395779


namespace NUMINAMATH_CALUDE_monomial_exponents_sum_l3957_395795

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (a b : ℕ) (c d : ℤ) : Prop :=
  a = 3 ∧ b = 1 ∧ c = 3 ∧ d = 1

theorem monomial_exponents_sum (m n : ℕ) :
  like_terms m 1 3 n → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_monomial_exponents_sum_l3957_395795


namespace NUMINAMATH_CALUDE_bite_size_samples_per_half_l3957_395724

def total_pies : ℕ := 13
def halves_per_pie : ℕ := 2
def total_tasters : ℕ := 130

theorem bite_size_samples_per_half : 
  (total_tasters / (total_pies * halves_per_pie) : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_bite_size_samples_per_half_l3957_395724


namespace NUMINAMATH_CALUDE_algae_free_day_l3957_395723

/-- The number of days it takes for the pond to be completely covered in algae -/
def total_days : ℕ := 20

/-- The fraction of the pond covered by algae on a given day -/
def algae_coverage (day : ℕ) : ℚ :=
  if day ≥ total_days then 1
  else (1 / 2) ^ (total_days - day)

/-- The day on which the pond is 87.5% algae-free -/
def target_day : ℕ :=
  total_days - 3

theorem algae_free_day :
  algae_coverage target_day = 1 - (7 / 8) :=
sorry

end NUMINAMATH_CALUDE_algae_free_day_l3957_395723


namespace NUMINAMATH_CALUDE_inscribed_angle_chord_length_l3957_395751

/-- Given a circle with radius R and an inscribed angle α that subtends a chord of length a,
    prove that a = 2R sin α. -/
theorem inscribed_angle_chord_length (R : ℝ) (α : ℝ) (a : ℝ) 
    (h_circle : R > 0) 
    (h_angle : 0 < α ∧ α < π) 
    (h_chord : a > 0) : 
  a = 2 * R * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_inscribed_angle_chord_length_l3957_395751


namespace NUMINAMATH_CALUDE_oil_price_reduction_l3957_395705

/-- Prove that the amount spent on oil is 1500 given the conditions --/
theorem oil_price_reduction (original_price reduced_price amount_spent : ℝ) 
  (h1 : reduced_price = original_price * (1 - 0.2))
  (h2 : reduced_price = 30)
  (h3 : amount_spent / reduced_price - amount_spent / original_price = 10) :
  amount_spent = 1500 := by
  sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l3957_395705


namespace NUMINAMATH_CALUDE_trihedral_angle_sum_bounds_l3957_395796

-- Define a trihedral angle
structure TrihedralAngle where
  α : Real
  β : Real
  γ : Real

-- State the theorem
theorem trihedral_angle_sum_bounds (t : TrihedralAngle) :
  180 < t.α + t.β + t.γ ∧ t.α + t.β + t.γ < 540 := by
  sorry

end NUMINAMATH_CALUDE_trihedral_angle_sum_bounds_l3957_395796


namespace NUMINAMATH_CALUDE_ironman_age_is_48_l3957_395785

-- Define the ages as natural numbers
def thor_age : ℕ := 1456
def captain_america_age : ℕ := thor_age / 13
def peter_parker_age : ℕ := captain_america_age / 7
def doctor_strange_age : ℕ := peter_parker_age * 4
def ironman_age : ℕ := peter_parker_age + 32

-- State the theorem
theorem ironman_age_is_48 :
  (thor_age = 13 * captain_america_age) ∧
  (captain_america_age = 7 * peter_parker_age) ∧
  (4 * peter_parker_age = doctor_strange_age) ∧
  (ironman_age = peter_parker_age + 32) ∧
  (thor_age = 1456) →
  ironman_age = 48 := by
  sorry

end NUMINAMATH_CALUDE_ironman_age_is_48_l3957_395785


namespace NUMINAMATH_CALUDE_six_digit_square_from_three_squares_l3957_395781

/-- A function that concatenates three two-digit numbers into a six-digit number -/
def concatenate (a b c : Nat) : Nat :=
  10000 * a + 100 * b + c

/-- A predicate that checks if a number is a two-digit perfect square -/
def is_two_digit_perfect_square (n : Nat) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ ∃ k, k * k = n

/-- The main theorem statement -/
theorem six_digit_square_from_three_squares :
  ∀ a b c : Nat,
    is_two_digit_perfect_square a →
    is_two_digit_perfect_square b →
    is_two_digit_perfect_square c →
    (∃ t : Nat, t * t = concatenate a b c) →
    concatenate a b c = 166464 ∨ concatenate a b c = 646416 := by
  sorry


end NUMINAMATH_CALUDE_six_digit_square_from_three_squares_l3957_395781


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l3957_395752

theorem unique_root_quadratic (X Y Z : ℝ) (hX : X ≠ 0) (hY : Y ≠ 0) (hZ : Z ≠ 0) :
  (∀ t : ℝ, X * t^2 - Y * t + Z = 0 ↔ t = Y) → X = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l3957_395752


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3957_395749

theorem simplify_trig_expression : 
  (Real.sin (35 * π / 180))^2 - 1/2 = -2 * (Real.cos (10 * π / 180) * Real.cos (80 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3957_395749


namespace NUMINAMATH_CALUDE_polynomial_shift_root_existence_l3957_395793

/-- A polynomial of degree 10 with leading coefficient 1 -/
def Polynomial10 := {p : Polynomial ℝ // p.degree = 10 ∧ p.leadingCoeff = 1}

theorem polynomial_shift_root_existence (P Q : Polynomial10) 
  (h : ∀ x : ℝ, P.val.eval x ≠ Q.val.eval x) :
  ∃ x : ℝ, (P.val.eval (x + 1)) = (Q.val.eval (x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_shift_root_existence_l3957_395793


namespace NUMINAMATH_CALUDE_cosine_function_properties_l3957_395710

/-- 
Given a cosine function y = a * cos(b * x + c) where:
1. The minimum occurs at x = 0
2. The peak-to-peak amplitude is 6
Prove that c = π
-/
theorem cosine_function_properties (a b c : ℝ) : 
  (∀ x, a * Real.cos (b * x + c) ≥ a * Real.cos c) →  -- minimum at x = 0
  (2 * |a| = 6) →                                     -- peak-to-peak amplitude is 6
  c = π :=
by sorry

end NUMINAMATH_CALUDE_cosine_function_properties_l3957_395710


namespace NUMINAMATH_CALUDE_extreme_values_and_max_l3957_395762

/-- The function f(x) with parameters a, b, and c -/
def f (a b c x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8 * c

/-- The derivative of f(x) with respect to x -/
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 6 * a * x + 3 * b

theorem extreme_values_and_max (a b c : ℝ) :
  (f' a b 1 = 0 ∧ f' a b 2 = 0) →
  (a = -3 ∧ b = 4) ∧
  (c = -2 → ∀ x ∈ Set.Icc 0 3, f a b c x ≤ -7) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_and_max_l3957_395762


namespace NUMINAMATH_CALUDE_no_multiple_of_four_l3957_395709

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_form_1C34 (n : ℕ) : Prop :=
  ∃ C : ℕ, C < 10 ∧ n = 1000 + 100 * C + 34

theorem no_multiple_of_four :
  ¬∃ n : ℕ, is_four_digit n ∧ has_form_1C34 n ∧ 4 ∣ n :=
sorry

end NUMINAMATH_CALUDE_no_multiple_of_four_l3957_395709


namespace NUMINAMATH_CALUDE_minimize_sum_of_distances_l3957_395739

/-- Given points A, B, and C in a 2D plane, where:
    A has coordinates (4, 6)
    B has coordinates (3, 0)
    C has coordinates (k, 0)
    This theorem states that the value of k that minimizes
    the sum of distances AC + BC is 3. -/
theorem minimize_sum_of_distances :
  let A : ℝ × ℝ := (4, 6)
  let B : ℝ × ℝ := (3, 0)
  let C : ℝ → ℝ × ℝ := λ k => (k, 0)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let total_distance (k : ℝ) : ℝ := distance A (C k) + distance B (C k)
  ∃ k₀ : ℝ, k₀ = 3 ∧ ∀ k : ℝ, total_distance k₀ ≤ total_distance k :=
by sorry

end NUMINAMATH_CALUDE_minimize_sum_of_distances_l3957_395739


namespace NUMINAMATH_CALUDE_largest_sample_number_l3957_395771

def systematic_sampling (total : ℕ) (start : ℕ) (interval : ℕ) : ℕ :=
  let sample_size := total / interval
  start + interval * (sample_size - 1)

theorem largest_sample_number :
  systematic_sampling 500 7 25 = 482 := by
  sorry

end NUMINAMATH_CALUDE_largest_sample_number_l3957_395771


namespace NUMINAMATH_CALUDE_gcd_problem_l3957_395721

theorem gcd_problem (b : ℤ) (h : 504 ∣ b) : 
  Nat.gcd (4*b^3 + 2*b^2 + 5*b + 63).natAbs b.natAbs = 63 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3957_395721


namespace NUMINAMATH_CALUDE_negation_equivalence_l3957_395703

-- Define the original proposition
def original_proposition (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → x^2 + a*x + 3 ≥ 0

-- Define the negation of the proposition
def negation_proposition (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ x^2 + a*x + 3 < 0

-- Theorem stating that the negation is correct
theorem negation_equivalence (a : ℝ) :
  ¬(original_proposition a) ↔ negation_proposition a :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3957_395703


namespace NUMINAMATH_CALUDE_equal_diagonals_implies_quad_or_pent_l3957_395737

/-- A convex n-gon with n ≥ 4 -/
structure ConvexNGon where
  n : ℕ
  convex : n ≥ 4

/-- The property that all diagonals of a polygon are equal -/
def all_diagonals_equal (F : ConvexNGon) : Prop := sorry

/-- The property that a polygon is a quadrilateral -/
def is_quadrilateral (F : ConvexNGon) : Prop := F.n = 4

/-- The property that a polygon is a pentagon -/
def is_pentagon (F : ConvexNGon) : Prop := F.n = 5

/-- Theorem: If all diagonals of a convex n-gon (n ≥ 4) are equal, 
    then it is either a quadrilateral or a pentagon -/
theorem equal_diagonals_implies_quad_or_pent (F : ConvexNGon) :
  all_diagonals_equal F → is_quadrilateral F ∨ is_pentagon F := by sorry

end NUMINAMATH_CALUDE_equal_diagonals_implies_quad_or_pent_l3957_395737


namespace NUMINAMATH_CALUDE_sqrt_plus_square_zero_l3957_395722

theorem sqrt_plus_square_zero (m n : ℝ) : 
  Real.sqrt (m + 1) + (n - 2)^2 = 0 → m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_square_zero_l3957_395722


namespace NUMINAMATH_CALUDE_cupcake_combinations_l3957_395792

/-- The number of ways to distribute n indistinguishable objects into k distinguishable bins -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The total number of cupcakes to be purchased -/
def total_cupcakes : ℕ := 7

/-- The number of cupcake types available -/
def cupcake_types : ℕ := 5

/-- The number of cupcake types that must have at least one selected -/
def required_types : ℕ := 4

/-- The number of remaining cupcakes after selecting one of each required type -/
def remaining_cupcakes : ℕ := total_cupcakes - required_types

theorem cupcake_combinations : 
  stars_and_bars remaining_cupcakes cupcake_types = 35 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_combinations_l3957_395792


namespace NUMINAMATH_CALUDE_a_value_proof_l3957_395783

theorem a_value_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_a_value_proof_l3957_395783


namespace NUMINAMATH_CALUDE_sheets_in_box_l3957_395784

/-- The number of sheets needed per printer -/
def sheets_per_printer : ℕ := 7

/-- The number of printers that can be filled -/
def num_printers : ℕ := 31

/-- The total number of sheets in the box -/
def total_sheets : ℕ := sheets_per_printer * num_printers

theorem sheets_in_box : total_sheets = 217 := by
  sorry

end NUMINAMATH_CALUDE_sheets_in_box_l3957_395784


namespace NUMINAMATH_CALUDE_discount_comparison_l3957_395772

/-- The cost difference between Option 2 and Option 1 for buying suits and ties -/
def cost_difference (x : ℝ) : ℝ :=
  (3600 + 36*x) - (40*x + 3200)

theorem discount_comparison (x : ℝ) (h : x > 20) :
  cost_difference x ≥ 0 ∧ cost_difference 30 > 0 := by
  sorry

#eval cost_difference 30

end NUMINAMATH_CALUDE_discount_comparison_l3957_395772


namespace NUMINAMATH_CALUDE_fifty_seventh_pair_l3957_395747

def pair_sequence : ℕ → ℕ × ℕ
| n => sorry

theorem fifty_seventh_pair :
  pair_sequence 57 = (2, 10) := by sorry

end NUMINAMATH_CALUDE_fifty_seventh_pair_l3957_395747


namespace NUMINAMATH_CALUDE_largest_number_l3957_395794

/-- Represents a number with a finite or repeating decimal expansion -/
structure DecimalNumber where
  integerPart : ℕ
  finitePart : List ℕ
  repeatingPart : List ℕ

/-- The set of numbers to compare -/
def numberSet : Set DecimalNumber := {
  ⟨8, [1, 2, 3, 5], []⟩,
  ⟨8, [1, 2, 3], [5]⟩,
  ⟨8, [1, 2, 3], [4, 5]⟩,
  ⟨8, [1, 2], [3, 4, 5]⟩,
  ⟨8, [1], [2, 3, 4, 5]⟩
}

/-- Converts a DecimalNumber to a real number -/
def toReal (d : DecimalNumber) : ℝ :=
  sorry

/-- Compares two DecimalNumbers -/
def greaterThan (a b : DecimalNumber) : Prop :=
  toReal a > toReal b

/-- Theorem stating that 8.123̅5 is the largest number in the set -/
theorem largest_number (n : DecimalNumber) :
  n ∈ numberSet →
  greaterThan ⟨8, [1, 2, 3], [5]⟩ n ∨ n = ⟨8, [1, 2, 3], [5]⟩ :=
  sorry

end NUMINAMATH_CALUDE_largest_number_l3957_395794


namespace NUMINAMATH_CALUDE_ball_distribution_ratio_l3957_395743

def num_balls : ℕ := 25
def num_bins : ℕ := 5

def count_distribution (d : List ℕ) : ℕ :=
  (List.prod (d.map (λ x => Nat.choose num_balls x))) / (Nat.factorial (List.length d))

theorem ball_distribution_ratio :
  let r := count_distribution [6, 7, 4, 4, 4] * Nat.factorial 5
  let s := count_distribution [5, 5, 5, 5, 5]
  (r : ℚ) / s = 10 := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_ratio_l3957_395743


namespace NUMINAMATH_CALUDE_sugar_amount_l3957_395776

/-- Represents the amounts of ingredients in pounds -/
structure Ingredients where
  sugar : ℝ
  flour : ℝ
  baking_soda : ℝ

/-- The ratios and conditions given in the problem -/
def satisfies_conditions (i : Ingredients) : Prop :=
  i.sugar / i.flour = 3 / 8 ∧
  i.flour / i.baking_soda = 10 ∧
  i.flour / (i.baking_soda + 60) = 8

/-- The theorem stating that under the given conditions, the amount of sugar is 900 pounds -/
theorem sugar_amount (i : Ingredients) :
  satisfies_conditions i → i.sugar = 900 := by
  sorry

end NUMINAMATH_CALUDE_sugar_amount_l3957_395776


namespace NUMINAMATH_CALUDE_lacrosse_football_difference_l3957_395797

/-- Represents the number of bottles filled for each team and the total --/
structure BottleCounts where
  total : ℕ
  football : ℕ
  soccer : ℕ
  rugby : ℕ
  lacrosse : ℕ

/-- The difference in bottles between lacrosse and football teams --/
def bottleDifference (counts : BottleCounts) : ℕ :=
  counts.lacrosse - counts.football

/-- Theorem stating the difference in bottles between lacrosse and football teams --/
theorem lacrosse_football_difference (counts : BottleCounts) 
  (h1 : counts.total = 254)
  (h2 : counts.football = 11 * 6)
  (h3 : counts.soccer = 53)
  (h4 : counts.rugby = 49)
  (h5 : counts.total = counts.football + counts.soccer + counts.rugby + counts.lacrosse) :
  bottleDifference counts = 20 := by
  sorry

#check lacrosse_football_difference

end NUMINAMATH_CALUDE_lacrosse_football_difference_l3957_395797


namespace NUMINAMATH_CALUDE_a_2n_is_perfect_square_l3957_395704

/-- Definition of a_n: number of natural numbers with digit sum n and digits in {1,3,4} -/
def a (n : ℕ) : ℕ := sorry

/-- Theorem: a_{2n} is a perfect square for all natural numbers n -/
theorem a_2n_is_perfect_square (n : ℕ) : ∃ k : ℕ, a (2 * n) = k ^ 2 := by sorry

end NUMINAMATH_CALUDE_a_2n_is_perfect_square_l3957_395704


namespace NUMINAMATH_CALUDE_complex_calculations_l3957_395714

theorem complex_calculations :
  let z₁ : ℂ := 1 - 2*I
  let z₂ : ℂ := 3 + 4*I
  let z₃ : ℂ := -2 + I
  let w₁ : ℂ := 1 + 2*I
  let w₂ : ℂ := 3 - 4*I
  (z₁ * z₂ * z₃ = 12 + 9*I) ∧
  (w₁ / w₂ = -1/5 + 2/5*I) := by
sorry

end NUMINAMATH_CALUDE_complex_calculations_l3957_395714


namespace NUMINAMATH_CALUDE_intersection_P_Q_l3957_395775

def P : Set ℕ := {0, 2, 4, 6}
def Q : Set ℕ := {x | x ≤ 3}

theorem intersection_P_Q : P ∩ Q = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l3957_395775


namespace NUMINAMATH_CALUDE_function_symmetry_origin_l3957_395768

/-- The function f(x) = x^3 + x is symmetric with respect to the origin. -/
theorem function_symmetry_origin (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + x
  f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_origin_l3957_395768


namespace NUMINAMATH_CALUDE_sphere_radius_is_correct_l3957_395765

/-- A truncated cone with a tangent sphere -/
structure TruncatedConeWithSphere where
  r_bottom : ℝ
  r_top : ℝ
  sphere_radius : ℝ
  is_tangent : Bool

/-- The specific truncated cone with tangent sphere from the problem -/
def problem_cone : TruncatedConeWithSphere :=
  { r_bottom := 20
  , r_top := 5
  , sphere_radius := 10
  , is_tangent := true }

/-- Theorem stating that the sphere radius is correct -/
theorem sphere_radius_is_correct (c : TruncatedConeWithSphere) :
  c.r_bottom = 20 ∧ c.r_top = 5 ∧ c.is_tangent = true → c.sphere_radius = 10 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_is_correct_l3957_395765


namespace NUMINAMATH_CALUDE_one_minus_repeating_8_l3957_395732

/-- The value of the repeating decimal 0.888... -/
def repeating_decimal_8 : ℚ := 8/9

/-- Proof that 1 - 0.888... = 1/9 -/
theorem one_minus_repeating_8 : 1 - repeating_decimal_8 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_one_minus_repeating_8_l3957_395732


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3957_395730

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (X^3 + 3•X^2 - 4) = (X^2 + X - 2) * q + 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3957_395730


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l3957_395769

theorem basketball_lineup_combinations (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 6) :
  (n.factorial / (n - k).factorial) = 360360 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l3957_395769


namespace NUMINAMATH_CALUDE_handshake_problem_l3957_395741

theorem handshake_problem :
  let n : ℕ := 6  -- number of people
  let handshakes := n * (n - 1) / 2  -- formula for total handshakes
  handshakes = 15
  := by sorry

end NUMINAMATH_CALUDE_handshake_problem_l3957_395741


namespace NUMINAMATH_CALUDE_problem_statement_l3957_395713

theorem problem_statement (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x + 2) = (-5 * x^2 + 20 * x + 40) / (x - 3)) →
  P + Q = 50 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3957_395713


namespace NUMINAMATH_CALUDE_multiple_is_two_l3957_395789

/-- The grading method used by a teacher for a test -/
structure GradingMethod where
  totalQuestions : ℕ
  studentScore : ℕ
  correctAnswers : ℕ
  scoreCalculation : ℕ → ℕ → ℕ → ℕ → ℕ

/-- The multiple used for incorrect responses in the grading method -/
def incorrectResponseMultiple (gm : GradingMethod) : ℕ :=
  let incorrectAnswers := gm.totalQuestions - gm.correctAnswers
  (gm.correctAnswers - gm.studentScore) / incorrectAnswers

/-- Theorem stating that the multiple used for incorrect responses is 2 -/
theorem multiple_is_two (gm : GradingMethod) 
  (h1 : gm.totalQuestions = 100)
  (h2 : gm.studentScore = 76)
  (h3 : gm.correctAnswers = 92)
  (h4 : gm.scoreCalculation = fun total correct incorrect multiple => 
    correct - multiple * incorrect) :
  incorrectResponseMultiple gm = 2 := by
  sorry


end NUMINAMATH_CALUDE_multiple_is_two_l3957_395789


namespace NUMINAMATH_CALUDE_mountain_climb_theorem_l3957_395745

/-- Represents the mountain climbing scenario -/
structure MountainClimb where
  x : ℝ  -- Height of the mountain in meters
  male_speed : ℝ  -- Speed of male team
  female_speed : ℝ  -- Speed of female team

/-- The main theorem about the mountain climbing scenario -/
theorem mountain_climb_theorem (mc : MountainClimb) 
  (h1 : mc.x / (mc.x - 600) = mc.male_speed / mc.female_speed)  -- Condition when male team reaches summit
  (h2 : mc.male_speed / mc.female_speed = 3 / 2)  -- Speed ratio
  : mc.male_speed / mc.female_speed = 3 / 2  -- 1. Speed ratio is 3:2
  ∧ mc.x = 1800  -- 2. Mountain height is 1800 meters
  ∧ ∀ b : ℝ, b > 0 → b / mc.male_speed < (600 - b) / mc.female_speed → b < 360  -- 3. Point B is less than 360 meters from summit
  := by sorry

end NUMINAMATH_CALUDE_mountain_climb_theorem_l3957_395745


namespace NUMINAMATH_CALUDE_area_of_triangle_abc_is_150_over_7_l3957_395707

/-- Given a circle with center O and radius r, and points A and B on a line passing through O,
    this function calculates the area of triangle ABC, where C is the intersection of tangents
    drawn from A and B to the circle. -/
def triangle_area_from_tangents (r OA AB : ℝ) : ℝ :=
  sorry

/-- Theorem stating that for a circle with radius 12 and points A and B such that OA = 15 and AB = 5,
    the area of triangle ABC formed by the intersection of tangents is 150/7. -/
theorem area_of_triangle_abc_is_150_over_7 :
  triangle_area_from_tangents 12 15 5 = 150 / 7 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_abc_is_150_over_7_l3957_395707


namespace NUMINAMATH_CALUDE_product_expansion_l3957_395780

theorem product_expansion (x : ℝ) : (2*x + 3) * (3*x^2 + 4*x + 1) = 6*x^3 + 17*x^2 + 14*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l3957_395780


namespace NUMINAMATH_CALUDE_rectangle_decomposition_theorem_l3957_395738

/-- A function that checks if a rectangle can be decomposed into n and m+n congruent squares -/
def has_unique_decomposition (m : ℕ+) : Prop :=
  ∃! n : ℕ+, ∃ a b : ℕ+, a^2 - b^2 = n ∧ a^2 - b^2 = m + n

/-- A function that checks if a number is an odd prime -/
def is_odd_prime (p : ℕ+) : Prop :=
  Nat.Prime p.val ∧ p.val % 2 = 1

/-- The main theorem stating the equivalence of the two conditions -/
theorem rectangle_decomposition_theorem (m : ℕ+) :
  has_unique_decomposition m ↔ 
  (∃ p : ℕ+, is_odd_prime p ∧ (m = p ∨ m = 2 * p ∨ m = 4 * p)) :=
sorry

end NUMINAMATH_CALUDE_rectangle_decomposition_theorem_l3957_395738


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l3957_395778

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := x + 2*y = 0

-- Theorem statement
theorem common_chord_of_circles :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → common_chord x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l3957_395778


namespace NUMINAMATH_CALUDE_ada_original_seat_l3957_395736

-- Define the type for seats
inductive Seat : Type
  | one : Seat
  | two : Seat
  | three : Seat
  | four : Seat
  | five : Seat

-- Define the type for friends
inductive Friend : Type
  | ada : Friend
  | bea : Friend
  | ceci : Friend
  | dee : Friend
  | edie : Friend

-- Define the seating arrangement as a function from Friend to Seat
def SeatingArrangement : Type := Friend → Seat

-- Define what it means for a seat to be an end seat
def isEndSeat (s : Seat) : Prop :=
  s = Seat.one ∨ s = Seat.five

-- Define the movement of friends
def moveRight (s : Seat) (n : Nat) : Seat :=
  match s, n with
  | Seat.one, 1 => Seat.two
  | Seat.one, 2 => Seat.three
  | Seat.two, 1 => Seat.three
  | Seat.two, 2 => Seat.four
  | Seat.three, 1 => Seat.four
  | Seat.three, 2 => Seat.five
  | Seat.four, 1 => Seat.five
  | _, _ => s  -- Default case: no movement or invalid movement

def moveLeft (s : Seat) (n : Nat) : Seat :=
  match s, n with
  | Seat.two, 1 => Seat.one
  | Seat.three, 1 => Seat.two
  | Seat.four, 1 => Seat.three
  | Seat.five, 1 => Seat.four
  | _, _ => s  -- Default case: no movement or invalid movement

-- Theorem statement
theorem ada_original_seat (initial final : SeatingArrangement) :
  (∀ f : Friend, f ≠ Friend.ada → initial f ≠ Seat.five) →  -- No one except possibly Ada starts in seat 5
  (initial Friend.bea = moveLeft (final Friend.bea) 2) →   -- Bea moved 2 seats right
  (initial Friend.ceci = moveRight (final Friend.ceci) 1) → -- Ceci moved 1 seat left
  (initial Friend.dee = final Friend.edie ∧ initial Friend.edie = final Friend.dee) → -- Dee and Edie switched
  (isEndSeat (final Friend.ada)) →  -- Ada ends up in an end seat
  (initial Friend.ada = Seat.two) :=  -- Prove Ada started in seat 2
sorry

end NUMINAMATH_CALUDE_ada_original_seat_l3957_395736


namespace NUMINAMATH_CALUDE_set_intersections_and_union_l3957_395759

def A : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 = 0}
def B : Set (ℝ × ℝ) := {p | 3 * p.1 + p.2 = 0}
def C : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 = 3}

theorem set_intersections_and_union :
  (A ∩ B = {(0, 0)}) ∧
  (A ∩ C = ∅) ∧
  ((A ∩ B) ∪ (B ∩ C) = {(0, 0), (3/5, -9/5)}) := by
  sorry

end NUMINAMATH_CALUDE_set_intersections_and_union_l3957_395759


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3957_395717

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | (x - a) * (x - 2*a) < 0}
  S = if a < 0 then Set.Ioo (2*a) a
      else if a = 0 then ∅
      else Set.Ioo a (2*a) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3957_395717


namespace NUMINAMATH_CALUDE_ball_distribution_ratio_l3957_395733

theorem ball_distribution_ratio : 
  let total_balls : ℕ := 20
  let num_bins : ℕ := 5
  let config_A : List ℕ := [2, 6, 4, 4, 4]
  let config_B : List ℕ := [4, 4, 4, 4, 4]
  
  let prob_A := (Nat.choose num_bins 1) * (Nat.choose (num_bins - 1) 1) * 
                (Nat.choose total_balls 2) * (Nat.choose (total_balls - 2) 6) * 
                (Nat.choose (total_balls - 2 - 6) 4) * (Nat.choose (total_balls - 2 - 6 - 4) 4) * 
                (Nat.choose (total_balls - 2 - 6 - 4 - 4) 4)
  
  let prob_B := (Nat.choose total_balls 4) * (Nat.choose (total_balls - 4) 4) * 
                (Nat.choose (total_balls - 4 - 4) 4) * (Nat.choose (total_balls - 4 - 4 - 4) 4) * 
                (Nat.choose (total_balls - 4 - 4 - 4 - 4) 4)
  
  prob_A / prob_B = 10 := by
  sorry

#check ball_distribution_ratio

end NUMINAMATH_CALUDE_ball_distribution_ratio_l3957_395733


namespace NUMINAMATH_CALUDE_kyle_gas_and_maintenance_amount_l3957_395718

/-- Calculates the amount left for gas and maintenance given Kyle's income and expenses --/
def amount_for_gas_and_maintenance (monthly_income : ℝ) (rent : ℝ) (utilities : ℝ) 
  (retirement_savings : ℝ) (groceries : ℝ) (insurance : ℝ) (miscellaneous : ℝ) 
  (car_payment : ℝ) : ℝ :=
  monthly_income - (rent + utilities + retirement_savings + groceries + insurance + miscellaneous + car_payment)

/-- Theorem stating that Kyle's amount left for gas and maintenance is $350 --/
theorem kyle_gas_and_maintenance_amount :
  amount_for_gas_and_maintenance 3200 1250 150 400 300 200 200 350 = 350 := by
  sorry

end NUMINAMATH_CALUDE_kyle_gas_and_maintenance_amount_l3957_395718


namespace NUMINAMATH_CALUDE_correct_balance_amount_l3957_395727

/-- The amount Carlos must give LeRoy to balance their adjusted shares -/
def balance_amount (A B C : ℝ) : ℝ := 0.35 * A - 0.65 * B + 0.35 * C

/-- Theorem stating the correct amount Carlos must give LeRoy -/
theorem correct_balance_amount (A B C : ℝ) (hB_lt_A : B < A) (hB_lt_C : B < C) :
  balance_amount A B C = (0.35 * (A + B + C) - B) := by sorry

end NUMINAMATH_CALUDE_correct_balance_amount_l3957_395727


namespace NUMINAMATH_CALUDE_n_has_nine_digits_l3957_395719

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 30 -/
axiom n_div_30 : 30 ∣ n

/-- n^2 is a perfect cube -/
axiom n_sq_cube : ∃ k : ℕ, n^2 = k^3

/-- n^3 is a perfect square -/
axiom n_cube_square : ∃ k : ℕ, n^3 = k^2

/-- n is the smallest positive integer satisfying the conditions -/
axiom n_smallest : ∀ m : ℕ, m > 0 → (30 ∣ m) → (∃ k : ℕ, m^2 = k^3) → (∃ k : ℕ, m^3 = k^2) → m ≥ n

/-- The number of digits in n -/
def digits (x : ℕ) : ℕ := sorry

theorem n_has_nine_digits : digits n = 9 := by sorry

end NUMINAMATH_CALUDE_n_has_nine_digits_l3957_395719


namespace NUMINAMATH_CALUDE_minimum_cubes_required_l3957_395740

/-- Represents a 3D grid of unit cubes -/
def CubeGrid := List (List (List Bool))

/-- Checks if a cube in the grid shares at least one face with another cube -/
def sharesface (grid : CubeGrid) : Bool :=
  sorry

/-- Generates the front view of the grid -/
def frontView (grid : CubeGrid) : List (List Bool) :=
  sorry

/-- Generates the side view of the grid -/
def sideView (grid : CubeGrid) : List (List Bool) :=
  sorry

/-- The given front view -/
def givenFrontView : List (List Bool) :=
  [[true, true, false],
   [true, true, false],
   [true, false, false]]

/-- The given side view -/
def givenSideView : List (List Bool) :=
  [[true, true, true, false],
   [false, true, false, false],
   [false, false, true, false]]

/-- Counts the number of cubes in the grid -/
def countCubes (grid : CubeGrid) : Nat :=
  sorry

theorem minimum_cubes_required :
  ∃ (grid : CubeGrid),
    sharesface grid ∧
    frontView grid = givenFrontView ∧
    sideView grid = givenSideView ∧
    countCubes grid = 5 ∧
    (∀ (other : CubeGrid),
      sharesface other →
      frontView other = givenFrontView →
      sideView other = givenSideView →
      countCubes other ≥ 5) :=
  sorry

end NUMINAMATH_CALUDE_minimum_cubes_required_l3957_395740


namespace NUMINAMATH_CALUDE_cube_with_cut_corners_has_36_edges_l3957_395725

/-- A cube with cut corners is a polyhedron resulting from cutting off each corner of a cube
    such that the cutting planes do not intersect within or on the cube. -/
structure CubeWithCutCorners where
  -- We don't need to define the structure explicitly for this problem

/-- The number of edges in a cube with cut corners -/
def num_edges_cube_with_cut_corners : ℕ := 36

/-- Theorem stating that a cube with cut corners has 36 edges -/
theorem cube_with_cut_corners_has_36_edges (c : CubeWithCutCorners) :
  num_edges_cube_with_cut_corners = 36 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_cut_corners_has_36_edges_l3957_395725


namespace NUMINAMATH_CALUDE_number_divided_by_0_08_equals_12_5_l3957_395760

theorem number_divided_by_0_08_equals_12_5 (x : ℝ) : x / 0.08 = 12.5 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_0_08_equals_12_5_l3957_395760


namespace NUMINAMATH_CALUDE_surprise_shop_revenue_loss_l3957_395746

/-- Calculates the potential revenue loss for a shop during Christmas holiday closures over multiple years. -/
def potential_revenue_loss (days_closed : ℕ) (daily_revenue : ℕ) (years : ℕ) : ℕ :=
  days_closed * daily_revenue * years

/-- Proves that the total potential revenue lost by the "Surprise" shop during 6 years of Christmas holiday closures is $90,000. -/
theorem surprise_shop_revenue_loss :
  potential_revenue_loss 3 5000 6 = 90000 := by
  sorry

end NUMINAMATH_CALUDE_surprise_shop_revenue_loss_l3957_395746


namespace NUMINAMATH_CALUDE_area_traced_on_concentric_spheres_l3957_395716

/-- The area traced by a smaller sphere moving between two concentric spheres -/
theorem area_traced_on_concentric_spheres 
  (R1 R2 A1 : ℝ) 
  (h1 : 0 < R1) 
  (h2 : R1 < R2) 
  (h3 : 0 < A1) : 
  ∃ A2 : ℝ, A2 = A1 * (R2/R1)^2 := by
sorry

end NUMINAMATH_CALUDE_area_traced_on_concentric_spheres_l3957_395716


namespace NUMINAMATH_CALUDE_sand_pouring_problem_l3957_395774

/-- Represents the fraction of sand remaining after n pourings -/
def remaining_sand (n : ℕ) : ℚ :=
  2 / (n + 2)

/-- The number of pourings required to reach exactly 1/5 of the original sand -/
def required_pourings : ℕ := 8

theorem sand_pouring_problem :
  remaining_sand required_pourings = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_sand_pouring_problem_l3957_395774


namespace NUMINAMATH_CALUDE_probability_not_losing_l3957_395735

theorem probability_not_losing (p_win p_draw : ℝ) 
  (h_win : p_win = 0.3) 
  (h_draw : p_draw = 0.2) : 
  p_win + p_draw = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_losing_l3957_395735


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3957_395798

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : RealFunction)
  (h : ∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3957_395798


namespace NUMINAMATH_CALUDE_north_pond_duck_count_l3957_395788

/-- The number of ducks at Lake Michigan -/
def lake_michigan_ducks : ℕ := 100

/-- The number of ducks at North Pond -/
def north_pond_ducks : ℕ := 2 * lake_michigan_ducks + 6

theorem north_pond_duck_count : north_pond_ducks = 206 := by
  sorry

end NUMINAMATH_CALUDE_north_pond_duck_count_l3957_395788


namespace NUMINAMATH_CALUDE_fruit_count_l3957_395754

theorem fruit_count (apples pears tangerines : ℕ) : 
  apples = 45 →
  pears = apples - 21 →
  pears = tangerines - 18 →
  tangerines = 42 := by
sorry

end NUMINAMATH_CALUDE_fruit_count_l3957_395754


namespace NUMINAMATH_CALUDE_merchant_tea_cups_l3957_395757

theorem merchant_tea_cups (a b c : ℕ) 
  (h1 : a + b = 11) 
  (h2 : b + c = 15) 
  (h3 : a + c = 14) : 
  a + b + c = 20 := by
sorry

end NUMINAMATH_CALUDE_merchant_tea_cups_l3957_395757


namespace NUMINAMATH_CALUDE_jessica_quarters_problem_l3957_395753

/-- The number of quarters Jessica's sister gave her -/
def quarters_given (initial : ℕ) (final : ℕ) : ℕ := final - initial

theorem jessica_quarters_problem (initial : ℕ) (final : ℕ) 
  (h1 : initial = 8) 
  (h2 : final = 11) : 
  quarters_given initial final = 3 := by
  sorry

end NUMINAMATH_CALUDE_jessica_quarters_problem_l3957_395753


namespace NUMINAMATH_CALUDE_wipes_count_l3957_395729

/-- The number of wipes initially in the container -/
def initial_wipes : ℕ := 70

/-- The number of wipes used during the day -/
def wipes_used : ℕ := 20

/-- The number of wipes added after using some -/
def wipes_added : ℕ := 10

/-- The number of wipes left at night -/
def wipes_at_night : ℕ := 60

theorem wipes_count : initial_wipes - wipes_used + wipes_added = wipes_at_night := by
  sorry

end NUMINAMATH_CALUDE_wipes_count_l3957_395729


namespace NUMINAMATH_CALUDE_village_population_l3957_395756

/-- The number of residents who speak Bashkir -/
def bashkir_speakers : ℕ := 912

/-- The number of residents who speak Russian -/
def russian_speakers : ℕ := 653

/-- The number of residents who speak both Bashkir and Russian -/
def bilingual_speakers : ℕ := 435

/-- The total number of residents in the village -/
def total_residents : ℕ := bashkir_speakers + russian_speakers - bilingual_speakers

theorem village_population :
  total_residents = 1130 :=
by sorry

end NUMINAMATH_CALUDE_village_population_l3957_395756


namespace NUMINAMATH_CALUDE_product_value_l3957_395761

def product_term (n : ℕ) : ℚ :=
  (n * (n + 2)) / ((n + 1) * (n + 1))

def product_sequence : ℕ → ℚ
  | 0 => 1
  | n + 1 => product_sequence n * product_term (n + 1)

theorem product_value : product_sequence 98 = 50 / 99 := by
  sorry

end NUMINAMATH_CALUDE_product_value_l3957_395761


namespace NUMINAMATH_CALUDE_assignment_result_l3957_395728

def assignment_sequence (initial_a : ℕ) : ℕ :=
  let a₁ := initial_a
  let a₂ := a₁ + 1
  a₂

theorem assignment_result : assignment_sequence 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_assignment_result_l3957_395728


namespace NUMINAMATH_CALUDE_fraction_change_l3957_395782

theorem fraction_change (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (2*x + 2*y) / (2*x * 2*y) = (1/2) * ((x + y) / (x * y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_change_l3957_395782


namespace NUMINAMATH_CALUDE_kittens_found_on_monday_l3957_395786

def solve_cat_problem (initial_cats : ℕ) (tuesday_cats : ℕ) (adoptions : ℕ) (cats_per_adoption : ℕ) (final_cats : ℕ) : ℕ :=
  initial_cats + tuesday_cats - (adoptions * cats_per_adoption) - final_cats

theorem kittens_found_on_monday :
  solve_cat_problem 20 1 3 2 17 = 2 := by
  sorry

end NUMINAMATH_CALUDE_kittens_found_on_monday_l3957_395786


namespace NUMINAMATH_CALUDE_regular_polygon_reciprocal_sum_l3957_395701

/-- Given a regular polygon with n sides, where the reciprocal of the side length
    equals the sum of reciprocals of two specific diagonals, prove that n = 7. -/
theorem regular_polygon_reciprocal_sum (n : ℕ) (R : ℝ) (h_n : n ≥ 3) :
  (1 : ℝ) / (2 * R * Real.sin (π / n)) =
    1 / (2 * R * Real.sin (2 * π / n)) + 1 / (2 * R * Real.sin (3 * π / n)) →
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_reciprocal_sum_l3957_395701


namespace NUMINAMATH_CALUDE_aron_vacuum_time_l3957_395750

/-- Represents the cleaning schedule and total cleaning time for Aron. -/
structure CleaningSchedule where
  vacuum_frequency : Nat  -- Number of days Aron vacuums per week
  dust_time : Nat         -- Minutes Aron spends dusting per day
  dust_frequency : Nat    -- Number of days Aron dusts per week
  total_cleaning_time : Nat  -- Total minutes Aron spends cleaning per week

/-- Calculates the number of minutes Aron spends vacuuming each day. -/
def vacuum_time_per_day (schedule : CleaningSchedule) : Nat :=
  (schedule.total_cleaning_time - schedule.dust_time * schedule.dust_frequency) / schedule.vacuum_frequency

/-- Theorem stating that Aron spends 30 minutes vacuuming each day. -/
theorem aron_vacuum_time (schedule : CleaningSchedule) 
    (h1 : schedule.vacuum_frequency = 3)
    (h2 : schedule.dust_time = 20)
    (h3 : schedule.dust_frequency = 2)
    (h4 : schedule.total_cleaning_time = 130) :
    vacuum_time_per_day schedule = 30 := by
  sorry


end NUMINAMATH_CALUDE_aron_vacuum_time_l3957_395750


namespace NUMINAMATH_CALUDE_number_of_students_l3957_395764

def total_pencils : ℕ := 195
def pencils_per_student : ℕ := 3

theorem number_of_students : 
  total_pencils / pencils_per_student = 65 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l3957_395764


namespace NUMINAMATH_CALUDE_animal_pairing_theorem_l3957_395726

/-- Represents the number of dogs in the problem -/
def num_dogs : ℕ := 5

/-- Represents the number of cats in the problem -/
def num_cats : ℕ := 4

/-- Represents the number of bowls of milk in the problem -/
def num_bowls : ℕ := 7

/-- Represents the total number of animals (dogs and cats) -/
def total_animals : ℕ := num_dogs + num_cats

/-- Represents the number of ways to pair dogs and cats -/
def num_pairings : ℕ := num_dogs * num_cats

theorem animal_pairing_theorem :
  num_pairings = 20 ∧
  total_animals = num_bowls + 2 :=
sorry

end NUMINAMATH_CALUDE_animal_pairing_theorem_l3957_395726


namespace NUMINAMATH_CALUDE_parabola_through_circle_center_l3957_395706

/-- Represents a circle in the 2D plane --/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- Represents a parabola in the 2D plane --/
structure Parabola where
  equation : ℝ → ℝ → Prop
  vertex : ℝ × ℝ
  axis_of_symmetry : ℝ → ℝ → Prop

/-- The given circle --/
def given_circle : Circle :=
  { equation := λ x y => x^2 + y^2 - 2*x + 6*y + 9 = 0 }

/-- Theorem stating the properties of the parabola passing through the circle's center --/
theorem parabola_through_circle_center (p : Parabola) :
  p.vertex = (0, 0) →
  (p.axis_of_symmetry = λ x y => x = 0 ∨ y = 0) →
  (∃ x y, given_circle.equation x y ∧ p.equation x y) →
  (∀ x y, p.equation x y ↔ (y = -3*x^2 ∨ y^2 = 9*x)) :=
sorry

end NUMINAMATH_CALUDE_parabola_through_circle_center_l3957_395706
