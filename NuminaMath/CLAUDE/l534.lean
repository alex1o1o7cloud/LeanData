import Mathlib

namespace NUMINAMATH_CALUDE_positive_interval_for_quadratic_l534_53465

theorem positive_interval_for_quadratic (x : ℝ) :
  (x + 1) * (x - 3) > 0 ↔ x < -1 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_interval_for_quadratic_l534_53465


namespace NUMINAMATH_CALUDE_repair_cost_calculation_l534_53401

def purchase_price : ℚ := 14000
def transportation_charges : ℚ := 1000
def selling_price : ℚ := 30000
def profit_percentage : ℚ := 50

theorem repair_cost_calculation (repair_cost : ℚ) :
  (purchase_price + repair_cost + transportation_charges) * (1 + profit_percentage / 100) = selling_price →
  repair_cost = 5000 := by
  sorry

end NUMINAMATH_CALUDE_repair_cost_calculation_l534_53401


namespace NUMINAMATH_CALUDE_rectangle_area_l534_53448

/-- The area of a rectangle with length 4 cm and width 2 cm is 8 cm² -/
theorem rectangle_area : 
  ∀ (length width area : ℝ),
  length = 4 →
  width = 2 →
  area = length * width →
  area = 8 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l534_53448


namespace NUMINAMATH_CALUDE_magician_earnings_calculation_l534_53447

/-- The amount of money earned by a magician selling card decks -/
def magician_earnings (price_per_deck : ℕ) (initial_decks : ℕ) (final_decks : ℕ) : ℕ :=
  (initial_decks - final_decks) * price_per_deck

/-- Theorem: The magician earns $56 -/
theorem magician_earnings_calculation :
  magician_earnings 7 16 8 = 56 := by
  sorry

end NUMINAMATH_CALUDE_magician_earnings_calculation_l534_53447


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l534_53476

theorem cubic_equation_solution : ∃! (x : ℕ), x^3 = 2011^2 + 2011 * 2012 + 2012^2 + 2011^3 :=
  by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l534_53476


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_in_specific_pyramid_l534_53458

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid where
  base_side : ℝ
  lateral_face_is_equilateral : Bool

/-- A cube inscribed in a pyramid -/
structure InscribedCube where
  pyramid : Pyramid
  bottom_face_on_base : Bool
  top_face_edges_on_lateral_faces : Bool

/-- The volume of an inscribed cube -/
noncomputable def inscribed_cube_volume (cube : InscribedCube) : ℝ :=
  sorry

theorem inscribed_cube_volume_in_specific_pyramid :
  ∀ (cube : InscribedCube),
    cube.pyramid.base_side = 2 ∧
    cube.pyramid.lateral_face_is_equilateral = true ∧
    cube.bottom_face_on_base = true ∧
    cube.top_face_edges_on_lateral_faces = true →
    inscribed_cube_volume cube = 2 * Real.sqrt 6 / 9 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_in_specific_pyramid_l534_53458


namespace NUMINAMATH_CALUDE_lynne_total_spent_l534_53471

/-- The total amount Lynne spent on books and magazines -/
def total_spent (cat_books : ℕ) (solar_books : ℕ) (magazines : ℕ) (book_cost : ℕ) (magazine_cost : ℕ) : ℕ :=
  (cat_books + solar_books) * book_cost + magazines * magazine_cost

/-- Theorem stating that Lynne spent $75 in total -/
theorem lynne_total_spent :
  total_spent 7 2 3 7 4 = 75 := by
  sorry

#eval total_spent 7 2 3 7 4

end NUMINAMATH_CALUDE_lynne_total_spent_l534_53471


namespace NUMINAMATH_CALUDE_tan_alpha_value_l534_53472

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2)) 
  (h2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l534_53472


namespace NUMINAMATH_CALUDE_probability_intersection_is_zero_l534_53429

def f (x : Nat) : Int :=
  6 * x - 4

def g (x : Nat) : Int :=
  2 * x - 1

def domain : Finset Nat :=
  {1, 2, 3, 4, 5, 6}

def A : Finset Int :=
  Finset.image f domain

def B : Finset Int :=
  Finset.image g domain

theorem probability_intersection_is_zero :
  (A ∩ B).card / (A ∪ B).card = 0 :=
sorry

end NUMINAMATH_CALUDE_probability_intersection_is_zero_l534_53429


namespace NUMINAMATH_CALUDE_inequality_solution_set_l534_53485

def f (x : ℝ) := abs x + abs (x - 4)

theorem inequality_solution_set :
  {x : ℝ | f (x^2 + 2) > f x} = {x | x < -2 ∨ x > Real.sqrt 2} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l534_53485


namespace NUMINAMATH_CALUDE_stamp_reorganization_l534_53407

/-- Represents the stamp reorganization problem --/
theorem stamp_reorganization (
  initial_books : Nat)
  (pages_per_book : Nat)
  (initial_stamps_per_page : Nat)
  (new_stamps_per_page : Nat)
  (filled_books : Nat)
  (filled_pages_in_last_book : Nat)
  (h1 : initial_books = 10)
  (h2 : pages_per_book = 36)
  (h3 : initial_stamps_per_page = 5)
  (h4 : new_stamps_per_page = 8)
  (h5 : filled_books = 7)
  (h6 : filled_pages_in_last_book = 28) :
  (initial_books * pages_per_book * initial_stamps_per_page) -
  (filled_books * pages_per_book + filled_pages_in_last_book) * new_stamps_per_page = 8 := by
  sorry

#check stamp_reorganization

end NUMINAMATH_CALUDE_stamp_reorganization_l534_53407


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l534_53464

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  (∀ n : ℕ, b (n + 1) > b n) →
  (∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d) →
  b 5 * b 6 = 14 →
  b 4 * b 7 = -324 ∨ b 4 * b 7 = -36 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l534_53464


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l534_53425

theorem imaginary_part_of_complex_fraction :
  let i : ℂ := Complex.I
  let z : ℂ := (3 - 4*i) / (1 + i)
  Complex.im z = -7/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l534_53425


namespace NUMINAMATH_CALUDE_fourth_circle_radius_l534_53493

/-- Represents a configuration of seven consecutively tangent circles between two parallel lines -/
structure CircleConfiguration where
  radii : Fin 7 → ℝ
  largest_radius : radii 6 = 24
  smallest_radius : radii 0 = 6
  tangent : ∀ i : Fin 6, radii i < radii (i.succ)

/-- The theorem stating that the radius of the fourth circle is 12√2 -/
theorem fourth_circle_radius (config : CircleConfiguration) : config.radii 3 = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_circle_radius_l534_53493


namespace NUMINAMATH_CALUDE_mobile_phone_sales_growth_l534_53413

/-- Represents the sales growth of mobile phones over two months -/
theorem mobile_phone_sales_growth 
  (initial_sales : ℝ) 
  (final_sales : ℝ) 
  (monthly_growth_rate : ℝ) 
  (h1 : initial_sales = 400) 
  (h2 : final_sales = 900) :
  initial_sales * (1 + monthly_growth_rate)^2 = final_sales := by
  sorry

end NUMINAMATH_CALUDE_mobile_phone_sales_growth_l534_53413


namespace NUMINAMATH_CALUDE_inequality_solution_l534_53499

theorem inequality_solution : 
  {x : ℝ | (x - 2)^2 < 3*x + 4} = {x : ℝ | 0 ≤ x ∧ x < 7} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l534_53499


namespace NUMINAMATH_CALUDE_circle_radius_from_polar_equation_l534_53419

/-- Given a circle with polar equation ρ² - 2ρcosθ + 4ρsinθ + 4 = 0, its radius is 1 -/
theorem circle_radius_from_polar_equation :
  ∀ ρ θ : ℝ,
  ρ^2 - 2*ρ*(Real.cos θ) + 4*ρ*(Real.sin θ) + 4 = 0 →
  ∃ x y : ℝ,
  (x - 1)^2 + (y + 2)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_polar_equation_l534_53419


namespace NUMINAMATH_CALUDE_mrs_hilt_picture_frame_perimeter_l534_53479

/-- The perimeter of a rectangular picture frame. -/
def perimeter_picture_frame (height length : ℕ) : ℕ :=
  2 * (height + length)

/-- Theorem: The perimeter of Mrs. Hilt's picture frame is 44 inches. -/
theorem mrs_hilt_picture_frame_perimeter :
  perimeter_picture_frame 12 10 = 44 := by
  sorry

#eval perimeter_picture_frame 12 10

end NUMINAMATH_CALUDE_mrs_hilt_picture_frame_perimeter_l534_53479


namespace NUMINAMATH_CALUDE_marbles_after_sharing_undetermined_l534_53409

/-- Represents the items Carolyn has -/
structure CarolynItems where
  marbles : ℕ
  oranges : ℕ

/-- Represents the sharing action -/
def share (items : CarolynItems) (shared : ℕ) : Prop :=
  shared ≤ items.marbles + items.oranges

/-- Theorem stating that the number of marbles Carolyn ends up with is undetermined -/
theorem marbles_after_sharing_undetermined 
  (initial : CarolynItems) 
  (shared : ℕ) 
  (h1 : initial.marbles = 47)
  (h2 : initial.oranges = 6)
  (h3 : share initial shared)
  (h4 : shared = 42) :
  ∃ (final : CarolynItems), final.marbles ≤ initial.marbles ∧ 
    final.marbles + final.oranges = initial.marbles + initial.oranges - shared :=
sorry

end NUMINAMATH_CALUDE_marbles_after_sharing_undetermined_l534_53409


namespace NUMINAMATH_CALUDE_square_sum_zero_iff_both_zero_l534_53452

theorem square_sum_zero_iff_both_zero (x y : ℝ) : x^2 + y^2 = 0 ↔ x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_iff_both_zero_l534_53452


namespace NUMINAMATH_CALUDE_negative_two_squared_minus_zero_power_six_m_divided_by_two_m_l534_53498

-- First problem
theorem negative_two_squared_minus_zero_power : ((-2 : ℤ)^2) - ((-2 : ℤ)^0) = 3 := by sorry

-- Second problem
theorem six_m_divided_by_two_m (m : ℝ) (hm : m ≠ 0) : (6 * m) / (2 * m) = 3 := by sorry

end NUMINAMATH_CALUDE_negative_two_squared_minus_zero_power_six_m_divided_by_two_m_l534_53498


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_20_l534_53459

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem largest_four_digit_sum_20 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 20 → n ≤ 9920 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_20_l534_53459


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l534_53441

theorem geometric_series_ratio (a : ℝ) (S : ℝ) (r : ℝ) 
  (h1 : a = 520) 
  (h2 : S = 3250) 
  (h3 : S = a / (1 - r)) 
  (h4 : 0 ≤ r) 
  (h5 : r < 1) : 
  r = 273 / 325 := by sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l534_53441


namespace NUMINAMATH_CALUDE_roots_cubed_equation_reciprocal_squares_equation_sum_and_reciprocal_equation_quotient_roots_equation_l534_53416

-- Define the quadratic equation and its roots
variable (p q : ℝ)
variable (x₁ x₂ : ℝ)

-- Define the original equation
def original_eq (x : ℝ) : Prop := x^2 + p*x + q = 0

-- Define that x₁ and x₂ are roots of the original equation
axiom root_x₁ : original_eq p q x₁
axiom root_x₂ : original_eq p q x₂

-- Part a
theorem roots_cubed_equation :
  ∀ y, y^2 + (p^3 - 3*p*q)*y + q^3 = 0 ↔ (y = x₁^3 ∨ y = x₂^3) :=
sorry

-- Part b
theorem reciprocal_squares_equation :
  ∀ y, q^2*y^2 + (2*q - p^2)*y + 1 = 0 ↔ (y = 1/x₁^2 ∨ y = 1/x₂^2) :=
sorry

-- Part c
theorem sum_and_reciprocal_equation :
  ∀ y, q*y^2 + p*(q + 1)*y + (q + 1)^2 = 0 ↔ (y = x₁ + 1/x₂ ∨ y = x₂ + 1/x₁) :=
sorry

-- Part d
theorem quotient_roots_equation :
  ∀ y, q*y^2 + (2*q - p^2)*y + q = 0 ↔ (y = x₂/x₁ ∨ y = x₁/x₂) :=
sorry

end NUMINAMATH_CALUDE_roots_cubed_equation_reciprocal_squares_equation_sum_and_reciprocal_equation_quotient_roots_equation_l534_53416


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l534_53481

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x > 1 → |x| > 1) ∧ 
  (∃ x : ℝ, |x| > 1 ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l534_53481


namespace NUMINAMATH_CALUDE_wand_cost_is_60_l534_53412

/-- The cost of a magic wand at Wizards Park -/
def wand_cost : ℕ → Prop := λ x =>
  -- Kate buys 3 wands and sells 2 of them
  -- She sells each wand for $5 more than she paid
  -- She collected $130 after the sale
  2 * (x + 5) = 130

/-- The cost of each wand is $60 -/
theorem wand_cost_is_60 : wand_cost 60 := by sorry

end NUMINAMATH_CALUDE_wand_cost_is_60_l534_53412


namespace NUMINAMATH_CALUDE_range_of_k_l534_53439

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the non-collinearity of e₁ and e₂
variable (h_non_collinear : ¬ ∃ (c : ℝ), e₁ = c • e₂)

-- Define vectors a and b
def a : V := 2 • e₁ + e₂
def b (k : ℝ) : V := k • e₁ + 3 • e₂

-- Define the condition that a and b form a basis
variable (h_basis : ∀ (k : ℝ), k ≠ 6 → LinearIndependent ℝ ![a, b k])

-- Theorem statement
theorem range_of_k : 
  {k : ℝ | k ≠ 6} = {k : ℝ | LinearIndependent ℝ ![a, b k]} :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l534_53439


namespace NUMINAMATH_CALUDE_consecutive_integer_product_divisible_by_six_l534_53444

theorem consecutive_integer_product_divisible_by_six (n : ℤ) : 
  ∃ k : ℤ, n * (n + 1) * (n + 2) = 6 * k := by
  sorry

#check consecutive_integer_product_divisible_by_six

end NUMINAMATH_CALUDE_consecutive_integer_product_divisible_by_six_l534_53444


namespace NUMINAMATH_CALUDE_lions_scored_18_l534_53403

-- Define the total score and winning margin
def total_score : ℕ := 52
def winning_margin : ℕ := 16

-- Define the Lions' score as a function of the total score and winning margin
def lions_score (total : ℕ) (margin : ℕ) : ℕ :=
  (total - margin) / 2

-- Theorem statement
theorem lions_scored_18 :
  lions_score total_score winning_margin = 18 := by
  sorry

end NUMINAMATH_CALUDE_lions_scored_18_l534_53403


namespace NUMINAMATH_CALUDE_right_angled_triangle_l534_53474

theorem right_angled_triangle (A B C : Real) (h : A + B + C = Real.pi) 
  (eq : (Real.sin A)^2 + (Real.sin B)^2 + (Real.sin C)^2 = 
        2 * ((Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2)) : 
  A = Real.pi/2 ∨ B = Real.pi/2 ∨ C = Real.pi/2 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l534_53474


namespace NUMINAMATH_CALUDE_expression_equality_l534_53430

theorem expression_equality : 
  (12^4 + 324) * (24^4 + 324) * (36^4 + 324) / 
  ((6^4 + 324) * (18^4 + 324) * (30^4 + 324)) = 84/35 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l534_53430


namespace NUMINAMATH_CALUDE_largest_fraction_l534_53438

theorem largest_fraction :
  (202 : ℚ) / 403 > 5 / 11 ∧
  (202 : ℚ) / 403 > 7 / 16 ∧
  (202 : ℚ) / 403 > 23 / 50 ∧
  (202 : ℚ) / 403 > 99 / 200 :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_l534_53438


namespace NUMINAMATH_CALUDE_unique_solution_condition_l534_53492

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 3) ↔ d ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l534_53492


namespace NUMINAMATH_CALUDE_fourth_side_length_l534_53435

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The lengths of the four sides -/
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  /-- The quadrilateral is inscribed in the circle -/
  inscribed : True
  /-- The quadrilateral is not a rectangle -/
  not_rectangle : True

/-- Theorem: In a quadrilateral inscribed in a circle with radius 150√2,
    if three sides have length 150, then the fourth side has length 300√2 -/
theorem fourth_side_length (q : InscribedQuadrilateral)
    (h_radius : q.radius = 150 * Real.sqrt 2)
    (h_side1 : q.side1 = 150)
    (h_side2 : q.side2 = 150)
    (h_side3 : q.side3 = 150) :
    q.side4 = 300 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_length_l534_53435


namespace NUMINAMATH_CALUDE_sam_book_purchase_l534_53466

theorem sam_book_purchase (initial_amount : ℕ) (book_cost : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 79)
  (h2 : book_cost = 7)
  (h3 : remaining_amount = 16) :
  (initial_amount - remaining_amount) / book_cost = 9 := by
  sorry

end NUMINAMATH_CALUDE_sam_book_purchase_l534_53466


namespace NUMINAMATH_CALUDE_lemonade_proportion_l534_53404

/-- Given that 40 lemons make 50 gallons of lemonade, prove that 12 lemons make 15 gallons -/
theorem lemonade_proportion :
  let lemons_for_50 : ℚ := 40
  let gallons_50 : ℚ := 50
  let gallons_15 : ℚ := 15
  let lemons_for_15 : ℚ := 12
  (lemons_for_50 / gallons_50 = lemons_for_15 / gallons_15) := by sorry

end NUMINAMATH_CALUDE_lemonade_proportion_l534_53404


namespace NUMINAMATH_CALUDE_last_digit_of_special_number_l534_53417

/-- A function that returns the last element of a list -/
def lastDigit (digits : List Nat) : Nat :=
  match digits.reverse with
  | [] => 0  -- Default value for empty list
  | d :: _ => d

/-- Check if a two-digit number is divisible by 13 -/
def isDivisibleBy13 (n : Nat) : Prop :=
  n % 13 = 0

theorem last_digit_of_special_number :
  ∀ (digits : List Nat),
    digits.length = 2019 →
    digits.head? = some 6 →
    (∀ i, i < digits.length - 1 →
      isDivisibleBy13 (digits[i]! * 10 + digits[i+1]!)) →
    lastDigit digits = 2 := by
  sorry

#check last_digit_of_special_number

end NUMINAMATH_CALUDE_last_digit_of_special_number_l534_53417


namespace NUMINAMATH_CALUDE_largest_choir_size_l534_53405

theorem largest_choir_size :
  ∃ (x r m : ℕ),
    (r * x + 3 = m) ∧
    ((r - 3) * (x + 2) = m) ∧
    (m < 150) ∧
    (∀ (x' r' m' : ℕ),
      (r' * x' + 3 = m') ∧
      ((r' - 3) * (x' + 2) = m') ∧
      (m' < 150) →
      m' ≤ m) ∧
    m = 759 :=
by sorry

end NUMINAMATH_CALUDE_largest_choir_size_l534_53405


namespace NUMINAMATH_CALUDE_students_not_taking_languages_l534_53411

theorem students_not_taking_languages (total : ℕ) (french : ℕ) (spanish : ℕ) (both : ℕ) 
  (h1 : total = 28)
  (h2 : french = 5)
  (h3 : spanish = 10)
  (h4 : both = 4) :
  total - (french + spanish - both) = 17 := by
  sorry

#check students_not_taking_languages

end NUMINAMATH_CALUDE_students_not_taking_languages_l534_53411


namespace NUMINAMATH_CALUDE_sum_specific_sequence_l534_53483

theorem sum_specific_sequence : 
  (1000 + 20 + 1000 + 30 + 1000 + 40 + 1000 + 10) = 4100 := by
  sorry

end NUMINAMATH_CALUDE_sum_specific_sequence_l534_53483


namespace NUMINAMATH_CALUDE_min_sum_squared_distances_min_sum_squared_distances_achievable_l534_53453

/-- The minimum sum of squared distances from a point on a circle to two fixed points -/
theorem min_sum_squared_distances (x y : ℝ) :
  (x - 3)^2 + (y - 4)^2 = 4 →
  (x + 2)^2 + y^2 + (x - 2)^2 + y^2 ≥ 26 := by
  sorry

/-- The minimum sum of squared distances is achievable -/
theorem min_sum_squared_distances_achievable :
  ∃ x y : ℝ, (x - 3)^2 + (y - 4)^2 = 4 ∧
  (x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squared_distances_min_sum_squared_distances_achievable_l534_53453


namespace NUMINAMATH_CALUDE_sequence_general_term_l534_53497

theorem sequence_general_term 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h : ∀ n, S n = 2 * a n - 3) : 
  ∀ n, a n = 3 * 2^(n - 1) := by
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l534_53497


namespace NUMINAMATH_CALUDE_exists_prime_triplet_l534_53406

/-- A structure representing a prime triplet (a, b, c) -/
structure PrimeTriplet where
  a : Nat
  b : Nat
  c : Nat
  h_prime_a : Nat.Prime a
  h_prime_b : Nat.Prime b
  h_prime_c : Nat.Prime c
  h_order : a < b ∧ b < c ∧ c < 100
  h_geometric : (b + 1)^2 = (a + 1) * (c + 1)

/-- Theorem stating the existence of prime triplets satisfying the given conditions -/
theorem exists_prime_triplet : ∃ t : PrimeTriplet, True := by
  sorry

end NUMINAMATH_CALUDE_exists_prime_triplet_l534_53406


namespace NUMINAMATH_CALUDE_smallest_n_divisible_sixty_satisfies_smallest_n_is_sixty_l534_53455

theorem smallest_n_divisible (n : ℕ) : n > 0 ∧ 24 ∣ n^2 ∧ 450 ∣ n^3 → n ≥ 60 :=
by sorry

theorem sixty_satisfies : 24 ∣ 60^2 ∧ 450 ∣ 60^3 :=
by sorry

theorem smallest_n_is_sixty : ∃! n : ℕ, n > 0 ∧ 24 ∣ n^2 ∧ 450 ∣ n^3 ∧ ∀ m : ℕ, (m > 0 ∧ 24 ∣ m^2 ∧ 450 ∣ m^3) → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_sixty_satisfies_smallest_n_is_sixty_l534_53455


namespace NUMINAMATH_CALUDE_unknown_number_solution_l534_53443

theorem unknown_number_solution (x : ℝ) : 
  4.7 * 13.26 + 4.7 * 9.43 + 4.7 * x = 470 ↔ x = 77.31 := by
sorry

end NUMINAMATH_CALUDE_unknown_number_solution_l534_53443


namespace NUMINAMATH_CALUDE_doughnut_boxes_l534_53446

theorem doughnut_boxes (total_doughnuts : ℕ) (doughnuts_per_box : ℕ) (h1 : total_doughnuts = 48) (h2 : doughnuts_per_box = 12) :
  total_doughnuts / doughnuts_per_box = 4 := by
  sorry

end NUMINAMATH_CALUDE_doughnut_boxes_l534_53446


namespace NUMINAMATH_CALUDE_vector_linear_combination_l534_53437

/-- Given vectors a, b, and c in ℝ², prove that c is a linear combination of a and b -/
theorem vector_linear_combination (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) 
  (hb : b = (1, -1)) 
  (hc : c = (-1, 2)) : 
  ∃ (k l : ℝ), c = k • a + l • b ∧ k = (1/2 : ℝ) ∧ l = (-3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vector_linear_combination_l534_53437


namespace NUMINAMATH_CALUDE_probability_divisible_by_3_l534_53402

/-- The set of digits to choose from -/
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}

/-- A four-digit number formed from the given set of digits -/
structure FourDigitNumber where
  d₁ : ℕ
  d₂ : ℕ
  d₃ : ℕ
  d₄ : ℕ
  h₁ : d₁ ∈ digits
  h₂ : d₂ ∈ digits
  h₃ : d₃ ∈ digits
  h₄ : d₄ ∈ digits
  h₅ : d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄
  h₆ : d₁ ≠ 0

/-- The value of a four-digit number -/
def FourDigitNumber.value (n : FourDigitNumber) : ℕ :=
  1000 * n.d₁ + 100 * n.d₂ + 10 * n.d₃ + n.d₄

/-- A number is divisible by 3 if the sum of its digits is divisible by 3 -/
def FourDigitNumber.divisibleBy3 (n : FourDigitNumber) : Prop :=
  (n.d₁ + n.d₂ + n.d₃ + n.d₄) % 3 = 0

/-- The set of all valid four-digit numbers -/
def allFourDigitNumbers : Finset FourDigitNumber :=
  sorry

/-- The set of all four-digit numbers divisible by 3 -/
def divisibleBy3Numbers : Finset FourDigitNumber :=
  sorry

/-- The main theorem -/
theorem probability_divisible_by_3 :
  (Finset.card divisibleBy3Numbers : ℚ) / (Finset.card allFourDigitNumbers) = 8 / 15 :=
sorry

end NUMINAMATH_CALUDE_probability_divisible_by_3_l534_53402


namespace NUMINAMATH_CALUDE_ratio_equality_l534_53421

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_abc : a^2 + b^2 + c^2 = 49)
  (sum_xyz : x^2 + y^2 + z^2 = 64)
  (dot_product : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l534_53421


namespace NUMINAMATH_CALUDE_coeff_x_cubed_eq_60_l534_53426

/-- The coefficient of x^3 in the expansion of x(1+2x)^6 -/
def coeff_x_cubed : ℕ :=
  (Finset.range 7).sum (fun k => k.choose 6 * 2^k * if k = 2 then 1 else 0)

/-- Theorem stating that the coefficient of x^3 in x(1+2x)^6 is 60 -/
theorem coeff_x_cubed_eq_60 : coeff_x_cubed = 60 := by
  sorry

end NUMINAMATH_CALUDE_coeff_x_cubed_eq_60_l534_53426


namespace NUMINAMATH_CALUDE_fraction_addition_l534_53460

theorem fraction_addition : (3/4) / (5/8) + 1/2 = 17/10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l534_53460


namespace NUMINAMATH_CALUDE_calculate_expression_l534_53414

theorem calculate_expression : 9^6 * 3^3 / 27^4 = 27 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l534_53414


namespace NUMINAMATH_CALUDE_acute_triangle_existence_l534_53408

/-- Given n positive real numbers satisfying the max-min relation,
    there exist three that form an acute triangle when n ≥ 13 -/
theorem acute_triangle_existence (n : ℕ) (h : n ≥ 13) :
  ∀ (a : Fin n → ℝ),
  (∀ i, a i > 0) →
  (∀ i j, a i ≤ n * a j) →
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    a i ^ 2 + a j ^ 2 > a k ^ 2 ∧
    a i ^ 2 + a k ^ 2 > a j ^ 2 ∧
    a j ^ 2 + a k ^ 2 > a i ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_existence_l534_53408


namespace NUMINAMATH_CALUDE_perfume_bottle_size_l534_53440

/-- Given the following conditions:
    1. 320 rose petals make an ounce of perfume
    2. Each rose produces 8 petals
    3. There are 12 roses per bush
    4. Fern harvests 800 bushes
    5. Fern makes 20 bottles of perfume
    Prove that the size of each bottle of perfume is 12 ounces. -/
theorem perfume_bottle_size 
  (petals_per_ounce : ℕ) 
  (petals_per_rose : ℕ) 
  (roses_per_bush : ℕ) 
  (harvested_bushes : ℕ) 
  (bottles_produced : ℕ) 
  (h1 : petals_per_ounce = 320) 
  (h2 : petals_per_rose = 8) 
  (h3 : roses_per_bush = 12) 
  (h4 : harvested_bushes = 800) 
  (h5 : bottles_produced = 20) :
  (harvested_bushes * roses_per_bush * petals_per_rose) / petals_per_ounce / bottles_produced = 12 := by
  sorry

end NUMINAMATH_CALUDE_perfume_bottle_size_l534_53440


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l534_53410

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_children_per_family : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : average_children_per_family = 3)
  (h3 : childless_families = 3) :
  (total_families * average_children_per_family) / (total_families - childless_families) = 3.75 := by
sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l534_53410


namespace NUMINAMATH_CALUDE_min_value_interval_l534_53470

def f (x : ℝ) := 3 * x - x^3

theorem min_value_interval (a : ℝ) :
  (∃ x ∈ Set.Ioo (a^2 - 12) a, ∀ y ∈ Set.Ioo (a^2 - 12) a, f y ≥ f x) →
  a ∈ Set.Ioo (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_interval_l534_53470


namespace NUMINAMATH_CALUDE_stewart_farm_ratio_l534_53456

theorem stewart_farm_ratio : 
  ∀ (horse_food_per_day : ℕ) (total_horse_food : ℕ) (num_sheep : ℕ),
    horse_food_per_day = 230 →
    total_horse_food = 12880 →
    num_sheep = 32 →
    let num_horses := total_horse_food / horse_food_per_day
    (num_sheep : ℚ) / num_horses = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_ratio_l534_53456


namespace NUMINAMATH_CALUDE_solution_l534_53478

def problem (x : ℝ) (number : ℝ) : Prop :=
  number = 3639 + 11.95 - x

theorem solution (x : ℝ) (number : ℝ) 
  (h1 : problem x number) 
  (h2 : x = 596.95) : 
  number = 3054 := by
  sorry

end NUMINAMATH_CALUDE_solution_l534_53478


namespace NUMINAMATH_CALUDE_line_mb_value_l534_53496

/-- Given a line passing through points (0, -1) and (1, 1) with equation y = mx + b, prove that mb = -2 -/
theorem line_mb_value (m b : ℝ) : 
  (0 : ℝ) = m * 0 + b → -- The line passes through (0, -1)
  (1 : ℝ) = m * 1 + b → -- The line passes through (1, 1)
  m * b = -2 := by
sorry

end NUMINAMATH_CALUDE_line_mb_value_l534_53496


namespace NUMINAMATH_CALUDE_winning_configurations_l534_53434

/-- Represents a wall configuration in the brick removal game -/
structure WallConfig :=
  (walls : List Nat)

/-- Calculates the nim-value of a single wall -/
def nimValue (wall : Nat) : Nat :=
  sorry

/-- Calculates the nim-sum of a list of nim-values -/
def nimSum (values : List Nat) : Nat :=
  sorry

/-- Determines if a configuration is a winning position for the second player -/
def isWinningForSecondPlayer (config : WallConfig) : Prop :=
  nimSum (config.walls.map nimValue) = 0

/-- The list of all possible starting configurations -/
def startingConfigs : List WallConfig :=
  [⟨[7, 3, 2]⟩, ⟨[7, 4, 1]⟩, ⟨[8, 3, 1]⟩, ⟨[7, 2, 2]⟩, ⟨[7, 3, 3]⟩]

/-- The main theorem to be proved -/
theorem winning_configurations :
  (∀ c ∈ startingConfigs, isWinningForSecondPlayer c ↔ (c = ⟨[7, 3, 2]⟩ ∨ c = ⟨[8, 3, 1]⟩)) :=
  sorry

end NUMINAMATH_CALUDE_winning_configurations_l534_53434


namespace NUMINAMATH_CALUDE_absolute_value_of_S_eq_121380_l534_53488

/-- The sum of all integers b for which x^2 + bx + 2023b can be factored over the integers -/
def S : ℤ := sorry

/-- The polynomial x^2 + bx + 2023b -/
def polynomial (x b : ℤ) : ℤ := x^2 + b*x + 2023*b

/-- Predicate to check if a polynomial can be factored over the integers -/
def is_factorable (b : ℤ) : Prop := ∃ (p q : ℤ → ℤ), ∀ x, polynomial x b = p x * q x

theorem absolute_value_of_S_eq_121380 : |S| = 121380 := by sorry

end NUMINAMATH_CALUDE_absolute_value_of_S_eq_121380_l534_53488


namespace NUMINAMATH_CALUDE_number_satisfies_equation_l534_53487

theorem number_satisfies_equation : ∃ (n : ℕ), n = 14 ∧ 2^n - 2^(n-2) = 3 * 2^12 :=
by
  sorry

end NUMINAMATH_CALUDE_number_satisfies_equation_l534_53487


namespace NUMINAMATH_CALUDE_cost_of_seven_cds_cost_of_seven_cds_is_112_l534_53428

/-- The cost of seven CDs given that two identical CDs cost $32 -/
theorem cost_of_seven_cds : ℝ :=
  let cost_of_two : ℝ := 32
  let cost_of_one : ℝ := cost_of_two / 2
  7 * cost_of_one

/-- Proof that the cost of seven CDs is $112 -/
theorem cost_of_seven_cds_is_112 : cost_of_seven_cds = 112 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_seven_cds_cost_of_seven_cds_is_112_l534_53428


namespace NUMINAMATH_CALUDE_pen_pencil_cost_l534_53475

/-- Given a pen and pencil where the pen costs twice as much as the pencil and the pen costs $4,
    prove that the total cost of the pen and pencil is $6. -/
theorem pen_pencil_cost (pen_cost pencil_cost : ℝ) : 
  pen_cost = 4 → pen_cost = 2 * pencil_cost → pen_cost + pencil_cost = 6 := by
sorry

end NUMINAMATH_CALUDE_pen_pencil_cost_l534_53475


namespace NUMINAMATH_CALUDE_inequality_proof_l534_53415

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z > 0) :
  (x + y + z) * (x + y - z) * (x - y + z) / (x * y * z) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l534_53415


namespace NUMINAMATH_CALUDE_white_dandelions_on_saturday_l534_53463

/-- Represents the state of a dandelion -/
inductive DandelionState
  | Yellow
  | White
  | Dispersed

/-- Represents a day of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents the lifecycle of a dandelion -/
def dandelionLifecycle (openDay : Day) (currentDay : Day) : DandelionState :=
  sorry

/-- Counts the number of dandelions in a specific state on a given day -/
def countDandelions (day : Day) (state : DandelionState) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem white_dandelions_on_saturday :
  (countDandelions Day.Monday DandelionState.Yellow = 20) →
  (countDandelions Day.Monday DandelionState.White = 14) →
  (countDandelions Day.Wednesday DandelionState.Yellow = 15) →
  (countDandelions Day.Wednesday DandelionState.White = 11) →
  (countDandelions Day.Saturday DandelionState.White = 6) :=
by sorry

end NUMINAMATH_CALUDE_white_dandelions_on_saturday_l534_53463


namespace NUMINAMATH_CALUDE_alex_movie_count_l534_53445

theorem alex_movie_count (total_different_movies : ℕ) 
  (movies_watched_together : ℕ) 
  (dalton_movies : ℕ) 
  (hunter_movies : ℕ) 
  (h1 : total_different_movies = 30)
  (h2 : movies_watched_together = 2)
  (h3 : dalton_movies = 7)
  (h4 : hunter_movies = 12) :
  total_different_movies - movies_watched_together - dalton_movies - hunter_movies = 9 := by
  sorry

end NUMINAMATH_CALUDE_alex_movie_count_l534_53445


namespace NUMINAMATH_CALUDE_min_value_cubic_function_l534_53454

theorem min_value_cubic_function (x : ℝ) (h : x > 0) :
  x^3 + 9*x + 81/x^4 ≥ 21 ∧ ∃ y > 0, y^3 + 9*y + 81/y^4 = 21 :=
sorry

end NUMINAMATH_CALUDE_min_value_cubic_function_l534_53454


namespace NUMINAMATH_CALUDE_maltese_cross_to_square_l534_53480

/-- Represents a piece of the Maltese cross -/
structure Piece where
  area : ℝ

/-- Represents the Maltese cross -/
structure MalteseCross where
  pieces : Finset Piece
  total_area : ℝ

/-- Represents a square -/
structure Square where
  side_length : ℝ

/-- A function that checks if a set of pieces can form a square -/
def can_form_square (pieces : Finset Piece) : Prop :=
  ∃ (s : Square), s.side_length^2 = (pieces.sum (λ p => p.area))

theorem maltese_cross_to_square (cross : MalteseCross) : 
  cross.total_area = 17 → 
  (∃ (cut_pieces : Finset Piece), 
    cut_pieces.card = 7 ∧ 
    (cut_pieces.sum (λ p => p.area) = cross.total_area) ∧
    can_form_square cut_pieces) := by
  sorry

end NUMINAMATH_CALUDE_maltese_cross_to_square_l534_53480


namespace NUMINAMATH_CALUDE_total_shells_is_245_l534_53473

/-- The number of shells each person has -/
structure Shells :=
  (david : ℕ)
  (mia : ℕ)
  (ava : ℕ)
  (alice : ℕ)
  (liam : ℕ)

/-- The conditions of the problem -/
def shellConditions (s : Shells) : Prop :=
  s.david = 15 ∧
  s.mia = 4 * s.david ∧
  s.ava = s.mia + 20 ∧
  s.alice = s.ava / 2 ∧
  s.liam = 2 * (s.alice - s.david)

/-- The total number of shells -/
def totalShells (s : Shells) : ℕ :=
  s.david + s.mia + s.ava + s.alice + s.liam

/-- Theorem: Given the conditions, the total number of shells is 245 -/
theorem total_shells_is_245 (s : Shells) (h : shellConditions s) : totalShells s = 245 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_is_245_l534_53473


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l534_53468

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given condition
  2 * a * Real.sin B - Real.sqrt 5 * b * Real.cos A = 0 →
  -- Theorem 1: cos A = 2/3
  Real.cos A = 2/3 ∧
  -- Theorem 2: If a = √5 and b = 2, area = √5
  (a = Real.sqrt 5 ∧ b = 2 → 
    (1/2) * a * b * Real.sin C = Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l534_53468


namespace NUMINAMATH_CALUDE_product_minus_sum_of_first_45_primes_l534_53489

def first_n_primes (n : ℕ) : List ℕ :=
  (List.range 1000).filter Nat.Prime |> List.take n

theorem product_minus_sum_of_first_45_primes :
  ∃ x : ℕ, (List.prod (first_n_primes 45) - List.sum (first_n_primes 45) = x) :=
by
  sorry

end NUMINAMATH_CALUDE_product_minus_sum_of_first_45_primes_l534_53489


namespace NUMINAMATH_CALUDE_trig_identity_l534_53449

theorem trig_identity (x : ℝ) : 
  (Real.sin x ^ 6 + Real.cos x ^ 6 - 1) ^ 3 + 27 * Real.sin x ^ 6 * Real.cos x ^ 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l534_53449


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l534_53482

/-- Calculates the discount percentage given item costs and final amount spent -/
theorem discount_percentage_calculation 
  (hand_mitts_cost apron_cost utensils_cost final_amount : ℚ)
  (nieces : ℕ)
  (h1 : hand_mitts_cost = 14)
  (h2 : apron_cost = 16)
  (h3 : utensils_cost = 10)
  (h4 : nieces = 3)
  (h5 : final_amount = 135) :
  let knife_cost := 2 * utensils_cost
  let single_set_cost := hand_mitts_cost + apron_cost + utensils_cost + knife_cost
  let total_cost := nieces * single_set_cost
  let discount_amount := total_cost - final_amount
  let discount_percentage := (discount_amount / total_cost) * 100
  discount_percentage = 25 := by
sorry


end NUMINAMATH_CALUDE_discount_percentage_calculation_l534_53482


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l534_53424

theorem diophantine_equation_solutions :
  ∀ m n : ℤ, m^2 - 2*m*n - 3*n^2 = 5 ↔ 
    ((m = 4 ∧ n = 1) ∨ (m = 2 ∧ n = -1) ∨ (m = -4 ∧ n = -1) ∨ (m = -2 ∧ n = 1)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l534_53424


namespace NUMINAMATH_CALUDE_decimal_2009_to_octal_l534_53494

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem decimal_2009_to_octal :
  decimal_to_octal 2009 = [3, 7, 3, 1] :=
by sorry

end NUMINAMATH_CALUDE_decimal_2009_to_octal_l534_53494


namespace NUMINAMATH_CALUDE_largest_root_ratio_l534_53431

def f (x : ℝ) : ℝ := 1 - x - 4*x^2 + x^4

def g (x : ℝ) : ℝ := 16 - 8*x - 16*x^2 + x^4

theorem largest_root_ratio :
  ∃ (x₁ x₂ : ℝ),
    (∀ y, f y = 0 → y ≤ x₁) ∧
    (f x₁ = 0) ∧
    (∀ z, g z = 0 → z ≤ x₂) ∧
    (g x₂ = 0) ∧
    x₁ / x₂ = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_root_ratio_l534_53431


namespace NUMINAMATH_CALUDE_roger_has_two_more_candies_l534_53450

/-- The number of candy bags Sandra has -/
def sandra_bags : ℕ := 2

/-- The number of candy pieces in each of Sandra's bags -/
def sandra_pieces_per_bag : ℕ := 6

/-- The number of candy bags Roger has -/
def roger_bags : ℕ := 2

/-- The number of candy pieces in Roger's first bag -/
def roger_bag1_pieces : ℕ := 11

/-- The number of candy pieces in Roger's second bag -/
def roger_bag2_pieces : ℕ := 3

/-- Theorem stating that Roger has 2 more pieces of candy than Sandra -/
theorem roger_has_two_more_candies : 
  (roger_bag1_pieces + roger_bag2_pieces) - (sandra_bags * sandra_pieces_per_bag) = 2 := by
  sorry

end NUMINAMATH_CALUDE_roger_has_two_more_candies_l534_53450


namespace NUMINAMATH_CALUDE_q_is_false_l534_53462

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q :=
by sorry

end NUMINAMATH_CALUDE_q_is_false_l534_53462


namespace NUMINAMATH_CALUDE_base_equation_solution_l534_53484

/-- Converts a list of digits in base b to its decimal representation -/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Checks if a list of digits is valid in base b -/
def valid_digits (digits : List Nat) (b : Nat) : Prop :=
  digits.all (· < b)

theorem base_equation_solution :
  ∃! b : Nat, b > 1 ∧
    valid_digits [3, 4, 6, 4] b ∧
    valid_digits [4, 6, 2, 3] b ∧
    valid_digits [1, 0, 0, 0, 0] b ∧
    to_decimal [3, 4, 6, 4] b + to_decimal [4, 6, 2, 3] b = to_decimal [1, 0, 0, 0, 0] b :=
by
  sorry

end NUMINAMATH_CALUDE_base_equation_solution_l534_53484


namespace NUMINAMATH_CALUDE_polynomial_equality_implies_sum_l534_53418

theorem polynomial_equality_implies_sum (m n : ℝ) : 
  (∀ x : ℝ, (x + 8) * (x - 1) = x^2 + m*x + n) → m + n = -1 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_implies_sum_l534_53418


namespace NUMINAMATH_CALUDE_singer_work_hours_l534_53461

-- Define the number of songs
def num_songs : ℕ := 3

-- Define the number of days per song
def days_per_song : ℕ := 10

-- Define the total number of hours worked
def total_hours : ℕ := 300

-- Define the function to calculate hours per day
def hours_per_day (n s d t : ℕ) : ℚ :=
  t / (n * d)

-- Theorem statement
theorem singer_work_hours :
  hours_per_day num_songs days_per_song total_hours = 10 := by
  sorry

end NUMINAMATH_CALUDE_singer_work_hours_l534_53461


namespace NUMINAMATH_CALUDE_truck_transport_problem_l534_53432

/-- Represents the problem of determining the number of trucks needed to transport goods --/
theorem truck_transport_problem (truck_capacity : ℕ) (partial_load : ℕ) (remaining_goods : ℕ) :
  truck_capacity = 8 →
  partial_load = 4 →
  remaining_goods = 20 →
  ∃ (num_trucks : ℕ) (total_goods : ℕ),
    num_trucks = 6 ∧
    total_goods = 44 ∧
    partial_load * num_trucks + remaining_goods = total_goods ∧
    0 < total_goods - truck_capacity * (num_trucks - 1) ∧
    total_goods - truck_capacity * (num_trucks - 1) < truck_capacity :=
by sorry

end NUMINAMATH_CALUDE_truck_transport_problem_l534_53432


namespace NUMINAMATH_CALUDE_hazel_walk_l534_53422

/-- The distance Hazel walked in the first hour -/
def first_hour_distance : ℝ := 2

/-- The distance Hazel walked in the second hour -/
def second_hour_distance (x : ℝ) : ℝ := 2 * x

/-- The total distance Hazel walked in 2 hours -/
def total_distance : ℝ := 6

theorem hazel_walk :
  first_hour_distance + second_hour_distance first_hour_distance = total_distance :=
by sorry

end NUMINAMATH_CALUDE_hazel_walk_l534_53422


namespace NUMINAMATH_CALUDE_eugene_model_house_l534_53427

theorem eugene_model_house (cards_per_deck : ℕ) (unused_cards : ℕ) (toothpicks_per_card : ℕ) (toothpicks_per_box : ℕ) :
  cards_per_deck = 52 →
  unused_cards = 16 →
  toothpicks_per_card = 75 →
  toothpicks_per_box = 450 →
  toothpicks_per_box = 450 :=
by
  sorry

end NUMINAMATH_CALUDE_eugene_model_house_l534_53427


namespace NUMINAMATH_CALUDE_hostel_problem_solution_l534_53491

/-- Represents the hostel problem with given initial conditions -/
structure HostelProblem where
  initial_students : ℕ
  budget_decrease : ℕ
  expenditure_increase : ℕ
  new_total_expenditure : ℕ

/-- Calculates the number of new students given a HostelProblem -/
def new_students (problem : HostelProblem) : ℕ :=
  -- Implementation not provided, as per instructions
  sorry

/-- Theorem stating that for the given problem, 35 new students joined -/
theorem hostel_problem_solution :
  let problem : HostelProblem := {
    initial_students := 100,
    budget_decrease := 10,
    expenditure_increase := 400,
    new_total_expenditure := 5400
  }
  new_students problem = 35 := by
  sorry

end NUMINAMATH_CALUDE_hostel_problem_solution_l534_53491


namespace NUMINAMATH_CALUDE_peter_pizza_fraction_l534_53423

theorem peter_pizza_fraction (total_slices : ℕ) (peter_alone : ℕ) (shared_paul : ℚ) (shared_patty : ℚ) :
  total_slices = 16 →
  peter_alone = 3 →
  shared_paul = 1 / 2 →
  shared_patty = 1 / 2 →
  (peter_alone : ℚ) / total_slices + shared_paul / total_slices + shared_patty / total_slices = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_peter_pizza_fraction_l534_53423


namespace NUMINAMATH_CALUDE_notebook_pen_combinations_l534_53469

theorem notebook_pen_combinations (notebooks : Finset α) (pens : Finset β) 
  (h1 : notebooks.card = 4) (h2 : pens.card = 5) :
  (notebooks.product pens).card = 20 := by
  sorry

end NUMINAMATH_CALUDE_notebook_pen_combinations_l534_53469


namespace NUMINAMATH_CALUDE_deal_or_no_deal_boxes_to_eliminate_l534_53442

def box_values : List ℝ := [0.01, 1, 5, 10, 25, 50, 100, 200, 300, 400, 500, 750, 1000, 5000, 10000, 25000, 50000, 75000, 100000, 200000, 300000, 400000, 500000, 750000, 1000000]

def total_boxes : ℕ := 30

def threshold : ℝ := 200000

theorem deal_or_no_deal_boxes_to_eliminate :
  let high_value_boxes := (box_values.filter (λ x => x ≥ threshold)).length
  let boxes_to_keep := 2 * high_value_boxes
  total_boxes - boxes_to_keep = 16 :=
by sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_boxes_to_eliminate_l534_53442


namespace NUMINAMATH_CALUDE_four_numbers_sum_product_l534_53467

/-- Given four real numbers x₁, x₂, x₃, x₄, if the sum of any one number and the product 
    of the other three is equal to 2, then the only possible solutions are 
    (1, 1, 1, 1) and (-1, -1, -1, 3) and its permutations. -/
theorem four_numbers_sum_product (x₁ x₂ x₃ x₄ : ℝ) : 
  (x₁ + x₂ * x₃ * x₄ = 2) ∧ 
  (x₂ + x₃ * x₄ * x₁ = 2) ∧ 
  (x₃ + x₄ * x₁ * x₂ = 2) ∧ 
  (x₄ + x₁ * x₂ * x₃ = 2) →
  ((x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1 ∧ x₄ = 1) ∨
   (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = 3) ∨
   (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = 3 ∧ x₄ = -1) ∨
   (x₁ = -1 ∧ x₂ = 3 ∧ x₃ = -1 ∧ x₄ = -1) ∨
   (x₁ = 3 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = -1)) :=
by sorry

end NUMINAMATH_CALUDE_four_numbers_sum_product_l534_53467


namespace NUMINAMATH_CALUDE_width_of_specific_box_l534_53490

/-- A rectangular box with given dimensions -/
structure RectangularBox where
  height : ℝ
  length : ℝ
  width : ℝ
  diagonal : ℝ
  height_positive : height > 0
  length_eq_twice_height : length = 2 * height
  diagonal_formula : diagonal^2 = length^2 + width^2 + height^2

/-- Theorem stating the width of a specific rectangular box -/
theorem width_of_specific_box :
  ∀ (box : RectangularBox),
    box.height = 8 ∧ 
    box.diagonal = 20 →
    box.width = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_width_of_specific_box_l534_53490


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l534_53400

/-- A fraction is reducible if the GCD of its numerator and denominator is greater than 1 -/
def IsReducible (n : ℕ) : Prop :=
  Nat.gcd (n - 17) (3 * n + 4) > 1

/-- The fraction (n-17)/(3n+4) is non-zero for positive n -/
def IsNonZero (n : ℕ) : Prop :=
  n > 0 ∧ n ≠ 17

theorem least_reducible_fraction :
  IsReducible 22 ∧ IsNonZero 22 ∧ ∀ n < 22, ¬(IsReducible n ∧ IsNonZero n) :=
sorry

end NUMINAMATH_CALUDE_least_reducible_fraction_l534_53400


namespace NUMINAMATH_CALUDE_sum_distances_bound_l534_53436

/-- A convex quadrilateral with side lengths p, q, r, s, where p ≤ q ≤ r ≤ s -/
structure ConvexQuadrilateral where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  p_le_q : p ≤ q
  q_le_r : q ≤ r
  r_le_s : r ≤ s
  convex : True  -- Assuming convexity without formal definition

/-- The sum of distances from an interior point to each side of the quadrilateral -/
def sum_distances (quad : ConvexQuadrilateral) (P : ℝ × ℝ) : ℝ :=
  sorry  -- Definition of the sum of distances

/-- Theorem: The sum of distances from any interior point to each side 
    is less than or equal to 3 times the sum of all side lengths -/
theorem sum_distances_bound (quad : ConvexQuadrilateral) (P : ℝ × ℝ) :
  sum_distances quad P ≤ 3 * (quad.p + quad.q + quad.r + quad.s) :=
sorry

end NUMINAMATH_CALUDE_sum_distances_bound_l534_53436


namespace NUMINAMATH_CALUDE_quadratic_square_of_binomial_l534_53433

/-- If x^2 + 110x + d is equal to the square of a binomial, then d = 3025 -/
theorem quadratic_square_of_binomial (d : ℝ) :
  (∃ b : ℝ, ∀ x : ℝ, x^2 + 110*x + d = (x + b)^2) → d = 3025 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_square_of_binomial_l534_53433


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l534_53457

/-- The line x = k intersects the parabola x = -3y^2 - 2y + 7 at exactly one point if and only if k = 22/3 -/
theorem line_parabola_intersection (k : ℝ) : 
  (∃! y : ℝ, k = -3 * y^2 - 2 * y + 7) ↔ k = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l534_53457


namespace NUMINAMATH_CALUDE_three_valid_starting_days_l534_53477

/-- Represents the days of the week -/
inductive Weekday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the number of occurrences of a specific weekday in a 30-day month starting on a given day -/
def countWeekday (start : Weekday) (day : Weekday) : Nat :=
  sorry

/-- Checks if the number of Tuesdays and Fridays are equal in a 30-day month starting on a given day -/
def equalTuesdaysFridays (start : Weekday) : Prop :=
  countWeekday start Weekday.Tuesday = countWeekday start Weekday.Friday

/-- The set of all possible starting days that result in equal Tuesdays and Fridays -/
def validStartingDays : Finset Weekday :=
  sorry

/-- Theorem stating that there are exactly 3 valid starting days for a 30-day month with equal Tuesdays and Fridays -/
theorem three_valid_starting_days :
  Finset.card validStartingDays = 3 :=
sorry

end NUMINAMATH_CALUDE_three_valid_starting_days_l534_53477


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l534_53451

def U : Set ℝ := Set.univ

def A : Set ℝ := {-1, 0, 1, 2, 3}

def B : Set ℝ := {x : ℝ | x ≥ 2}

theorem intersection_complement_theorem : A ∩ (U \ B) = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l534_53451


namespace NUMINAMATH_CALUDE_system_solution_ratio_l534_53495

theorem system_solution_ratio (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
  (eq1 : 8 * x - 6 * y = c) (eq2 : 10 * y - 15 * x = d) : c / d = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l534_53495


namespace NUMINAMATH_CALUDE_work_completion_proof_l534_53486

/-- The number of days B takes to finish the work alone -/
def B : ℝ := 10

/-- The number of days A and B work together -/
def together_days : ℝ := 2

/-- The number of days B takes to finish the remaining work after A leaves -/
def B_remaining : ℝ := 3.0000000000000004

/-- The number of days A takes to finish the work alone -/
def A : ℝ := 4

theorem work_completion_proof :
  2 * (1 / A + 1 / B) + B_remaining * (1 / B) = 1 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_proof_l534_53486


namespace NUMINAMATH_CALUDE_book_cost_price_l534_53420

theorem book_cost_price (cost : ℝ) : 
  (cost * 1.18 - cost * 1.12 = 18) → cost = 300 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_price_l534_53420
