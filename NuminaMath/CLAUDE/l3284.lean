import Mathlib

namespace NUMINAMATH_CALUDE_cookie_jar_problem_l3284_328406

/-- Represents the number of raisins in the larger cookie -/
def larger_cookie_raisins : ℕ := 12

/-- Represents the total number of raisins in the jar -/
def total_raisins : ℕ := 100

/-- Represents the range of cookies in the jar -/
def cookie_range : Set ℕ := {n | 5 ≤ n ∧ n ≤ 10}

theorem cookie_jar_problem (n : ℕ) (h_n : n ∈ cookie_range) :
  ∃ (r : ℕ),
    r + 1 = larger_cookie_raisins ∧
    (n - 1) * r + (r + 1) = total_raisins :=
sorry

end NUMINAMATH_CALUDE_cookie_jar_problem_l3284_328406


namespace NUMINAMATH_CALUDE_string_length_problem_l3284_328477

theorem string_length_problem (total_length remaining_length used_length : ℝ) : 
  total_length = 90 →
  remaining_length = total_length - 30 →
  used_length = (8 / 15) * remaining_length →
  used_length = 32 := by
sorry

end NUMINAMATH_CALUDE_string_length_problem_l3284_328477


namespace NUMINAMATH_CALUDE_product_of_primes_summing_to_91_l3284_328453

theorem product_of_primes_summing_to_91 (p q : ℕ) : 
  Prime p → Prime q → p + q = 91 → p * q = 178 := by
  sorry

end NUMINAMATH_CALUDE_product_of_primes_summing_to_91_l3284_328453


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l3284_328437

/-- Given two adjacent points (1,2) and (4,6) on a square, prove that the area of the square is 25. -/
theorem square_area_from_adjacent_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (4, 6)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l3284_328437


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3284_328451

-- Define set A
def A : Set ℝ := {x | -1 ≤ 2*x + 1 ∧ 2*x + 1 ≤ 3}

-- Define set B
def B : Set ℝ := {x | x ≠ 0 ∧ (x - 2) / x ≤ 0}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3284_328451


namespace NUMINAMATH_CALUDE_tan_beta_value_l3284_328470

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_value_l3284_328470


namespace NUMINAMATH_CALUDE_ryosuke_trip_gas_cost_l3284_328445

/-- Calculates the cost of gas for a trip given odometer readings, fuel efficiency, and gas price -/
def gas_cost_for_trip (initial_reading final_reading : ℕ) (fuel_efficiency : ℚ) (gas_price : ℚ) : ℚ :=
  let distance := final_reading - initial_reading
  let gas_used := (distance : ℚ) / fuel_efficiency
  gas_used * gas_price

/-- Theorem: The cost of gas for Ryosuke's trip is approximately $3.47 -/
theorem ryosuke_trip_gas_cost :
  let cost := gas_cost_for_trip 74568 74592 28 (405/100)
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1/100) ∧ |cost - (347/100)| < ε := by
  sorry

#eval gas_cost_for_trip 74568 74592 28 (405/100)

end NUMINAMATH_CALUDE_ryosuke_trip_gas_cost_l3284_328445


namespace NUMINAMATH_CALUDE_tenth_pebble_count_l3284_328400

def pebble_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 5
  | 2 => 12
  | 3 => 22
  | (n + 4) => pebble_sequence (n + 3) + 3 * (n + 4) - 2

theorem tenth_pebble_count : pebble_sequence 9 = 145 := by
  sorry

end NUMINAMATH_CALUDE_tenth_pebble_count_l3284_328400


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3284_328463

theorem polynomial_remainder (x : ℝ) : 
  (x^5 + 2*x^3 - x + 4) % (x - 2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3284_328463


namespace NUMINAMATH_CALUDE_fraction_difference_equals_negative_one_l3284_328412

theorem fraction_difference_equals_negative_one 
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x - y = x * y) : 
  1 / x - 1 / y = -1 := by
sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_negative_one_l3284_328412


namespace NUMINAMATH_CALUDE_cat_adoption_cost_l3284_328478

/-- The cost to get each cat ready for adoption -/
def cat_cost : ℝ := 50

/-- The cost to get each adult dog ready for adoption -/
def adult_dog_cost : ℝ := 100

/-- The cost to get each puppy ready for adoption -/
def puppy_cost : ℝ := 150

/-- The number of cats adopted -/
def num_cats : ℕ := 2

/-- The number of adult dogs adopted -/
def num_adult_dogs : ℕ := 3

/-- The number of puppies adopted -/
def num_puppies : ℕ := 2

/-- The total cost to get all adopted animals ready -/
def total_cost : ℝ := 700

/-- Theorem stating that the cost to get each cat ready for adoption is $50 -/
theorem cat_adoption_cost : 
  cat_cost * num_cats + adult_dog_cost * num_adult_dogs + puppy_cost * num_puppies = total_cost :=
by sorry

end NUMINAMATH_CALUDE_cat_adoption_cost_l3284_328478


namespace NUMINAMATH_CALUDE_cubic_square_fraction_inequality_l3284_328489

theorem cubic_square_fraction_inequality {s r : ℝ} (hs : 0 < s) (hr : 0 < r) (hsr : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_square_fraction_inequality_l3284_328489


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l3284_328434

/-- Represents a parabola of the form x = 3y^2 - 9y + 4 -/
def Parabola : ℝ → ℝ := λ y => 3 * y^2 - 9 * y + 4

/-- The x-intercept of the parabola -/
def a : ℝ := Parabola 0

/-- The y-intercepts of the parabola -/
def y_intercepts : Set ℝ := {y | Parabola y = 0}

theorem parabola_intercepts_sum :
  ∃ (b c : ℝ), y_intercepts = {b, c} ∧ a + b + c = 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l3284_328434


namespace NUMINAMATH_CALUDE_difference_of_squares_l3284_328407

theorem difference_of_squares (a : ℝ) : (a + 3) * (a - 3) = a^2 - 9 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3284_328407


namespace NUMINAMATH_CALUDE_sequence_general_term_l3284_328430

-- Define the sequence a_n and its partial sum S_n
def S (a : ℕ → ℤ) (n : ℕ) : ℤ := 2 * a n + 1

-- State the theorem
theorem sequence_general_term (a : ℕ → ℤ) :
  (∀ n : ℕ, S a n = 2 * a n + 1) →
  (∀ n : ℕ, a n = -2^(n-1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3284_328430


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l3284_328402

theorem angle_in_fourth_quadrant (α : Real) 
  (h1 : Real.sin α < 0) (h2 : Real.cos α > 0) : 
  α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi) :=
sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l3284_328402


namespace NUMINAMATH_CALUDE_projection_magnitude_l3284_328461

def vector_a : ℝ × ℝ := (7, -4)
def vector_b : ℝ × ℝ := (-8, 6)

theorem projection_magnitude :
  let dot_product := vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2
  let magnitude_b := Real.sqrt (vector_b.1^2 + vector_b.2^2)
  let projection := dot_product / magnitude_b
  |projection| = 8 := by sorry

end NUMINAMATH_CALUDE_projection_magnitude_l3284_328461


namespace NUMINAMATH_CALUDE_parabola_directrix_l3284_328491

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -1/4 * y

-- Define the directrix
def directrix (y : ℝ) : Prop := y = -1/16

-- Theorem statement
theorem parabola_directrix :
  ∀ (x y : ℝ), parabola x y → directrix y :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3284_328491


namespace NUMINAMATH_CALUDE_track_team_composition_l3284_328498

/-- The number of children on a track team after changes in composition -/
theorem track_team_composition (initial_girls initial_boys girls_joined boys_quit : ℕ) :
  initial_girls = 18 →
  initial_boys = 15 →
  girls_joined = 7 →
  boys_quit = 4 →
  (initial_girls + girls_joined) + (initial_boys - boys_quit) = 36 := by
  sorry


end NUMINAMATH_CALUDE_track_team_composition_l3284_328498


namespace NUMINAMATH_CALUDE_ratatouille_price_proof_l3284_328486

def ratatouille_problem (eggplant_weight : ℝ) (zucchini_weight : ℝ) 
  (tomato_price : ℝ) (tomato_weight : ℝ)
  (onion_price : ℝ) (onion_weight : ℝ)
  (basil_price : ℝ) (basil_weight : ℝ)
  (total_quarts : ℝ) (price_per_quart : ℝ) : Prop :=
  let total_weight := eggplant_weight + zucchini_weight
  let other_ingredients_cost := tomato_price * tomato_weight + 
                                onion_price * onion_weight + 
                                basil_price * basil_weight * 2
  let total_cost := total_quarts * price_per_quart
  let eggplant_zucchini_cost := total_cost - other_ingredients_cost
  let price_per_pound := eggplant_zucchini_cost / total_weight
  price_per_pound = 2

theorem ratatouille_price_proof :
  ratatouille_problem 5 4 3.5 4 1 3 2.5 1 4 10 := by
  sorry

end NUMINAMATH_CALUDE_ratatouille_price_proof_l3284_328486


namespace NUMINAMATH_CALUDE_coal_burn_duration_l3284_328493

/-- Given a factory with 300 tons of coal, this theorem establishes the relationship
    between the number of days the coal can burn and the average daily consumption. -/
theorem coal_burn_duration (x : ℝ) (y : ℝ) (h : x > 0) :
  y = 300 / x ↔ y * x = 300 :=
by sorry

end NUMINAMATH_CALUDE_coal_burn_duration_l3284_328493


namespace NUMINAMATH_CALUDE_circle_path_in_triangle_l3284_328427

/-- The path length of the center of a circle rolling inside a triangle --/
def circle_path_length (a b c r : ℝ) : ℝ :=
  (a - 2*r) + (b - 2*r) + (c - 2*r)

/-- Theorem stating the path length of a circle's center rolling inside a specific triangle --/
theorem circle_path_in_triangle : 
  let a : ℝ := 8
  let b : ℝ := 10
  let c : ℝ := 12.5
  let r : ℝ := 1.5
  circle_path_length a b c r = 21.5 := by
  sorry

#check circle_path_in_triangle

end NUMINAMATH_CALUDE_circle_path_in_triangle_l3284_328427


namespace NUMINAMATH_CALUDE_compound_composition_l3284_328424

def atomic_weight_N : ℕ := 14
def atomic_weight_H : ℕ := 1
def atomic_weight_Br : ℕ := 80
def molecular_weight : ℕ := 98

theorem compound_composition (n : ℕ) : 
  atomic_weight_N + n * atomic_weight_H + atomic_weight_Br = molecular_weight → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_compound_composition_l3284_328424


namespace NUMINAMATH_CALUDE_equation_classification_l3284_328414

-- Define what a linear equation in two variables is
def is_linear_equation_in_two_variables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y + c

-- Define the properties of the equation in question
def has_two_unknowns_and_degree_one (f : ℝ → ℝ → ℝ) : Prop :=
  (∃ (x y : ℝ), f x y ≠ f x 0 ∧ f x y ≠ f 0 y) ∧ 
  (∀ (x y : ℝ), ∃ (a b c : ℝ), f x y = a * x + b * y + c)

-- State the theorem
theorem equation_classification 
  (f : ℝ → ℝ → ℝ) 
  (h : has_two_unknowns_and_degree_one f) : 
  is_linear_equation_in_two_variables f :=
sorry

end NUMINAMATH_CALUDE_equation_classification_l3284_328414


namespace NUMINAMATH_CALUDE_equation_solution_l3284_328499

theorem equation_solution : ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -1 ∧
  ∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3284_328499


namespace NUMINAMATH_CALUDE_sheep_count_l3284_328492

theorem sheep_count (total animals : ℕ) (cows goats : ℕ) : 
  total = 200 → 
  cows = 40 → 
  goats = 104 → 
  animals = cows + goats + (total - cows - goats) → 
  total - cows - goats = 56 := by
sorry

end NUMINAMATH_CALUDE_sheep_count_l3284_328492


namespace NUMINAMATH_CALUDE_quadrilateral_area_72_l3284_328495

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ :=
  (q.C.x - q.A.x) * (q.C.y - q.B.y)

/-- Theorem: The y-coordinate of B in quadrilateral ABCD that makes its area 72 square units -/
theorem quadrilateral_area_72 (q : Quadrilateral) 
    (h1 : q.A = ⟨0, 0⟩) 
    (h2 : q.B = ⟨8, q.B.y⟩)
    (h3 : q.C = ⟨8, 16⟩)
    (h4 : q.D = ⟨0, 16⟩)
    (h5 : area q = 72) : 
    q.B.y = 9 := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_area_72_l3284_328495


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l3284_328447

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 20 →
  a * b + c + d = 100 →
  a * d + b * c = 250 →
  c * d = 144 →
  a^2 + b^2 + c^2 + d^2 ≤ 1760 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l3284_328447


namespace NUMINAMATH_CALUDE_victor_final_books_l3284_328432

/-- The number of books Victor has after various transactions -/
def final_books (initial : ℝ) (bought : ℝ) (given : ℝ) (donated : ℝ) : ℝ :=
  initial + bought - given - donated

/-- Theorem stating that Victor ends up with 19.8 books -/
theorem victor_final_books :
  final_books 35.5 12.3 7.2 20.8 = 19.8 := by
  sorry

end NUMINAMATH_CALUDE_victor_final_books_l3284_328432


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l3284_328408

theorem correct_mean_calculation (n : ℕ) (incorrect_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) :
  n = 30 ∧ 
  incorrect_mean = 150 ∧ 
  incorrect_value = 135 ∧ 
  correct_value = 165 →
  (n * incorrect_mean - incorrect_value + correct_value) / n = 151 := by sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l3284_328408


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l3284_328442

theorem factorial_equation_solution : ∃ k : ℕ, (4 * 3 * 2 * 1) * (2 * 1) = 2 * k * (3 * 2 * 1) ∧ k = 4 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l3284_328442


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l3284_328481

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500) → (∀ m : ℕ, m > n → m * (m + 1) ≥ 500) → n + (n + 1) = 43 :=
by sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l3284_328481


namespace NUMINAMATH_CALUDE_fraction_value_l3284_328484

theorem fraction_value (a b c d : ℝ) 
  (ha : a = 4 * b) 
  (hb : b = 3 * c) 
  (hc : c = 5 * d) : 
  (a * b) / (c * d) = 180 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l3284_328484


namespace NUMINAMATH_CALUDE_root_product_value_l3284_328479

theorem root_product_value (p q : ℝ) : 
  3 * p ^ 2 + 9 * p - 21 = 0 →
  3 * q ^ 2 + 9 * q - 21 = 0 →
  (3 * p - 4) * (6 * q - 8) = -22 := by
sorry

end NUMINAMATH_CALUDE_root_product_value_l3284_328479


namespace NUMINAMATH_CALUDE_sum_equals_zero_l3284_328405

def f (x : ℝ) : ℝ := x^2 * (1 - x)^2

theorem sum_equals_zero :
  f (1/7) - f (2/7) + f (3/7) - f (4/7) + f (5/7) - f (6/7) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_zero_l3284_328405


namespace NUMINAMATH_CALUDE_fiftieth_ring_squares_l3284_328467

/-- The number of squares in the nth ring around a 3x3 centered square -/
def ring_squares (n : ℕ) : ℕ :=
  if n = 1 then 16
  else if n = 2 then 24
  else 33 + 24 * (n - 1)

/-- The 50th ring contains 1209 unit squares -/
theorem fiftieth_ring_squares :
  ring_squares 50 = 1209 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_ring_squares_l3284_328467


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_segments_ratio_l3284_328476

theorem right_triangle_hypotenuse_segments_ratio 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_ratio : a / b = 3 / 4) :
  let d := (a * c) / (a + b)
  (c - d) / d = 16 / 9 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_segments_ratio_l3284_328476


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3284_328474

theorem polynomial_division_remainder : ∃ q : Polynomial ℂ, 
  (X^4 - 1) * (X^3 - 1) = (X^2 + 1) * q + (2 + X) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3284_328474


namespace NUMINAMATH_CALUDE_bread_flour_calculation_l3284_328419

theorem bread_flour_calculation (x : ℝ) : 
  x > 0 ∧ 
  x + 10 > 0 ∧ 
  x * (1 + x / 100) + (x + 10) * (1 + (x + 10) / 100) = 112.5 → 
  x = 35 :=
by sorry

end NUMINAMATH_CALUDE_bread_flour_calculation_l3284_328419


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3284_328439

theorem min_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (sum_eq : x + y + z = 3) (z_eq : z = 1) :
  1/x + 1/y + 1/z ≥ 3 ∧ (1/x + 1/y + 1/z = 3 ↔ x = 1 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3284_328439


namespace NUMINAMATH_CALUDE_sequence_inequality_l3284_328462

theorem sequence_inequality (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a n + a (2 * n) ≥ 3 * n)
  (h2 : ∀ n : ℕ, a (n + 1) + n ≤ 2 * Real.sqrt (a n * (n + 1)))
  (h3 : ∀ n : ℕ, 0 ≤ a n) :
  ∀ n : ℕ, a n ≥ n :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3284_328462


namespace NUMINAMATH_CALUDE_max_regions_quadratic_polynomials_l3284_328468

/-- The maximum number of regions into which the coordinate plane can be divided
    by n quadratic polynomials of the form y = a_i * x^2 + b_i * x + c_i,
    where i = 1, 2, ..., n, is n^2 + 1. -/
theorem max_regions_quadratic_polynomials (n : ℕ) :
  (∃ (polynomials : Fin n → ℝ → ℝ),
    (∀ i : Fin n, ∃ (a b c : ℝ), ∀ x, polynomials i x = a * x^2 + b * x + c) →
    ∃ (num_regions : ℕ),
      num_regions ≤ n^2 + 1 ∧
      ∀ (m : ℕ), m > n^2 + 1 →
        ¬∃ (regions : Fin m → Set (ℝ × ℝ)),
          (∀ i j, i ≠ j → regions i ∩ regions j = ∅) ∧
          (⋃ i, regions i) = Set.univ ∧
          ∀ i, ∃ x y, (x, y) ∈ regions i ∧
            (∀ j, polynomials j y = x ∨ y > polynomials j x)) :=
sorry

end NUMINAMATH_CALUDE_max_regions_quadratic_polynomials_l3284_328468


namespace NUMINAMATH_CALUDE_class_average_mark_l3284_328450

theorem class_average_mark (n1 n2 : ℕ) (avg2 avg_total : ℝ) (h1 : n1 = 30) (h2 : n2 = 50) 
    (h3 : avg2 = 90) (h4 : avg_total = 71.25) : 
  (n1 + n2 : ℝ) * avg_total = n1 * ((n1 + n2 : ℝ) * avg_total - n2 * avg2) / n1 + n2 * avg2 := by
  sorry

end NUMINAMATH_CALUDE_class_average_mark_l3284_328450


namespace NUMINAMATH_CALUDE_circle_transformation_l3284_328411

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Translates a point to the right by a given amount -/
def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)

/-- The main theorem -/
theorem circle_transformation :
  let T : ℝ × ℝ := (-2, 6)
  let reflected := reflect_x T
  let final := translate_right reflected 5
  final = (3, -6) := by sorry

end NUMINAMATH_CALUDE_circle_transformation_l3284_328411


namespace NUMINAMATH_CALUDE_midline_leg_relation_l3284_328452

/-- A right triangle with legs a and b, and midlines K₁ and K₂. -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  K₁ : ℝ
  K₂ : ℝ
  a_positive : 0 < a
  b_positive : 0 < b
  K₁_eq : K₁^2 = (a/2)^2 + b^2
  K₂_eq : K₂^2 = a^2 + (b/2)^2

/-- The main theorem about the relationship between midlines and leg in a right triangle. -/
theorem midline_leg_relation (t : RightTriangle) : 16 * t.K₂^2 - 4 * t.K₁^2 = 15 * t.a^2 := by
  sorry

end NUMINAMATH_CALUDE_midline_leg_relation_l3284_328452


namespace NUMINAMATH_CALUDE_square_area_ratio_sqrt_l3284_328417

theorem square_area_ratio_sqrt (side_C side_D : ℝ) (h1 : side_C = 45) (h2 : side_D = 60) :
  Real.sqrt ((side_C ^ 2) / (side_D ^ 2)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_sqrt_l3284_328417


namespace NUMINAMATH_CALUDE_rogers_broken_crayons_l3284_328401

/-- Given that Roger has 14 crayons in total, 2 new crayons, and 4 used crayons,
    prove that he has 8 broken crayons. -/
theorem rogers_broken_crayons :
  let total_crayons : ℕ := 14
  let new_crayons : ℕ := 2
  let used_crayons : ℕ := 4
  let broken_crayons : ℕ := total_crayons - new_crayons - used_crayons
  broken_crayons = 8 := by
  sorry

end NUMINAMATH_CALUDE_rogers_broken_crayons_l3284_328401


namespace NUMINAMATH_CALUDE_total_puzzle_time_l3284_328469

def puzzle_time (warm_up_time : ℕ) (additional_puzzles : ℕ) (time_factor : ℕ) : ℕ :=
  warm_up_time + additional_puzzles * (warm_up_time * time_factor)

theorem total_puzzle_time :
  puzzle_time 10 2 3 = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_total_puzzle_time_l3284_328469


namespace NUMINAMATH_CALUDE_trapezoid_properties_l3284_328409

/-- Represents a trapezoid ABCD with AB and CD as parallel bases (AB < CD) -/
structure Trapezoid where
  AD : ℝ  -- Length of larger base
  BC : ℝ  -- Length of smaller base
  AB : ℝ  -- Length of shorter leg
  midline : ℝ  -- Length of midline
  midpoint_segment : ℝ  -- Length of segment connecting midpoints of bases
  angle1 : ℝ  -- Angle at one end of larger base (in degrees)
  angle2 : ℝ  -- Angle at other end of larger base (in degrees)

/-- Theorem stating the properties of the specific trapezoid in the problem -/
theorem trapezoid_properties (T : Trapezoid) 
  (h1 : T.midline = 5)
  (h2 : T.midpoint_segment = 3)
  (h3 : T.angle1 = 30)
  (h4 : T.angle2 = 60) :
  T.AD = 8 ∧ T.BC = 2 ∧ T.AB = 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_properties_l3284_328409


namespace NUMINAMATH_CALUDE_sarahs_wallet_l3284_328428

theorem sarahs_wallet (total_amount : ℕ) (total_bills : ℕ) (five_dollar_count : ℕ) (ten_dollar_count : ℕ) : 
  total_amount = 100 →
  total_bills = 15 →
  five_dollar_count + ten_dollar_count = total_bills →
  5 * five_dollar_count + 10 * ten_dollar_count = total_amount →
  five_dollar_count = 10 := by
sorry

end NUMINAMATH_CALUDE_sarahs_wallet_l3284_328428


namespace NUMINAMATH_CALUDE_class_size_l3284_328487

/-- The number of girls in Jungkook's class -/
def num_girls : ℕ := 9

/-- The number of boys in Jungkook's class -/
def num_boys : ℕ := 16

/-- The total number of students in Jungkook's class -/
def total_students : ℕ := num_girls + num_boys

theorem class_size : total_students = 25 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l3284_328487


namespace NUMINAMATH_CALUDE_sunny_lead_in_new_race_l3284_328438

/-- Represents the race conditions and results -/
structure RaceData where
  initial_race_length : ℝ
  initial_sunny_lead : ℝ
  new_race_length : ℝ
  sunny_speed_increase : ℝ
  windy_speed_decrease : ℝ
  sunny_initial_lag : ℝ

/-- Calculates Sunny's lead at the end of the new race -/
def calculate_sunny_lead (data : RaceData) : ℝ :=
  sorry

/-- Theorem stating that given the race conditions, Sunny's lead at the end of the new race is 106.25 meters -/
theorem sunny_lead_in_new_race (data : RaceData) 
  (h1 : data.initial_race_length = 400)
  (h2 : data.initial_sunny_lead = 50)
  (h3 : data.new_race_length = 500)
  (h4 : data.sunny_speed_increase = 0.1)
  (h5 : data.windy_speed_decrease = 0.1)
  (h6 : data.sunny_initial_lag = 50) :
  calculate_sunny_lead data = 106.25 :=
sorry

end NUMINAMATH_CALUDE_sunny_lead_in_new_race_l3284_328438


namespace NUMINAMATH_CALUDE_population_increase_rate_l3284_328440

theorem population_increase_rate 
  (initial_population : ℕ) 
  (final_population : ℕ) 
  (increase_rate : ℚ) : 
  initial_population = 240 →
  final_population = 264 →
  increase_rate = (final_population - initial_population : ℚ) / initial_population * 100 →
  increase_rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_population_increase_rate_l3284_328440


namespace NUMINAMATH_CALUDE_sport_formulation_water_amount_l3284_328443

/-- Prove that the sport formulation of a flavored drink contains 30 ounces of water -/
theorem sport_formulation_water_amount :
  -- Standard formulation ratio
  let standard_ratio : Fin 3 → ℚ := ![1, 12, 30]
  -- Sport formulation ratios relative to standard
  let sport_flavoring_corn_ratio := 3
  let sport_flavoring_water_ratio := 1 / 2
  -- Amount of corn syrup in sport formulation
  let sport_corn_syrup := 2

  -- The amount of water in the sport formulation
  ∃ (water : ℚ),
    -- Sport formulation flavoring to corn syrup ratio
    sport_flavoring_corn_ratio * standard_ratio 0 / standard_ratio 1 = sport_corn_syrup / 2 ∧
    -- Sport formulation flavoring to water ratio
    sport_flavoring_water_ratio * standard_ratio 0 / standard_ratio 2 = 2 / water ∧
    -- The amount of water is 30 ounces
    water = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_sport_formulation_water_amount_l3284_328443


namespace NUMINAMATH_CALUDE_tetrahedron_volume_is_16_l3284_328496

/-- Represents a tetrahedron PQRS with given side lengths and base area -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ
  area_PQR : ℝ

/-- Calculates the volume of a tetrahedron -/
def tetrahedron_volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the given tetrahedron is 16 -/
theorem tetrahedron_volume_is_16 (t : Tetrahedron) 
  (h1 : t.PQ = 6)
  (h2 : t.PR = 4)
  (h3 : t.PS = 5)
  (h4 : t.QR = 5)
  (h5 : t.QS = 6)
  (h6 : t.RS = 15/2)
  (h7 : t.area_PQR = 12) :
  tetrahedron_volume t = 16 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_is_16_l3284_328496


namespace NUMINAMATH_CALUDE_system_solutions_l3284_328422

theorem system_solutions (x₁ x₂ x₃ x₄ x₅ : ℝ) :
  (x₁ + x₂ = x₃^2 ∧
   x₂ + x₃ = x₄^2 ∧
   x₃ + x₁ = x₅^2 ∧
   x₄ + x₅ = x₁^2 ∧
   x₅ + x₁ = x₂^2) →
  ((x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2) ∨
   (x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l3284_328422


namespace NUMINAMATH_CALUDE_complex_bound_l3284_328456

theorem complex_bound (z : ℂ) (h : Complex.abs (z + z⁻¹) = 1) :
  (Real.sqrt 5 - 1) / 2 ≤ Complex.abs z ∧ Complex.abs z ≤ (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_bound_l3284_328456


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l3284_328488

theorem unique_positive_integer_solution : ∃! (x : ℕ), x > 0 ∧ x - x^2 + 29 = 526 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l3284_328488


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l3284_328459

-- Define the probability of a boy or girl being born
def prob_boy_or_girl : ℚ := 1 / 2

-- Define the number of children in the family
def num_children : ℕ := 4

-- Theorem statement
theorem prob_at_least_one_boy_and_girl :
  (1 : ℚ) - (prob_boy_or_girl ^ num_children + prob_boy_or_girl ^ num_children) = 7 / 8 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l3284_328459


namespace NUMINAMATH_CALUDE_no_integer_root_l3284_328410

theorem no_integer_root (P : ℤ → ℤ) (h_poly : ∀ x y : ℤ, (x - y) ∣ (P x - P y)) 
  (h1 : P 1 = 10) (h_neg1 : P (-1) = 22) (h0 : P 0 = 4) :
  ∀ r : ℤ, P r ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_no_integer_root_l3284_328410


namespace NUMINAMATH_CALUDE_f_extrema_l3284_328472

open Real

noncomputable def f (x : ℝ) : ℝ := cos x + (x + 1) * sin x + 1

theorem f_extrema :
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc 0 (2 * π), f x ≥ min) ∧
    (∃ x ∈ Set.Icc 0 (2 * π), f x = min) ∧
    (∀ x ∈ Set.Icc 0 (2 * π), f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 (2 * π), f x = max) ∧
    min = -3 * π / 2 ∧
    max = π / 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_f_extrema_l3284_328472


namespace NUMINAMATH_CALUDE_onions_removed_l3284_328448

/-- Proves that 5 onions were removed from the scale given the problem conditions -/
theorem onions_removed (total_onions : ℕ) (remaining_onions : ℕ) (total_weight : ℚ) 
  (avg_weight_remaining : ℚ) (avg_weight_removed : ℚ) :
  total_onions = 40 →
  remaining_onions = 35 →
  total_weight = 768/100 →
  avg_weight_remaining = 190/1000 →
  avg_weight_removed = 206/1000 →
  total_onions - remaining_onions = 5 := by
  sorry

#check onions_removed

end NUMINAMATH_CALUDE_onions_removed_l3284_328448


namespace NUMINAMATH_CALUDE_flower_cost_proof_l3284_328425

/-- Proves that if Lilly saves $2 per day for 22 days and can buy 11 flowers with her savings, then each flower costs $4. -/
theorem flower_cost_proof (days : ℕ) (daily_savings : ℚ) (num_flowers : ℕ) 
  (h1 : days = 22) 
  (h2 : daily_savings = 2) 
  (h3 : num_flowers = 11) : 
  (days * daily_savings) / num_flowers = 4 := by
  sorry

#check flower_cost_proof

end NUMINAMATH_CALUDE_flower_cost_proof_l3284_328425


namespace NUMINAMATH_CALUDE_inequality_condition_l3284_328457

def f (x : ℝ) := x^2 - 4*x + 3

theorem inequality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, |x - 1| < b → |f x + 3| < a) ↔ b^2 + 2*b + 3 ≤ a :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l3284_328457


namespace NUMINAMATH_CALUDE_nisos_population_meets_capacity_l3284_328480

/-- Represents the state of Nisos island at a given time -/
structure NisosState where
  year : ℕ
  population : ℕ

/-- Calculates the population after a given number of 20-year periods -/
def population_after (initial_population : ℕ) (periods : ℕ) : ℕ :=
  initial_population * (4 ^ periods)

/-- Theorem: Nisos island population meets capacity limit after 60 years -/
theorem nisos_population_meets_capacity : 
  ∀ (initial_state : NisosState),
    initial_state.year = 1998 →
    initial_state.population = 100 →
    ∃ (final_state : NisosState),
      final_state.year = initial_state.year + 60 ∧
      final_state.population ≥ 7500 ∧
      final_state.population < population_after 100 4 :=
sorry

/-- The land area of Nisos island in hectares -/
def nisos_area : ℕ := 15000

/-- The land area required per person in hectares -/
def land_per_person : ℕ := 2

/-- The capacity of Nisos island -/
def nisos_capacity : ℕ := nisos_area / land_per_person

/-- The population growth factor per 20-year period -/
def growth_factor : ℕ := 4

/-- The number of 20-year periods in 60 years -/
def periods_in_60_years : ℕ := 3

end NUMINAMATH_CALUDE_nisos_population_meets_capacity_l3284_328480


namespace NUMINAMATH_CALUDE_original_figure_area_l3284_328444

/-- The area of the original figure given the properties of its intuitive diagram --/
theorem original_figure_area (height : ℝ) (top_angle : ℝ) (area_ratio : ℝ) : 
  height = 2 → 
  top_angle = 120 * π / 180 → 
  area_ratio = 2 * Real.sqrt 2 → 
  (1 / 2) * (4 * height) * (4 * height) * Real.sin top_angle * area_ratio = 8 * Real.sqrt 6 := by
  sorry

#check original_figure_area

end NUMINAMATH_CALUDE_original_figure_area_l3284_328444


namespace NUMINAMATH_CALUDE_probability_three_dice_divisible_by_10_l3284_328415

-- Define a die as having 6 faces
def die_faces : ℕ := 6

-- Define a function to check if a number is divisible by 2
def divisible_by_2 (n : ℕ) : Prop := n % 2 = 0

-- Define a function to check if a number is divisible by 5
def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

-- Define a function to check if a product is divisible by 10
def product_divisible_by_10 (a b c : ℕ) : Prop :=
  divisible_by_2 (a * b * c) ∧ divisible_by_5 (a * b * c)

-- Define the probability of the event
def probability_divisible_by_10 : ℚ :=
  (144 : ℚ) / (die_faces ^ 3 : ℚ)

-- State the theorem
theorem probability_three_dice_divisible_by_10 :
  probability_divisible_by_10 = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_three_dice_divisible_by_10_l3284_328415


namespace NUMINAMATH_CALUDE_max_binomial_coeff_x_minus_2_pow_5_l3284_328490

theorem max_binomial_coeff_x_minus_2_pow_5 :
  (Finset.range 6).sup (fun k => Nat.choose 5 k) = 10 := by
  sorry

end NUMINAMATH_CALUDE_max_binomial_coeff_x_minus_2_pow_5_l3284_328490


namespace NUMINAMATH_CALUDE_zachary_pushups_l3284_328441

/-- Given that Zachary did 14 crunches and a total of 67 push-ups and crunches,
    prove that Zachary did 53 push-ups. -/
theorem zachary_pushups :
  ∀ (zachary_pushups zachary_crunches : ℕ),
    zachary_crunches = 14 →
    zachary_pushups + zachary_crunches = 67 →
    zachary_pushups = 53 :=
by
  sorry

end NUMINAMATH_CALUDE_zachary_pushups_l3284_328441


namespace NUMINAMATH_CALUDE_b_fourth_zero_implies_b_squared_zero_l3284_328420

theorem b_fourth_zero_implies_b_squared_zero (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B ^ 4 = 0) : B ^ 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_b_fourth_zero_implies_b_squared_zero_l3284_328420


namespace NUMINAMATH_CALUDE_company_attendees_l3284_328473

theorem company_attendees (total : ℕ) (other : ℕ) (h_total : total = 185) (h_other : other = 20) : 
  ∃ (a : ℕ), 
    a + (2 * a) + (a + 10) + (a + 5) + other = total ∧ 
    a = 30 := by
  sorry

end NUMINAMATH_CALUDE_company_attendees_l3284_328473


namespace NUMINAMATH_CALUDE_area_difference_is_one_l3284_328458

-- Define the unit square
def unit_square : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define an equilateral triangle with side length 1
def unit_equilateral_triangle : Set (ℝ × ℝ) := sorry

-- Define the region R (union of square and 12 triangles)
def R : Set (ℝ × ℝ) := sorry

-- Define the smallest convex polygon S containing R
def S : Set (ℝ × ℝ) := sorry

-- Define the area function
noncomputable def area : Set (ℝ × ℝ) → ℝ := sorry

-- Theorem statement
theorem area_difference_is_one :
  area (S \ R) = 1 := by sorry

end NUMINAMATH_CALUDE_area_difference_is_one_l3284_328458


namespace NUMINAMATH_CALUDE_sequence_existence_iff_N_bound_l3284_328482

theorem sequence_existence_iff_N_bound (N : ℕ+) :
  (∃ s : ℕ → ℕ+, 
    (∀ n, s n < s (n + 1)) ∧ 
    (∃ p : ℕ+, ∀ n, s (n + 1) - s n = s (n + 1 + p) - s (n + p)) ∧
    (∀ n : ℕ+, s (s n) - s (s (n - 1)) ≤ N ∧ N < s (1 + s n) - s (s (n - 1))))
  ↔
  (∃ t : ℕ+, t^2 ≤ N ∧ N < t^2 + t) :=
by sorry

end NUMINAMATH_CALUDE_sequence_existence_iff_N_bound_l3284_328482


namespace NUMINAMATH_CALUDE_circle_properties_l3284_328433

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 3)^2 = 36

-- Theorem statement
theorem circle_properties :
  ∃ (C : ℝ × ℝ) (r : ℝ),
    (∀ (x y : ℝ), circle_equation x y ↔ (x - C.1)^2 + (y - C.2)^2 = r^2) ∧
    C = (-1, 3) ∧
    r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l3284_328433


namespace NUMINAMATH_CALUDE_sqrt_sum_2160_l3284_328464

theorem sqrt_sum_2160 (a b : ℕ+) : 
  a < b → 
  (a.val : ℝ).sqrt + (b.val : ℝ).sqrt = Real.sqrt 2160 → 
  a ∈ ({15, 60, 135, 240, 375} : Set ℕ+) := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_2160_l3284_328464


namespace NUMINAMATH_CALUDE_inverse_47_mod_48_l3284_328465

theorem inverse_47_mod_48 : ∃! x : ℕ, x ∈ Finset.range 48 ∧ (47 * x) % 48 = 1 :=
by
  use 47
  sorry

end NUMINAMATH_CALUDE_inverse_47_mod_48_l3284_328465


namespace NUMINAMATH_CALUDE_min_value_of_ab_l3284_328416

theorem min_value_of_ab (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h_seq : (1/4 * Real.log a) * (Real.log b) = (1/4)^2) : 
  (∀ x y : ℝ, x > 1 → y > 1 → (1/4 * Real.log x) * (Real.log y) = (1/4)^2 → a * b ≤ x * y) ∧ 
  a * b = Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_ab_l3284_328416


namespace NUMINAMATH_CALUDE_intersection_implies_a_zero_l3284_328418

theorem intersection_implies_a_zero (a : ℝ) : 
  let A : Set ℝ := {a^2, a + 1, -1}
  let B : Set ℝ := {2*a - 1, |a - 2|, 3*a^2 + 4}
  A ∩ B = {-1} → a = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_zero_l3284_328418


namespace NUMINAMATH_CALUDE_max_remainder_239_div_n_l3284_328423

theorem max_remainder_239_div_n (n : ℕ) (h : n < 135) :
  (Finset.range n).sup (λ m => 239 % m) = 119 := by
  sorry

end NUMINAMATH_CALUDE_max_remainder_239_div_n_l3284_328423


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l3284_328460

theorem inscribed_hexagon_area (circle_area : ℝ) (hexagon_area : ℝ) :
  circle_area = 100 * Real.pi →
  hexagon_area = 6 * (((Real.sqrt (circle_area / Real.pi))^2 * Real.sqrt 3) / 4) →
  hexagon_area = 150 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l3284_328460


namespace NUMINAMATH_CALUDE_max_a_for_function_inequality_l3284_328426

theorem max_a_for_function_inequality (f : ℝ → ℝ) (h : ∀ x, x ∈ [3, 5] → f x = 2 * x / (x - 1)) :
  (∃ a : ℝ, (∀ x, x ∈ [3, 5] → f x ≥ a) ∧ 
   (∀ b : ℝ, (∀ x, x ∈ [3, 5] → f x ≥ b) → b ≤ a)) →
  (∃ a : ℝ, a = 5/2 ∧ 
   (∀ x, x ∈ [3, 5] → f x ≥ a) ∧
   (∀ b : ℝ, (∀ x, x ∈ [3, 5] → f x ≥ b) → b ≤ a)) :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_function_inequality_l3284_328426


namespace NUMINAMATH_CALUDE_turtles_on_log_l3284_328454

theorem turtles_on_log (initial_turtles : ℕ) : initial_turtles = 50 → 226 = initial_turtles + (7 * initial_turtles - 6) - (3 * (initial_turtles + (7 * initial_turtles - 6)) / 7) := by
  sorry

end NUMINAMATH_CALUDE_turtles_on_log_l3284_328454


namespace NUMINAMATH_CALUDE_rectangle_area_change_l3284_328435

theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) :
  let A := L * B
  let L' := L / 2
  let A' := (3 / 2) * A
  let B' := A' / L'
  B' = 3 * B :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l3284_328435


namespace NUMINAMATH_CALUDE_difference_of_squares_75_25_l3284_328431

theorem difference_of_squares_75_25 : 75^2 - 25^2 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_75_25_l3284_328431


namespace NUMINAMATH_CALUDE_orange_stack_count_l3284_328485

/-- Calculates the number of oranges in a single layer of the pyramid -/
def layerOranges (baseWidth : ℕ) (baseLength : ℕ) (layer : ℕ) : ℕ :=
  (baseWidth - layer + 1) * (baseLength - layer + 1)

/-- Calculates the total number of oranges in the pyramid stack -/
def totalOranges (baseWidth : ℕ) (baseLength : ℕ) : ℕ :=
  let numLayers := min baseWidth baseLength
  (List.range numLayers).foldl (fun acc i => acc + layerOranges baseWidth baseLength i) 0

/-- Theorem stating that a pyramid-like stack of oranges with a 6x9 base contains 154 oranges -/
theorem orange_stack_count : totalOranges 6 9 = 154 := by
  sorry

end NUMINAMATH_CALUDE_orange_stack_count_l3284_328485


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3284_328429

/-- The curve C in the x-y plane -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 4) + p.2^2 = 1}

/-- The distance function from a point on C to the line x - y - 4 = 0 -/
def distance_to_line (p : ℝ × ℝ) : ℝ :=
  |p.1 - p.2 - 4|

/-- Theorem: The minimum distance from C to the line x - y - 4 = 0 is 4 - √5 -/
theorem min_distance_to_line :
  ∃ (min_dist : ℝ), min_dist = 4 - Real.sqrt 5 ∧
  (∀ p ∈ C, distance_to_line p ≥ min_dist) ∧
  (∃ p ∈ C, distance_to_line p = min_dist) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3284_328429


namespace NUMINAMATH_CALUDE_exponential_inequality_l3284_328497

theorem exponential_inequality (a b : ℝ) (h : a > b) : (0.9 : ℝ) ^ a < (0.9 : ℝ) ^ b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l3284_328497


namespace NUMINAMATH_CALUDE_not_regressive_a_regressive_increasing_is_arithmetic_l3284_328404

-- Definition of a regressive sequence
def IsRegressive (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, x n + x (n + 2) - x (n + 1) = x m

-- Part 1
def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => 3 * a n

theorem not_regressive_a : ¬ IsRegressive a := by sorry

-- Part 2
theorem regressive_increasing_is_arithmetic (b : ℕ → ℝ) 
  (h_regressive : IsRegressive b) (h_increasing : ∀ n : ℕ, b n < b (n + 1)) :
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) - b n = d := by sorry

end NUMINAMATH_CALUDE_not_regressive_a_regressive_increasing_is_arithmetic_l3284_328404


namespace NUMINAMATH_CALUDE_elias_bananas_l3284_328449

/-- The number of bananas in a dozen -/
def dozen : ℕ := 12

/-- The number of bananas Elias ate -/
def eaten : ℕ := 1

/-- The number of bananas left after Elias ate some -/
def bananas_left (initial : ℕ) (eaten : ℕ) : ℕ := initial - eaten

/-- Theorem: If Elias bought a dozen bananas and ate 1, he has 11 left -/
theorem elias_bananas : bananas_left dozen eaten = 11 := by
  sorry

end NUMINAMATH_CALUDE_elias_bananas_l3284_328449


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l3284_328494

/-- Given three consecutive even numbers whose sum is 246, the first number is 80 -/
theorem consecutive_even_numbers_sum (n : ℤ) : 
  (∃ (a b c : ℤ), a = n ∧ b = n + 2 ∧ c = n + 4 ∧ a + b + c = 246 ∧ Even a ∧ Even b ∧ Even c) → 
  n = 80 := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l3284_328494


namespace NUMINAMATH_CALUDE_aquarium_count_l3284_328413

theorem aquarium_count (total_animals : ℕ) (animals_per_aquarium : ℕ) 
  (h1 : total_animals = 40) 
  (h2 : animals_per_aquarium = 2) 
  (h3 : animals_per_aquarium > 0) : 
  total_animals / animals_per_aquarium = 20 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_count_l3284_328413


namespace NUMINAMATH_CALUDE_geometric_series_sum_specific_l3284_328471

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_specific : 
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 8
  geometric_series_sum a r n = 65535/196608 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_specific_l3284_328471


namespace NUMINAMATH_CALUDE_dandelion_puff_distribution_l3284_328403

theorem dandelion_puff_distribution (total : ℕ) (given_away : ℕ) (friends : ℕ) 
  (h1 : total = 85)
  (h2 : given_away = 36)
  (h3 : friends = 5)
  (h4 : given_away < total) : 
  (total - given_away) / friends = (total - given_away) / (total - given_away) / friends :=
by sorry

end NUMINAMATH_CALUDE_dandelion_puff_distribution_l3284_328403


namespace NUMINAMATH_CALUDE_coefficient_of_monomial_l3284_328421

/-- The coefficient of a monomial is the numerical factor multiplied by the variables. -/
def coefficient (m : ℝ) (x y : ℝ) : ℝ := m

/-- For the monomial -2π * x^2 * y, prove that its coefficient is -2π. -/
theorem coefficient_of_monomial :
  coefficient (-2 * Real.pi) (x ^ 2) y = -2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_monomial_l3284_328421


namespace NUMINAMATH_CALUDE_fraction_zero_l3284_328483

theorem fraction_zero (a : ℝ) : (a^2 - 1) / (a + 1) = 0 ↔ a = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_l3284_328483


namespace NUMINAMATH_CALUDE_car_speed_problem_l3284_328475

theorem car_speed_problem (distance : ℝ) (original_time : ℝ) (new_time_factor : ℝ) :
  distance = 450 ∧ 
  original_time = 6 ∧ 
  new_time_factor = 3/2 →
  (distance / (original_time * new_time_factor)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3284_328475


namespace NUMINAMATH_CALUDE_octal_addition_theorem_l3284_328466

/-- Represents a number in base 8 --/
def OctalNumber := List Nat

/-- Converts a natural number to its octal representation --/
def toOctal (n : Nat) : OctalNumber := sorry

/-- Converts an octal number to its decimal representation --/
def fromOctal (o : OctalNumber) : Nat := sorry

/-- Adds two octal numbers --/
def addOctal (a b : OctalNumber) : OctalNumber := sorry

theorem octal_addition_theorem :
  let a := [5, 3, 2, 6]
  let b := [1, 4, 7, 3]
  addOctal a b = [7, 0, 4, 3] := by sorry

end NUMINAMATH_CALUDE_octal_addition_theorem_l3284_328466


namespace NUMINAMATH_CALUDE_correct_good_carrots_l3284_328436

/-- The number of good carrots given the number of carrots picked by Haley and her mother, and the number of bad carrots. -/
def goodCarrots (haleyCarrots motherCarrots badCarrots : ℕ) : ℕ :=
  haleyCarrots + motherCarrots - badCarrots

/-- Theorem stating that the number of good carrots is 64 given the specific conditions. -/
theorem correct_good_carrots :
  goodCarrots 39 38 13 = 64 := by
  sorry

end NUMINAMATH_CALUDE_correct_good_carrots_l3284_328436


namespace NUMINAMATH_CALUDE_initial_weight_calculation_calvins_initial_weight_l3284_328455

/-- Calculates the initial weight of a person given their weight loss rate and final weight --/
theorem initial_weight_calculation 
  (weight_loss_per_month : ℕ) 
  (months : ℕ) 
  (final_weight : ℕ) : ℕ :=
  let total_weight_loss := weight_loss_per_month * months
  final_weight + total_weight_loss

/-- Proves that given the conditions, the initial weight was 250 pounds --/
theorem calvins_initial_weight :
  let weight_loss_per_month : ℕ := 8
  let months : ℕ := 12
  let final_weight : ℕ := 154
  initial_weight_calculation weight_loss_per_month months final_weight = 250 := by
  sorry

end NUMINAMATH_CALUDE_initial_weight_calculation_calvins_initial_weight_l3284_328455


namespace NUMINAMATH_CALUDE_intersection_when_a_is_4_subset_condition_l3284_328446

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 8*x + 7 < 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*x - a^2 - 2*a < 0}

-- Theorem 1: When a = 4, A ∩ B = (1, 6)
theorem intersection_when_a_is_4 : A ∩ (B 4) = Set.Ioo 1 6 := by sorry

-- Theorem 2: A ⊆ B if and only if a ∈ (-∞, -7] ∪ [5, +∞)
theorem subset_condition (a : ℝ) : A ⊆ B a ↔ a ≤ -7 ∨ a ≥ 5 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_4_subset_condition_l3284_328446
