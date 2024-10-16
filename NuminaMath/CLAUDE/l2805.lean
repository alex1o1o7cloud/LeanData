import Mathlib

namespace NUMINAMATH_CALUDE_angle_D_value_l2805_280529

theorem angle_D_value (A B C D : ℝ) 
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 2 * D - 10) :
  D = 70 := by
sorry

end NUMINAMATH_CALUDE_angle_D_value_l2805_280529


namespace NUMINAMATH_CALUDE_four_stamps_cost_l2805_280503

/-- The cost of a single stamp in dollars -/
def stamp_cost : ℚ := 34/100

/-- The cost of two stamps in dollars -/
def two_stamps_cost : ℚ := 68/100

/-- The cost of three stamps in dollars -/
def three_stamps_cost : ℚ := 102/100

/-- Proves that the cost of four stamps is $1.36 -/
theorem four_stamps_cost :
  stamp_cost * 4 = 136/100 :=
by sorry

end NUMINAMATH_CALUDE_four_stamps_cost_l2805_280503


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l2805_280521

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Checks if two triangles are similar -/
def Triangle.isSimilar (t1 t2 : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t2.a = k * t1.a ∧ t2.b = k * t1.b ∧ t2.c = k * t1.c

theorem similar_triangle_perimeter (t1 t2 : Triangle) :
  t1.isIsosceles ∧
  t1.a = 18 ∧ t1.b = 18 ∧ t1.c = 12 ∧
  t2.isSimilar t1 ∧
  min t2.a (min t2.b t2.c) = 30 →
  t2.perimeter = 120 := by
sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l2805_280521


namespace NUMINAMATH_CALUDE_smallest_x_value_l2805_280535

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = y / (215 + x)) : 
  ∀ z : ℕ+, z < x → (3 : ℚ) / 4 ≠ y / (215 + z) :=
sorry

end NUMINAMATH_CALUDE_smallest_x_value_l2805_280535


namespace NUMINAMATH_CALUDE_red_paint_amount_l2805_280560

/-- Given a paint mixture with a ratio of red:green:white as 4:3:5,
    and using 15 quarts of white paint, prove that the amount of
    red paint required is 12 quarts. -/
theorem red_paint_amount (red green white : ℚ) : 
  red / white = 4 / 5 →
  green / white = 3 / 5 →
  white = 15 →
  red = 12 := by
  sorry

end NUMINAMATH_CALUDE_red_paint_amount_l2805_280560


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2805_280537

-- Problem 1
theorem problem_1 : (1) - 2^2 + (π - 3)^0 + 0.5^(-1) = -1 := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) : (x - 2*y) * (x^2 + 2*x*y + 4*y^2) = x^3 - 8*y^3 := by sorry

-- Problem 3
theorem problem_3 (a : ℝ) : a * a^2 * a^3 + (-2*a^3)^2 - a^8 / a^2 = 4*a^6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2805_280537


namespace NUMINAMATH_CALUDE_max_a_value_l2805_280565

theorem max_a_value (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 + 2 + |x^3 - 2*x| ≥ a*x) → 
  a ≤ 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_max_a_value_l2805_280565


namespace NUMINAMATH_CALUDE_rachel_songs_theorem_l2805_280520

/-- The number of songs in each of Rachel's albums -/
def album_songs : List Nat := [5, 6, 8, 10, 12, 14, 16, 7, 9, 11, 13, 15, 17, 4, 6, 8, 10, 12, 14, 3]

/-- The total number of songs Rachel bought -/
def total_songs : Nat := album_songs.sum

theorem rachel_songs_theorem : total_songs = 200 := by
  sorry

end NUMINAMATH_CALUDE_rachel_songs_theorem_l2805_280520


namespace NUMINAMATH_CALUDE_coffee_payment_dimes_l2805_280562

/-- Represents the number of coins of each type used in the payment -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- The total value of the coins in cents -/
def totalValue (c : CoinCount) : ℕ :=
  c.pennies + 5 * c.nickels + 10 * c.dimes

/-- The total number of coins -/
def totalCoins (c : CoinCount) : ℕ :=
  c.pennies + c.nickels + c.dimes

theorem coffee_payment_dimes :
  ∃ (c : CoinCount),
    totalValue c = 200 ∧
    totalCoins c = 50 ∧
    c.dimes = 14 :=
by sorry

end NUMINAMATH_CALUDE_coffee_payment_dimes_l2805_280562


namespace NUMINAMATH_CALUDE_equation_solution_l2805_280554

theorem equation_solution : ∃ (z₁ z₂ : ℂ), 
  z₁ = -1 + Complex.I ∧ 
  z₂ = -1 - Complex.I ∧ 
  (∀ x : ℂ, x ≠ -2 → -x^3 = (4*x + 2)/(x + 2) ↔ (x = z₁ ∨ x = z₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2805_280554


namespace NUMINAMATH_CALUDE_value_of_x_l2805_280517

theorem value_of_x (x y z : ℝ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 96) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l2805_280517


namespace NUMINAMATH_CALUDE_inequality_proof_l2805_280573

theorem inequality_proof (K x : ℝ) (hK : K > 1) (hx_pos : x > 0) (hx_bound : x < π / K) :
  (Real.sin (K * x) / Real.sin x) < K * Real.exp (-(K^2 - 1) * x^2 / 6) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2805_280573


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l2805_280541

/-- Represents a business partnership --/
structure Partnership where
  investment_A : ℕ
  investment_B : ℕ
  investment_C : ℕ
  investment_D : ℕ
  profit_ratio_A : ℕ
  profit_ratio_B : ℕ
  profit_ratio_C : ℕ
  profit_ratio_D : ℕ
  C_profit_share : ℕ

/-- Calculates the total profit of a partnership --/
def calculate_total_profit (p : Partnership) : ℕ :=
  let x := p.C_profit_share / p.profit_ratio_C
  x * (p.profit_ratio_A + p.profit_ratio_B + p.profit_ratio_C + p.profit_ratio_D)

/-- Theorem stating that for the given partnership, the total profit is 144000 --/
theorem partnership_profit_calculation (p : Partnership)
  (h1 : p.investment_A = 27000)
  (h2 : p.investment_B = 72000)
  (h3 : p.investment_C = 81000)
  (h4 : p.investment_D = 63000)
  (h5 : p.profit_ratio_A = 2)
  (h6 : p.profit_ratio_B = 3)
  (h7 : p.profit_ratio_C = 4)
  (h8 : p.profit_ratio_D = 3)
  (h9 : p.C_profit_share = 48000) :
  calculate_total_profit p = 144000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_calculation_l2805_280541


namespace NUMINAMATH_CALUDE_equation_solution_l2805_280546

theorem equation_solution :
  ∃ x : ℚ, x ≠ 1 ∧ (x^2 - x + 2) / (x - 1) = x + 3 ∧ x = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2805_280546


namespace NUMINAMATH_CALUDE_expand_expression_l2805_280550

theorem expand_expression (x y : ℝ) : (x + 3) * (4 * x - 5 * y) = 4 * x^2 - 5 * x * y + 12 * x - 15 * y := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2805_280550


namespace NUMINAMATH_CALUDE_weight_estimate_for_178cm_l2805_280557

/-- Regression equation for weight based on height -/
def weight_regression (height : ℝ) : ℝ := 0.72 * height - 58.5

/-- The problem statement -/
theorem weight_estimate_for_178cm :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |weight_regression 178 - 70| < ε :=
sorry

end NUMINAMATH_CALUDE_weight_estimate_for_178cm_l2805_280557


namespace NUMINAMATH_CALUDE_unique_factorization_1386_l2805_280552

/-- Two-digit numbers are natural numbers between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A factorization of 1386 into two two-digit numbers -/
structure Factorization :=
  (a b : ℕ)
  (h1 : TwoDigitNumber a)
  (h2 : TwoDigitNumber b)
  (h3 : a * b = 1386)

/-- Two factorizations are considered the same if they have the same factors (in any order) -/
def Factorization.equiv (f g : Factorization) : Prop :=
  (f.a = g.a ∧ f.b = g.b) ∨ (f.a = g.b ∧ f.b = g.a)

/-- The main theorem stating that there is exactly one factorization of 1386 into two-digit numbers -/
theorem unique_factorization_1386 : 
  ∃! (f : Factorization), True :=
sorry

end NUMINAMATH_CALUDE_unique_factorization_1386_l2805_280552


namespace NUMINAMATH_CALUDE_melanie_grew_more_turnips_l2805_280500

/-- The number of turnips Melanie grew -/
def melanie_turnips : ℕ := 139

/-- The number of turnips Benny grew -/
def benny_turnips : ℕ := 113

/-- The difference in turnips between Melanie and Benny -/
def turnip_difference : ℕ := melanie_turnips - benny_turnips

theorem melanie_grew_more_turnips : turnip_difference = 26 := by
  sorry

end NUMINAMATH_CALUDE_melanie_grew_more_turnips_l2805_280500


namespace NUMINAMATH_CALUDE_dad_steps_l2805_280534

/-- Represents the number of steps taken by each person -/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- The ratio of steps between dad and Masha -/
def dad_masha_ratio (s : Steps) : Prop :=
  3 * s.masha = 5 * s.dad

/-- The ratio of steps between Masha and Yasha -/
def masha_yasha_ratio (s : Steps) : Prop :=
  3 * s.yasha = 5 * s.masha

/-- The total number of steps taken by Masha and Yasha -/
def masha_yasha_total (s : Steps) : Prop :=
  s.masha + s.yasha = 400

theorem dad_steps (s : Steps) 
  (h1 : dad_masha_ratio s)
  (h2 : masha_yasha_ratio s)
  (h3 : masha_yasha_total s) :
  s.dad = 90 := by
  sorry


end NUMINAMATH_CALUDE_dad_steps_l2805_280534


namespace NUMINAMATH_CALUDE_max_F_value_intersection_property_l2805_280531

noncomputable section

variables (a b : ℝ) (x x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := (Real.log x) / x

def g (a b x : ℝ) : ℝ := (1/2) * a * x + b

def F (a b x : ℝ) : ℝ := f x - g a b x

theorem max_F_value (h1 : a = 2) (h2 : b = -3) :
  ∃ (m : ℝ), m = 2 ∧ ∀ x > 0, F a b x ≤ m :=
sorry

theorem intersection_property (h1 : x₁ > 0) (h2 : x₂ > 0) (h3 : x₁ ≠ x₂)
  (h4 : f x₁ = g a b x₁) (h5 : f x₂ = g a b x₂) :
  (x₁ + x₂) * g a b (x₁ + x₂) > 2 :=
sorry

end NUMINAMATH_CALUDE_max_F_value_intersection_property_l2805_280531


namespace NUMINAMATH_CALUDE_vershoks_in_arshin_l2805_280526

/-- The number of vershoks in one arshin -/
def vershoks_per_arshin : ℕ := sorry

/-- Length of a plank in arshins -/
def plank_length : ℕ := 6

/-- Width of a plank in vershoks -/
def plank_width : ℕ := 6

/-- Side length of the room in arshins -/
def room_side : ℕ := 12

/-- Number of planks needed to cover the floor -/
def num_planks : ℕ := 64

theorem vershoks_in_arshin : 
  vershoks_per_arshin = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_vershoks_in_arshin_l2805_280526


namespace NUMINAMATH_CALUDE_rectangle_perimeter_from_square_l2805_280593

/-- A rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- The perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.width + r.length)

/-- A square formed by 5 identical rectangles -/
structure SquareFromRectangles where
  base_rectangle : Rectangle
  side_length : ℝ
  h_side_length : side_length = 5 * base_rectangle.width

/-- The perimeter of the square formed by rectangles -/
def SquareFromRectangles.perimeter (s : SquareFromRectangles) : ℝ :=
  4 * s.side_length

theorem rectangle_perimeter_from_square (s : SquareFromRectangles) 
    (h_perimeter_diff : s.perimeter = s.base_rectangle.perimeter + 10) :
    s.base_rectangle.perimeter = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_from_square_l2805_280593


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2805_280571

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, (1/2 : ℝ) * x^2 - 2*m*x + 4*m + 1 = 0 ∧ 
   ∀ y : ℝ, (1/2 : ℝ) * y^2 - 2*m*y + 4*m + 1 = 0 → y = x) → 
  m^2 - 2*m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2805_280571


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l2805_280519

-- Define the set of people
inductive Person : Type
| Alice : Person
| Bob : Person
| Carla : Person
| Derek : Person
| Eric : Person

-- Define a seating arrangement as a function from Person to ℕ (seat number)
def SeatingArrangement := Person → Fin 5

-- Define the adjacency relation for a circular table
def adjacent (s : SeatingArrangement) (p1 p2 : Person) : Prop :=
  (s p1 - s p2 = 1) ∨ (s p2 - s p1 = 1) ∨ (s p1 = 4 ∧ s p2 = 0) ∨ (s p1 = 0 ∧ s p2 = 4)

-- Define the seating restrictions
def validArrangement (s : SeatingArrangement) : Prop :=
  (¬ adjacent s Person.Alice Person.Bob) ∧
  (¬ adjacent s Person.Alice Person.Carla) ∧
  (¬ adjacent s Person.Derek Person.Eric) ∧
  (¬ adjacent s Person.Carla Person.Derek) ∧
  Function.Injective s

-- Theorem stating that no valid seating arrangement exists
theorem no_valid_arrangement : ¬ ∃ s : SeatingArrangement, validArrangement s := by
  sorry


end NUMINAMATH_CALUDE_no_valid_arrangement_l2805_280519


namespace NUMINAMATH_CALUDE_triangle_properties_l2805_280558

noncomputable section

-- Define the triangle ABC
variable (A B C : Real) -- Angles
variable (a b c : Real) -- Side lengths

-- Define the conditions
axiom angle_side_relation : 2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c
axiom c_value : c = Real.sqrt 7
axiom triangle_area : 1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2

-- Theorem to prove
theorem triangle_properties : C = π/3 ∧ a + b + c = 5 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2805_280558


namespace NUMINAMATH_CALUDE_probability_specific_pair_from_six_l2805_280589

/-- The probability of selecting a specific pair when choosing 2 from 6 -/
theorem probability_specific_pair_from_six (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 2) :
  (1 : ℚ) / (n.choose k) = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_specific_pair_from_six_l2805_280589


namespace NUMINAMATH_CALUDE_missing_number_is_36_l2805_280568

def known_numbers : List ℕ := [1, 22, 23, 24, 25, 27, 2]

theorem missing_number_is_36 (mean : ℚ) (total_count : ℕ) (h_mean : mean = 20) (h_count : total_count = 8) :
  ∃ x : ℕ, (x :: known_numbers).sum / total_count = mean :=
sorry

end NUMINAMATH_CALUDE_missing_number_is_36_l2805_280568


namespace NUMINAMATH_CALUDE_ezekiel_painted_faces_l2805_280549

/-- The number of faces on a cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The number of cuboids painted -/
def num_cuboids : ℕ := 5

/-- The total number of faces painted by Ezekiel -/
def total_faces_painted : ℕ := faces_per_cuboid * num_cuboids

theorem ezekiel_painted_faces :
  total_faces_painted = 30 :=
by sorry

end NUMINAMATH_CALUDE_ezekiel_painted_faces_l2805_280549


namespace NUMINAMATH_CALUDE_largest_number_in_sample_l2805_280523

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  total_items : ℕ
  first_number : ℕ
  second_number : ℕ
  sample_size : ℕ

/-- Calculates the largest number in a systematic sample -/
def largest_sample_number (s : SystematicSample) : ℕ :=
  s.first_number + (s.sample_size - 1) * (s.second_number - s.first_number)

/-- Theorem stating the largest number in the given systematic sample -/
theorem largest_number_in_sample :
  let s : SystematicSample := {
    total_items := 400,
    first_number := 8,
    second_number := 33,
    sample_size := 16
  }
  largest_sample_number s = 383 := by sorry

end NUMINAMATH_CALUDE_largest_number_in_sample_l2805_280523


namespace NUMINAMATH_CALUDE_sqrt_21_times_sqrt_7_minus_sqrt_3_l2805_280599

theorem sqrt_21_times_sqrt_7_minus_sqrt_3 :
  Real.sqrt 21 * Real.sqrt 7 - Real.sqrt 3 = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_21_times_sqrt_7_minus_sqrt_3_l2805_280599


namespace NUMINAMATH_CALUDE_square_area_ratio_l2805_280525

theorem square_area_ratio (y : ℝ) (h : y > 0) : 
  (y^2) / ((4*y)^2) = 1/16 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2805_280525


namespace NUMINAMATH_CALUDE_sum_of_integers_l2805_280516

theorem sum_of_integers (a b : ℕ+) (h1 : a.val^2 - b.val^2 = 44) (h2 : a.val * b.val = 120) : 
  a.val + b.val = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2805_280516


namespace NUMINAMATH_CALUDE_multiplication_fraction_equality_l2805_280530

theorem multiplication_fraction_equality : 7 * (1 / 21) * 42 = 14 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_fraction_equality_l2805_280530


namespace NUMINAMATH_CALUDE_unique_intersection_point_l2805_280591

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 5*x^2 + 12*x + 20

-- State the theorem
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = f p.2 ∧ p.2 = f p.1 ∧ p = (-5, -5) := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l2805_280591


namespace NUMINAMATH_CALUDE_cat_bird_hunting_l2805_280590

theorem cat_bird_hunting (day_catch : ℕ) (night_catch : ℕ) : 
  day_catch = 8 → night_catch = 2 * day_catch → day_catch + night_catch = 24 := by
  sorry

end NUMINAMATH_CALUDE_cat_bird_hunting_l2805_280590


namespace NUMINAMATH_CALUDE_part1_correct_part2_correct_l2805_280581

-- Define point P as a function of m
def P (m : ℝ) : ℝ × ℝ := (-3*m - 4, 2 + m)

-- Define point Q
def Q : ℝ × ℝ := (5, 8)

-- Theorem for part 1
theorem part1_correct :
  ∃ m : ℝ, P m = (-10, 4) ∧ (P m).2 = 4 := by sorry

-- Theorem for part 2
theorem part2_correct :
  ∃ m : ℝ, P m = (5, -1) ∧ (P m).1 = Q.1 := by sorry

end NUMINAMATH_CALUDE_part1_correct_part2_correct_l2805_280581


namespace NUMINAMATH_CALUDE_least_product_of_distinct_primes_above_50_l2805_280595

theorem least_product_of_distinct_primes_above_50 :
  ∃ (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    p > 50 ∧ q > 50 ∧ 
    p ≠ q ∧
    p * q = 3127 ∧
    ∀ (r s : ℕ), Prime r → Prime s → r > 50 → s > 50 → r ≠ s → r * s ≥ p * q :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_distinct_primes_above_50_l2805_280595


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2805_280578

theorem negation_of_existence (f : ℝ → Prop) : 
  (¬ ∃ x, f x) ↔ (∀ x, ¬ f x) :=
by sorry

theorem negation_of_quadratic_inequality : 
  (¬ ∃ x : ℝ, x^2 - 2*x - 3 < 0) ↔ (∀ x : ℝ, x^2 - 2*x - 3 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2805_280578


namespace NUMINAMATH_CALUDE_triangle_area_with_median_l2805_280506

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the median AM
def median (t : Triangle) : ℝ × ℝ := sorry

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

theorem triangle_area_with_median :
  ∀ (t : Triangle),
    length (t.A) (t.B) = 9 →
    length (t.A) (t.C) = 17 →
    length (t.A) (median t) = 12 →
    area t = 20 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_median_l2805_280506


namespace NUMINAMATH_CALUDE_sum_of_ages_l2805_280518

theorem sum_of_ages (age1 age2 : ℕ) : 
  age2 = age1 + 1 → age1 = 13 → age2 = 14 → age1 + age2 = 27 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2805_280518


namespace NUMINAMATH_CALUDE_smallest_prime_digit_sum_23_l2805_280536

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- Theorem stating that 599 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_digit_sum_23 :
  (is_prime 599) ∧ 
  (digit_sum 599 = 23) ∧ 
  (∀ n : ℕ, n < 599 → ¬(is_prime n ∧ digit_sum n = 23)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_digit_sum_23_l2805_280536


namespace NUMINAMATH_CALUDE_square_sum_of_special_integers_l2805_280545

theorem square_sum_of_special_integers (x y : ℕ+) 
  (h1 : x * y + x + y = 71)
  (h2 : x^2 * y + x * y^2 = 880) : 
  x^2 + y^2 = 146 := by sorry

end NUMINAMATH_CALUDE_square_sum_of_special_integers_l2805_280545


namespace NUMINAMATH_CALUDE_not_divisible_by_four_l2805_280507

theorem not_divisible_by_four (n : ℤ) : ¬(4 ∣ (1 + n + n^2 + n^3 + n^4)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_four_l2805_280507


namespace NUMINAMATH_CALUDE_complement_determines_interval_l2805_280528

-- Define the set A
def A (a b : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ b }

-- Define the complement of A
def C_U_A : Set ℝ := { x | x > 4 ∨ x < 3 }

-- Theorem statement
theorem complement_determines_interval :
  ∃ (a b : ℝ), A a b = (C_U_A)ᶜ ∧ a = 3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complement_determines_interval_l2805_280528


namespace NUMINAMATH_CALUDE_circle_theorem_l2805_280539

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y - 28 = 0

-- Define the line on which the center of the required circle lies
def centerLine (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the equation of the required circle
def requiredCircle (x y : ℝ) : Prop := x^2 + y^2 - x + 7*y - 32 = 0

-- Theorem statement
theorem circle_theorem :
  ∀ (x y : ℝ),
    (circle1 x y ∧ circle2 x y → requiredCircle x y) ∧
    (∃ (h k : ℝ), centerLine h k ∧ 
      ∀ (x y : ℝ), requiredCircle x y ↔ (x - h)^2 + (y - k)^2 = (h - x)^2 + (k - y)^2) :=
sorry

end NUMINAMATH_CALUDE_circle_theorem_l2805_280539


namespace NUMINAMATH_CALUDE_total_age_is_42_l2805_280586

/-- Given three people a, b, and c, where a is two years older than b, 
    b is twice as old as c, and b is 16 years old, 
    prove that the total of their ages is 42 years. -/
theorem total_age_is_42 (a b c : ℕ) : 
  a = b + 2 → b = 2 * c → b = 16 → a + b + c = 42 :=
by sorry

end NUMINAMATH_CALUDE_total_age_is_42_l2805_280586


namespace NUMINAMATH_CALUDE_matrix_A_nonsingular_l2805_280567

/-- Prove that the matrix A defined by the given conditions is nonsingular -/
theorem matrix_A_nonsingular 
  (k : ℕ) 
  (i j : Fin k → ℕ)
  (h_i : ∀ m n, m < n → i m < i n)
  (h_j : ∀ m n, m < n → j m < j n)
  (A : Matrix (Fin k) (Fin k) ℚ)
  (h_A : ∀ r s, A r s = (Nat.choose (i r + j s) (i r) : ℚ)) :
  Matrix.det A ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_A_nonsingular_l2805_280567


namespace NUMINAMATH_CALUDE_arcade_time_calculation_l2805_280509

/-- The number of hours spent at the arcade given the rate and total spend -/
def arcade_time (rate : ℚ) (interval : ℚ) (total_spend : ℚ) : ℚ :=
  (total_spend / rate * interval) / 60

/-- Theorem stating that given a rate of $0.50 per 6 minutes and a total spend of $15, 
    the time spent at the arcade is 3 hours -/
theorem arcade_time_calculation :
  arcade_time (1/2) 6 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arcade_time_calculation_l2805_280509


namespace NUMINAMATH_CALUDE_arithmetic_geometric_means_sum_l2805_280570

/-- Given real numbers a, b, c in geometric progression and non-zero real numbers x, y
    that are arithmetic means of a, b and b, c respectively, prove that a/x + b/y = 2 -/
theorem arithmetic_geometric_means_sum (a b c x y : ℝ) 
  (hgp : b^2 = a*c)  -- geometric progression condition
  (hx : x ≠ 0)       -- x is non-zero
  (hy : y ≠ 0)       -- y is non-zero
  (hax : 2*x = a + b)  -- x is arithmetic mean of a and b
  (hby : 2*y = b + c)  -- y is arithmetic mean of b and c
  : a/x + b/y = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_means_sum_l2805_280570


namespace NUMINAMATH_CALUDE_complement_of_N_in_M_l2805_280574

def M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℤ := {1, 2}

theorem complement_of_N_in_M :
  (M \ N) = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_N_in_M_l2805_280574


namespace NUMINAMATH_CALUDE_cube_side_length_l2805_280522

theorem cube_side_length (volume_submerged_min : ℝ) (volume_submerged_max : ℝ)
  (density_ratio : ℝ) (volume_above_min : ℝ) (volume_above_max : ℝ) :
  volume_submerged_min = 0.58 →
  volume_submerged_max = 0.87 →
  density_ratio = 0.95 →
  volume_above_min = 10 →
  volume_above_max = 29 →
  ∃ (s : ℕ), s = 4 ∧
    (volume_submerged_min * s^3 ≤ density_ratio * s^3) ∧
    (density_ratio * s^3 ≤ volume_submerged_max * s^3) ∧
    (volume_above_min ≤ s^3 - volume_submerged_max * s^3) ∧
    (s^3 - volume_submerged_min * s^3 ≤ volume_above_max) :=
by sorry

end NUMINAMATH_CALUDE_cube_side_length_l2805_280522


namespace NUMINAMATH_CALUDE_larger_cross_section_distance_l2805_280584

/-- Represents a right octagonal pyramid -/
structure RightOctagonalPyramid where
  -- We don't need to define the full structure, just what's necessary for the problem

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  area : ℝ
  distance_from_apex : ℝ

theorem larger_cross_section_distance
  (pyramid : RightOctagonalPyramid)
  (cs1 cs2 : CrossSection)
  (h_area1 : cs1.area = 256 * Real.sqrt 2)
  (h_area2 : cs2.area = 576 * Real.sqrt 2)
  (h_distance : cs2.distance_from_apex - cs1.distance_from_apex = 10)
  (h_parallel : True)  -- Assuming parallel, but not used in the proof
  (h_larger : cs2.area > cs1.area) :
  cs2.distance_from_apex = 30 := by
sorry


end NUMINAMATH_CALUDE_larger_cross_section_distance_l2805_280584


namespace NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l2805_280511

theorem greatest_integer_for_all_real_domain (b : ℤ) : 
  (∀ x : ℝ, (x^2 + b*x + 9 : ℝ) ≠ 0) ↔ b ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l2805_280511


namespace NUMINAMATH_CALUDE_square_land_area_l2805_280588

/-- The area of a square land plot with side length 20 units is 400 square units. -/
theorem square_land_area (side_length : ℝ) (h : side_length = 20) : side_length ^ 2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_square_land_area_l2805_280588


namespace NUMINAMATH_CALUDE_roman_numeral_calculation_l2805_280542

/-- Roman numeral values -/
def I : ℕ := 1
def V : ℕ := 5
def X : ℕ := 10
def L : ℕ := 50
def C : ℕ := 100
def D : ℕ := 500
def M : ℕ := 1000

/-- The theorem to prove -/
theorem roman_numeral_calculation : 2 * M + 5 * L + 7 * X + 9 * I = 2329 := by
  sorry

end NUMINAMATH_CALUDE_roman_numeral_calculation_l2805_280542


namespace NUMINAMATH_CALUDE_function_derivative_values_l2805_280569

theorem function_derivative_values (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = a * x^2 - b * Real.sin x) →
  (deriv f) 0 = 1 →
  (deriv f) (π/3) = 1/2 →
  a = 3 / (2 * π) ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_derivative_values_l2805_280569


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l2805_280504

theorem pure_imaginary_condition (x : ℝ) : 
  (((x^2 - 1) : ℂ) + (x - 1) * Complex.I).re = 0 ∧ 
  (((x^2 - 1) : ℂ) + (x - 1) * Complex.I).im ≠ 0 → 
  x = -1 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l2805_280504


namespace NUMINAMATH_CALUDE_binomial_1293_1_l2805_280580

theorem binomial_1293_1 : Nat.choose 1293 1 = 1293 := by
  sorry

end NUMINAMATH_CALUDE_binomial_1293_1_l2805_280580


namespace NUMINAMATH_CALUDE_biscuit_boxes_combination_exists_l2805_280502

theorem biscuit_boxes_combination_exists : ∃ (a b c d e : ℕ), 16*a + 17*b + 23*c + 39*d + 40*e = 100 := by
  sorry

end NUMINAMATH_CALUDE_biscuit_boxes_combination_exists_l2805_280502


namespace NUMINAMATH_CALUDE_power_of_product_squared_l2805_280532

theorem power_of_product_squared (a b : ℝ) : (-a^2 * b^3)^2 = a^4 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_squared_l2805_280532


namespace NUMINAMATH_CALUDE_candle_height_ratio_time_l2805_280598

/-- Represents a candle with its initial height and burning time. -/
structure Candle where
  initial_height : ℝ
  burning_time : ℝ

/-- The problem setup -/
def candle_problem : Prop :=
  let candle_a : Candle := { initial_height := 12, burning_time := 6 }
  let candle_b : Candle := { initial_height := 15, burning_time := 5 }
  let burn_rate (c : Candle) : ℝ := c.initial_height / c.burning_time
  let height_at_time (c : Candle) (t : ℝ) : ℝ := c.initial_height - (burn_rate c) * t
  ∃ t : ℝ, t > 0 ∧ height_at_time candle_a t = (1/3) * height_at_time candle_b t ∧ t = 7

/-- The theorem to be proved -/
theorem candle_height_ratio_time : candle_problem := by
  sorry

end NUMINAMATH_CALUDE_candle_height_ratio_time_l2805_280598


namespace NUMINAMATH_CALUDE_charles_total_money_l2805_280594

-- Define the value of each coin type in cents
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

-- Define the number of coins Charles found on his way to school
def found_pennies : ℕ := 6
def found_nickels : ℕ := 4
def found_dimes : ℕ := 3

-- Define the number of coins Charles already had at home
def home_nickels : ℕ := 3
def home_dimes : ℕ := 2
def home_quarters : ℕ := 1

-- Calculate the total value in cents
def total_cents : ℕ :=
  found_pennies * penny_value +
  (found_nickels + home_nickels) * nickel_value +
  (found_dimes + home_dimes) * dime_value +
  home_quarters * quarter_value

-- Theorem to prove
theorem charles_total_money :
  total_cents = 116 := by sorry

end NUMINAMATH_CALUDE_charles_total_money_l2805_280594


namespace NUMINAMATH_CALUDE_cuboid_edge_length_l2805_280551

/-- The surface area of a cuboid given its three edge lengths -/
def cuboidSurfaceArea (x y z : ℝ) : ℝ := 2 * (x * y + x * z + y * z)

/-- Theorem stating that if a cuboid with edges x, 5, and 6 has surface area 148, then x = 4 -/
theorem cuboid_edge_length (x : ℝ) :
  cuboidSurfaceArea x 5 6 = 148 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_edge_length_l2805_280551


namespace NUMINAMATH_CALUDE_range_of_a_given_solution_exact_range_of_a_l2805_280579

-- Define the inequality as a function of x and a
def inequality (x a : ℝ) : Prop := 2 * x^2 + a * x - a^2 > 0

-- State the theorem
theorem range_of_a_given_solution : 
  ∀ a : ℝ, inequality 2 a → -2 < a ∧ a < 4 :=
by
  sorry

-- Define the range of a
def range_of_a : Set ℝ := { a : ℝ | -2 < a ∧ a < 4 }

-- State that this is the exact range
theorem exact_range_of_a : 
  ∀ a : ℝ, a ∈ range_of_a ↔ inequality 2 a :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_given_solution_exact_range_of_a_l2805_280579


namespace NUMINAMATH_CALUDE_combined_box_weight_l2805_280543

def box1_weight : ℕ := 2
def box2_weight : ℕ := 11
def box3_weight : ℕ := 5

theorem combined_box_weight :
  box1_weight + box2_weight + box3_weight = 18 := by
  sorry

end NUMINAMATH_CALUDE_combined_box_weight_l2805_280543


namespace NUMINAMATH_CALUDE_quadratic_roots_bounds_l2805_280524

theorem quadratic_roots_bounds (a b : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + a*x + b = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ < x₂ →
  (∀ x, -1 < x ∧ x < 1 → x^2 + a*x + b < 0) →
  -1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_bounds_l2805_280524


namespace NUMINAMATH_CALUDE_olaf_initial_cars_l2805_280592

/-- The number of toy cars Olaf's uncle gave him -/
def uncle_cars : ℕ := 5

/-- The number of toy cars Olaf's grandpa gave him -/
def grandpa_cars : ℕ := 2 * uncle_cars

/-- The number of toy cars Olaf's dad gave him -/
def dad_cars : ℕ := 10

/-- The number of toy cars Olaf's mum gave him -/
def mum_cars : ℕ := dad_cars + 5

/-- The number of toy cars Olaf's auntie gave him -/
def auntie_cars : ℕ := 6

/-- The total number of toy cars Olaf has after receiving gifts -/
def total_cars : ℕ := 196

/-- The number of toy cars Olaf had initially -/
def initial_cars : ℕ := total_cars - (grandpa_cars + dad_cars + mum_cars + auntie_cars + uncle_cars)

theorem olaf_initial_cars : initial_cars = 150 := by
  sorry

end NUMINAMATH_CALUDE_olaf_initial_cars_l2805_280592


namespace NUMINAMATH_CALUDE_lg2_bounds_l2805_280564

theorem lg2_bounds :
  (10 : ℝ)^3 = 1000 ∧ (10 : ℝ)^4 = 10000 ∧
  (2 : ℝ)^10 = 1024 ∧ (2 : ℝ)^11 = 2048 ∧
  (2 : ℝ)^12 = 4096 ∧ (2 : ℝ)^13 = 8192 →
  3/10 < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < 4/13 := by
  sorry

end NUMINAMATH_CALUDE_lg2_bounds_l2805_280564


namespace NUMINAMATH_CALUDE_murtha_pebbles_l2805_280563

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Murtha's pebble collection problem -/
theorem murtha_pebbles : arithmetic_sum 3 3 18 = 513 := by
  sorry

end NUMINAMATH_CALUDE_murtha_pebbles_l2805_280563


namespace NUMINAMATH_CALUDE_simple_interest_time_period_l2805_280597

/-- Calculates the time period for a simple interest problem -/
theorem simple_interest_time_period 
  (P : ℝ) (R : ℝ) (A : ℝ) 
  (h_P : P = 1300)
  (h_R : R = 5)
  (h_A : A = 1456) :
  ∃ T : ℝ, T = 2.4 ∧ A = P + (P * R * T / 100) := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_time_period_l2805_280597


namespace NUMINAMATH_CALUDE_exists_m_even_function_l2805_280513

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x

-- State the theorem
theorem exists_m_even_function :
  ∃ m : ℝ, ∀ x : ℝ, f m x = f m (-x) :=
sorry

end NUMINAMATH_CALUDE_exists_m_even_function_l2805_280513


namespace NUMINAMATH_CALUDE_power_fraction_equality_l2805_280553

theorem power_fraction_equality : (3^9 : ℝ) / (9^3 : ℝ) = 27 := by sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l2805_280553


namespace NUMINAMATH_CALUDE_sqrt_one_eighth_same_type_as_sqrt_two_l2805_280576

theorem sqrt_one_eighth_same_type_as_sqrt_two :
  ∃ (q : ℚ), Real.sqrt (1/8 : ℝ) = q * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_sqrt_one_eighth_same_type_as_sqrt_two_l2805_280576


namespace NUMINAMATH_CALUDE_number_puzzle_l2805_280596

theorem number_puzzle (x : ℝ) : (72 / 6 + x = 17) ↔ (x = 5) := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l2805_280596


namespace NUMINAMATH_CALUDE_tutors_next_common_workday_l2805_280527

def tim_schedule : ℕ := 5
def uma_schedule : ℕ := 6
def victor_schedule : ℕ := 9
def xavier_schedule : ℕ := 8

theorem tutors_next_common_workday : 
  lcm (lcm (lcm tim_schedule uma_schedule) victor_schedule) xavier_schedule = 360 := by
  sorry

end NUMINAMATH_CALUDE_tutors_next_common_workday_l2805_280527


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2805_280547

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) :
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2805_280547


namespace NUMINAMATH_CALUDE_centipede_human_ratio_theorem_l2805_280582

/-- Represents the population of an island with centipedes, humans, and sheep. -/
structure IslandPopulation where
  centipedes : ℕ
  humans : ℕ
  sheep : ℕ

/-- The ratio of centipedes to humans on the island. -/
def centipede_human_ratio (pop : IslandPopulation) : ℚ :=
  pop.centipedes / pop.humans

/-- Theorem stating the ratio of centipedes to humans given the conditions. -/
theorem centipede_human_ratio_theorem (pop : IslandPopulation) 
  (h1 : pop.centipedes = 100)
  (h2 : pop.sheep = pop.humans / 2) :
  centipede_human_ratio pop = 100 / pop.humans := by
  sorry

end NUMINAMATH_CALUDE_centipede_human_ratio_theorem_l2805_280582


namespace NUMINAMATH_CALUDE_polygon_ABCDE_perimeter_l2805_280512

/-- A point in a 2D coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The perimeter of a polygon given its vertices -/
def perimeter (vertices : List Point) : ℝ := sorry

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

theorem polygon_ABCDE_perimeter :
  let A : Point := ⟨0, 8⟩
  let B : Point := ⟨4, 8⟩
  let C : Point := ⟨4, 4⟩
  let D : Point := ⟨8, 0⟩
  let E : Point := ⟨0, 0⟩
  perimeter [A, B, C, D, E] = 12 + 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_ABCDE_perimeter_l2805_280512


namespace NUMINAMATH_CALUDE_no_integers_satisfy_conditions_l2805_280508

def f (i : ℕ) : ℕ := 1 + i^(1/3) + i

theorem no_integers_satisfy_conditions :
  ¬∃ i : ℕ, 1 ≤ i ∧ i ≤ 3000 ∧ (∃ m : ℕ, i = m^3) ∧ f i = 1 + i^(1/3) + i :=
by sorry

end NUMINAMATH_CALUDE_no_integers_satisfy_conditions_l2805_280508


namespace NUMINAMATH_CALUDE_sixth_quiz_score_l2805_280585

def quiz_scores : List ℕ := [86, 91, 88, 84, 97]
def desired_average : ℕ := 95
def num_quizzes : ℕ := 6

theorem sixth_quiz_score :
  ∃ (score : ℕ),
    (quiz_scores.sum + score) / num_quizzes = desired_average ∧
    score = num_quizzes * desired_average - quiz_scores.sum :=
by sorry

end NUMINAMATH_CALUDE_sixth_quiz_score_l2805_280585


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l2805_280572

theorem smallest_solution_quadratic (x : ℝ) :
  (6 * x^2 - 29 * x + 35 = 0) → (x ≥ 7/3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l2805_280572


namespace NUMINAMATH_CALUDE_inequality_proof_l2805_280559

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ∧
  ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ≤ 9/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2805_280559


namespace NUMINAMATH_CALUDE_student_count_difference_l2805_280515

/-- Represents the number of students in each grade level -/
structure StudentCounts where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- The problem statement -/
theorem student_count_difference (counts : StudentCounts) : 
  counts.freshmen + counts.sophomores + counts.juniors + counts.seniors = 800 →
  counts.juniors = 208 →
  counts.sophomores = 200 →
  counts.seniors = 160 →
  counts.freshmen - counts.sophomores = 32 := by
  sorry

end NUMINAMATH_CALUDE_student_count_difference_l2805_280515


namespace NUMINAMATH_CALUDE_product_of_roots_l2805_280510

theorem product_of_roots : (16 : ℝ) ^ (1/4) * (32 : ℝ) ^ (1/5) = 4 := by sorry

end NUMINAMATH_CALUDE_product_of_roots_l2805_280510


namespace NUMINAMATH_CALUDE_game_winner_conditions_l2805_280575

/-- Represents the possible outcomes of the game -/
inductive GameOutcome
  | AWins
  | BWins

/-- Represents the game state -/
structure GameState where
  n : ℕ
  m : ℕ
  currentPlayer : Bool  -- true for A, false for B

/-- Determines the winner of the game given the initial state -/
def determineWinner (initialState : GameState) : GameOutcome :=
  if initialState.n = initialState.m then
    GameOutcome.BWins
  else
    GameOutcome.AWins

/-- Theorem stating the winning conditions for the game -/
theorem game_winner_conditions (n m : ℕ) (hn : n > 1) (hm : m > 1) :
  let initialState := GameState.mk n m true
  determineWinner initialState =
    if n = m then
      GameOutcome.BWins
    else
      GameOutcome.AWins :=
by
  sorry


end NUMINAMATH_CALUDE_game_winner_conditions_l2805_280575


namespace NUMINAMATH_CALUDE_number_problem_l2805_280533

theorem number_problem (x : ℝ) : (0.16 * (0.40 * x) = 6) → x = 93.75 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2805_280533


namespace NUMINAMATH_CALUDE_total_amount_distributed_l2805_280555

-- Define the shares of A, B, and C
def share_A : ℕ := sorry
def share_B : ℕ := sorry
def share_C : ℕ := 495

-- Define the amounts to be decreased
def decrease_A : ℕ := 25
def decrease_B : ℕ := 10
def decrease_C : ℕ := 15

-- Define the ratio of remaining amounts
def ratio_A : ℕ := 3
def ratio_B : ℕ := 2
def ratio_C : ℕ := 5

-- Theorem to prove
theorem total_amount_distributed :
  share_A + share_B + share_C = 1010 :=
by
  sorry

-- Lemma to ensure the ratio condition is met
lemma ratio_condition :
  (share_A - decrease_A) * ratio_B * ratio_C = 
  (share_B - decrease_B) * ratio_A * ratio_C ∧
  (share_B - decrease_B) * ratio_A * ratio_C = 
  (share_C - decrease_C) * ratio_A * ratio_B :=
by
  sorry

end NUMINAMATH_CALUDE_total_amount_distributed_l2805_280555


namespace NUMINAMATH_CALUDE_harrison_croissant_cost_l2805_280544

/-- The cost of croissants for Harrison in a year -/
def croissant_cost (regular_price almond_price : ℚ) (weeks_per_year : ℕ) : ℚ :=
  weeks_per_year * (regular_price + almond_price)

/-- Theorem: Harrison spends $468.00 on croissants in a year -/
theorem harrison_croissant_cost :
  croissant_cost (35/10) (55/10) 52 = 468 :=
sorry

end NUMINAMATH_CALUDE_harrison_croissant_cost_l2805_280544


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l2805_280505

/-- Given a point P in the Cartesian coordinate system, 
    find its symmetric point with respect to the x-axis. -/
def symmetric_point_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

/-- Theorem: The coordinates of the point symmetric to P (-1, 2) 
    with respect to the x-axis are (-1, -2). -/
theorem symmetric_point_theorem : 
  symmetric_point_x_axis (-1, 2) = (-1, -2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_theorem_l2805_280505


namespace NUMINAMATH_CALUDE_count_subset_pairs_formula_l2805_280587

/-- The number of pairs of non-empty subsets (A, B) of {1, 2, ..., n} such that
    the maximum element of A is less than the minimum element of B -/
def count_subset_pairs (n : ℕ) : ℕ :=
  (n - 2) * 2^(n - 1) + 1

/-- Theorem stating that for any integer n ≥ 3, the count of subset pairs
    satisfying the given condition is equal to (n-2) * 2^(n-1) + 1 -/
theorem count_subset_pairs_formula (n : ℕ) (h : n ≥ 3) :
  count_subset_pairs n = (n - 2) * 2^(n - 1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_count_subset_pairs_formula_l2805_280587


namespace NUMINAMATH_CALUDE_larger_cuboid_length_l2805_280583

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

/-- The dimensions of the smaller cuboid -/
def smallerCuboid : CuboidDimensions :=
  { length := 5, width := 4, height := 3 }

/-- The number of smaller cuboids that can be formed from the larger cuboid -/
def numberOfSmallerCuboids : ℕ := 32

/-- The width of the larger cuboid -/
def largerCuboidWidth : ℝ := 10

/-- The height of the larger cuboid -/
def largerCuboidHeight : ℝ := 12

theorem larger_cuboid_length :
  ∃ (largerLength : ℝ),
    cuboidVolume { length := largerLength, width := largerCuboidWidth, height := largerCuboidHeight } =
    (numberOfSmallerCuboids : ℝ) * cuboidVolume smallerCuboid ∧
    largerLength = 16 := by
  sorry

end NUMINAMATH_CALUDE_larger_cuboid_length_l2805_280583


namespace NUMINAMATH_CALUDE_part_I_part_II_l2805_280514

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 < x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | 5 - a < x ∧ x < a}

-- Define the complement of A relative to ℝ
def C_R_A : Set ℝ := {x | x ≤ 3 ∨ x ≥ 7}

-- Theorem for part (I)
theorem part_I : (C_R_A ∩ B) = {x | (2 < x ∧ x ≤ 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

-- Theorem for part (II)
theorem part_II (a : ℝ) : C a ⊆ (A ∪ B) → a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l2805_280514


namespace NUMINAMATH_CALUDE_sum_of_odd_coefficients_l2805_280538

theorem sum_of_odd_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (3*x - 2)^6 = a₀ + a₁*(2*x - 1) + a₂*(2*x - 1)^2 + a₃*(2*x - 1)^3 + 
                      a₄*(2*x - 1)^4 + a₅*(2*x - 1)^5 + a₆*(2*x - 1)^6) →
  a₁ + a₃ + a₅ = -63/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_odd_coefficients_l2805_280538


namespace NUMINAMATH_CALUDE_inequality_proof_l2805_280501

theorem inequality_proof (x : ℝ) : 2 ≤ (3 * x^2 - 6 * x + 6) / (x^2 - x + 1) ∧ (3 * x^2 - 6 * x + 6) / (x^2 - x + 1) ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2805_280501


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2805_280540

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℤ) 
  (h_arithmetic : ArithmeticSequence a)
  (h_eq : a 3 + a 9 = 4 * a 5)
  (h_a2 : a 2 = -8) :
  ∃ d : ℤ, d = 4 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2805_280540


namespace NUMINAMATH_CALUDE_x_intercept_is_one_l2805_280556

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ :=
  sorry

/-- The theorem stating that the x-intercept of the given line is 1 -/
theorem x_intercept_is_one :
  let l : Line := { x₁ := 2, y₁ := -2, x₂ := -1, y₂ := 4 }
  x_intercept l = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_is_one_l2805_280556


namespace NUMINAMATH_CALUDE_equation_solution_l2805_280548

theorem equation_solution : ∃ x : ℚ, 50 + 5 * x / (180 / 3) = 51 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2805_280548


namespace NUMINAMATH_CALUDE_integer_solution_congruence_l2805_280561

theorem integer_solution_congruence (x y z : ℤ) 
  (eq1 : x - 3*y + 2*z = 1)
  (eq2 : 2*x + y - 5*z = 7) :
  z ≡ 1 [ZMOD 7] :=
sorry

end NUMINAMATH_CALUDE_integer_solution_congruence_l2805_280561


namespace NUMINAMATH_CALUDE_square_fencing_cost_l2805_280577

/-- The cost of fencing one side of a square -/
def cost_per_side : ℕ := 69

/-- The number of sides in a square -/
def num_sides : ℕ := 4

/-- The total cost of fencing a square -/
def total_cost : ℕ := cost_per_side * num_sides

theorem square_fencing_cost : total_cost = 276 := by
  sorry

end NUMINAMATH_CALUDE_square_fencing_cost_l2805_280577


namespace NUMINAMATH_CALUDE_equation_solution_l2805_280566

theorem equation_solution : ∃ x : ℝ, (3*x + 4*x = 600 - (2*x + 6*x + x)) ∧ x = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2805_280566
