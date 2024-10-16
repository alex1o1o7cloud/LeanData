import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_foci_coordinates_l3744_374497

/-- The coordinates of the foci of the ellipse 25x^2 + 16y^2 = 1 are (0, 3/20) and (0, -3/20) -/
theorem ellipse_foci_coordinates :
  let ellipse := {(x, y) : ℝ × ℝ | 25 * x^2 + 16 * y^2 = 1}
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁ ∈ ellipse ∧ f₂ ∈ ellipse) ∧ 
    (∀ p ∈ ellipse, (dist p f₁) + (dist p f₂) = (dist (1/5, 0) (-1/5, 0))) ∧
    f₁ = (0, 3/20) ∧ f₂ = (0, -3/20) :=
by sorry


end NUMINAMATH_CALUDE_ellipse_foci_coordinates_l3744_374497


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_z_on_y_equals_x_l3744_374402

/-- The real part of the complex number z -/
def real_part (m : ℝ) : ℝ := m^2 - 8*m + 15

/-- The imaginary part of the complex number z -/
def imag_part (m : ℝ) : ℝ := m^2 - 5*m - 14

/-- The complex number z -/
def z (m : ℝ) : ℂ := Complex.mk (real_part m) (imag_part m)

/-- Condition for z to be in the fourth quadrant -/
def in_fourth_quadrant (m : ℝ) : Prop :=
  real_part m > 0 ∧ imag_part m < 0

/-- Condition for z to be on the line y = x -/
def on_y_equals_x (m : ℝ) : Prop :=
  real_part m = imag_part m

theorem z_in_fourth_quadrant :
  ∀ m : ℝ, in_fourth_quadrant m ↔ (-2 < m ∧ m < 3) ∨ (5 < m ∧ m < 7) :=
sorry

theorem z_on_y_equals_x :
  ∀ m : ℝ, on_y_equals_x m ↔ m = 29/3 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_z_on_y_equals_x_l3744_374402


namespace NUMINAMATH_CALUDE_negation_existential_derivative_l3744_374456

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

theorem negation_existential_derivative :
  (¬ ∃ x : ℝ, f' x ≥ 0) ↔ (∀ x : ℝ, f' x < 0) :=
sorry

end NUMINAMATH_CALUDE_negation_existential_derivative_l3744_374456


namespace NUMINAMATH_CALUDE_vote_count_proof_l3744_374470

theorem vote_count_proof (total votes_against votes_in_favor : ℕ) 
  (h1 : votes_in_favor = votes_against + 68)
  (h2 : votes_against = (40 : ℕ) * total / 100)
  (h3 : total = votes_in_favor + votes_against) :
  total = 340 :=
sorry

end NUMINAMATH_CALUDE_vote_count_proof_l3744_374470


namespace NUMINAMATH_CALUDE_parabola_translation_existence_l3744_374436

theorem parabola_translation_existence : ∃ (h k : ℝ),
  (0 = -(0 - h)^2 + k) ∧  -- passes through origin
  ((1/2) * (2*h) * k = 1) ∧  -- triangle area is 1
  (h^2 = k) ∧  -- vertex is (h, k)
  (h = 1 ∨ h = -1) ∧
  (k = 1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_existence_l3744_374436


namespace NUMINAMATH_CALUDE_f_is_even_count_f_eq_2016_l3744_374426

/-- The smallest factor of n that is not 1 -/
def smallest_factor (n : ℕ) : ℕ := sorry

/-- The function f as defined in the problem -/
def f (n : ℕ) : ℕ := n + smallest_factor n

/-- Theorem stating that f(n) is always even for n > 1 -/
theorem f_is_even (n : ℕ) (h : n > 1) : Even (f n) := by sorry

/-- Theorem stating that there are exactly 3 positive integers n such that f(n) = 2016 -/
theorem count_f_eq_2016 : ∃! (s : Finset ℕ), (∀ n ∈ s, f n = 2016) ∧ s.card = 3 := by sorry

end NUMINAMATH_CALUDE_f_is_even_count_f_eq_2016_l3744_374426


namespace NUMINAMATH_CALUDE_equation_solution_l3744_374451

theorem equation_solution (x y : ℝ) : 
  y = 3 * x → 
  (4 * y^2 - 3 * y + 5 = 3 * (8 * x^2 - 3 * y + 1)) ↔ 
  (x = (Real.sqrt 19 - 3) / 4 ∨ x = (-Real.sqrt 19 - 3) / 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3744_374451


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3744_374469

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (7 + 3 * z) = 13 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3744_374469


namespace NUMINAMATH_CALUDE_recycle_128_cans_l3744_374432

/-- The number of new cans that can be created through recycling, given an initial number of cans -/
def recycle_cans (initial_cans : ℕ) : ℕ :=
  if initial_cans < 2 then 0
  else (initial_cans / 2) + recycle_cans (initial_cans / 2)

/-- Theorem stating that recycling 128 cans produces 127 new cans -/
theorem recycle_128_cans :
  recycle_cans 128 = 127 := by
  sorry

end NUMINAMATH_CALUDE_recycle_128_cans_l3744_374432


namespace NUMINAMATH_CALUDE_divisibility_condition_l3744_374484

theorem divisibility_condition (a b : ℕ) (ha : a ≥ 2) (hb : b ≥ 2) :
  (∃ k : ℤ, (a * b - 1 : ℤ) = k * ((a - 1) * (b - 1))) ↔ (a = 2 ∧ b = 2) ∨ (a = 3 ∧ b = 3) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3744_374484


namespace NUMINAMATH_CALUDE_age_ratio_problem_l3744_374442

theorem age_ratio_problem (cindy_age jan_age marcia_age greg_age : ℕ) : 
  cindy_age = 5 →
  jan_age = cindy_age + 2 →
  ∃ k : ℕ, marcia_age = k * jan_age →
  greg_age = marcia_age + 2 →
  greg_age = 16 →
  marcia_age / jan_age = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l3744_374442


namespace NUMINAMATH_CALUDE_money_sharing_l3744_374488

theorem money_sharing (amanda ben carlos total : ℕ) : 
  amanda + ben + carlos = total →
  amanda = 3 * (ben / 5) →
  carlos = 9 * (ben / 5) →
  ben = 50 →
  total = 170 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_l3744_374488


namespace NUMINAMATH_CALUDE_convex_quadrilaterals_from_circle_points_convex_quadrilaterals_count_proof_l3744_374422

/-- The number of convex quadrilaterals formed from 12 points on a circle -/
theorem convex_quadrilaterals_from_circle_points : ℕ :=
  Nat.choose 12 4

/-- Proof that the number of convex quadrilaterals is correct -/
theorem convex_quadrilaterals_count_proof :
  convex_quadrilaterals_from_circle_points = 495 := by
  sorry


end NUMINAMATH_CALUDE_convex_quadrilaterals_from_circle_points_convex_quadrilaterals_count_proof_l3744_374422


namespace NUMINAMATH_CALUDE_josh_film_cost_l3744_374459

/-- The cost of each film Josh bought -/
def film_cost : ℚ := 5

/-- The number of films Josh bought -/
def num_films : ℕ := 9

/-- The number of books Josh bought -/
def num_books : ℕ := 4

/-- The cost of each book -/
def book_cost : ℚ := 4

/-- The number of CDs Josh bought -/
def num_cds : ℕ := 6

/-- The cost of each CD -/
def cd_cost : ℚ := 3

/-- The total amount Josh spent -/
def total_spent : ℚ := 79

theorem josh_film_cost :
  film_cost * num_films + book_cost * num_books + cd_cost * num_cds = total_spent :=
by sorry

end NUMINAMATH_CALUDE_josh_film_cost_l3744_374459


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_E_subset_B_implies_a_geq_neg_one_l3744_374433

-- Define the sets A and B
def A : Set ℝ := {x | (x + 3) * (x - 6) ≥ 0}
def B : Set ℝ := {x | (x + 2) / (x - 14) < 0}

-- Define the set E
def E (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a + 1}

-- Statement for the first part of the problem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x | x ≤ -3 ∨ x ≥ 14} := by sorry

-- Statement for the second part of the problem
theorem E_subset_B_implies_a_geq_neg_one (a : ℝ) :
  E a ⊆ B → a ≥ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_E_subset_B_implies_a_geq_neg_one_l3744_374433


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3744_374416

theorem simplify_square_roots : 
  (Real.sqrt 392 / Real.sqrt 352) + (Real.sqrt 180 / Real.sqrt 120) = 
  (7 * Real.sqrt 6 + 6 * Real.sqrt 11) / (2 * Real.sqrt 66) := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3744_374416


namespace NUMINAMATH_CALUDE_reciprocal_multiplier_l3744_374443

theorem reciprocal_multiplier (x m : ℝ) : 
  x > 0 → x = 7 → x - 4 = m * (1/x) → m = 21 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_multiplier_l3744_374443


namespace NUMINAMATH_CALUDE_flash_drive_problem_l3744_374430

/-- Represents the number of flash drives needed to store files -/
def min_flash_drives (total_files : ℕ) (drive_capacity : ℚ) 
  (file_sizes : List (ℕ × ℚ)) : ℕ :=
  sorry

/-- The problem statement -/
theorem flash_drive_problem :
  let total_files : ℕ := 40
  let drive_capacity : ℚ := 2
  let file_sizes : List (ℕ × ℚ) := [(4, 1.2), (16, 0.9), (20, 0.6)]
  min_flash_drives total_files drive_capacity file_sizes = 20 := by
  sorry

end NUMINAMATH_CALUDE_flash_drive_problem_l3744_374430


namespace NUMINAMATH_CALUDE_cars_produced_in_europe_l3744_374483

/-- The number of cars produced in Europe by a car company -/
def cars_in_europe (total_cars : ℕ) (cars_in_north_america : ℕ) : ℕ :=
  total_cars - cars_in_north_america

/-- Theorem stating that the number of cars produced in Europe is 2871 -/
theorem cars_produced_in_europe :
  cars_in_europe 6755 3884 = 2871 := by
  sorry

end NUMINAMATH_CALUDE_cars_produced_in_europe_l3744_374483


namespace NUMINAMATH_CALUDE_special_isosceles_triangle_sides_l3744_374420

/-- An isosceles triangle with specific incenter properties -/
structure SpecialIsoscelesTriangle where
  -- The length of the two equal sides
  side : ℝ
  -- The length of the base
  base : ℝ
  -- The distance from the vertex to the incenter along the altitude
  vertexToIncenter : ℝ
  -- The distance from the incenter to the base along the altitude
  incenterToBase : ℝ
  -- Ensure the triangle is isosceles
  isIsosceles : side > 0
  -- Ensure the incenter divides the altitude as specified
  incenterDivision : vertexToIncenter = 5 ∧ incenterToBase = 3

/-- The theorem stating the side lengths of the special isosceles triangle -/
theorem special_isosceles_triangle_sides 
  (t : SpecialIsoscelesTriangle) : t.side = 10 ∧ t.base = 12 := by
  sorry

#check special_isosceles_triangle_sides

end NUMINAMATH_CALUDE_special_isosceles_triangle_sides_l3744_374420


namespace NUMINAMATH_CALUDE_marble_problem_l3744_374458

/-- The total number of marbles given the conditions of the problem -/
def total_marbles : ℕ := 36

/-- Mario's share of marbles before Manny gives away 2 marbles -/
def mario_marbles : ℕ := 16

/-- Manny's share of marbles before giving away 2 marbles -/
def manny_marbles : ℕ := 20

/-- The ratio of Mario's marbles to Manny's marbles -/
def marble_ratio : Rat := 4 / 5

theorem marble_problem :
  (mario_marbles : ℚ) / (manny_marbles : ℚ) = marble_ratio ∧
  manny_marbles - 2 = 18 ∧
  total_marbles = mario_marbles + manny_marbles :=
by sorry

end NUMINAMATH_CALUDE_marble_problem_l3744_374458


namespace NUMINAMATH_CALUDE_livestream_sales_scientific_notation_l3744_374460

/-- Proves that 1814 billion yuan is equal to 1.814 × 10^12 yuan -/
theorem livestream_sales_scientific_notation :
  (1814 : ℝ) * (10^9 : ℝ) = 1.814 * (10^12 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_livestream_sales_scientific_notation_l3744_374460


namespace NUMINAMATH_CALUDE_complex_sum_equals_negative_one_l3744_374440

theorem complex_sum_equals_negative_one (z : ℂ) (h : z = Complex.exp (2 * Real.pi * Complex.I / 9)) :
  z^2 / (1 + z^3) + z^4 / (1 + z^6) + z^6 / (1 + z^9) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_negative_one_l3744_374440


namespace NUMINAMATH_CALUDE_pencil_buyers_difference_l3744_374491

theorem pencil_buyers_difference : ∀ (pencil_cost : ℕ) 
  (eighth_graders fifth_graders : ℕ),
  pencil_cost > 0 ∧
  pencil_cost * eighth_graders = 234 ∧
  pencil_cost * fifth_graders = 285 ∧
  fifth_graders ≤ 25 →
  fifth_graders - eighth_graders = 17 := by
sorry

end NUMINAMATH_CALUDE_pencil_buyers_difference_l3744_374491


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3744_374485

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 1 - Complex.I) → z = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3744_374485


namespace NUMINAMATH_CALUDE_square_function_properties_l3744_374492

-- Define the function f(x) = x^2 on (0, +∞)
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem square_function_properties :
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ →
    (f (x₁ * x₂) = f x₁ * f x₂) ∧
    ((f x₁ - f x₂) / (x₁ - x₂) > 0) ∧
    (f ((x₁ + x₂) / 2) < (f x₁ + f x₂) / 2) :=
by sorry

end NUMINAMATH_CALUDE_square_function_properties_l3744_374492


namespace NUMINAMATH_CALUDE_sqrt_144_squared_times_2_l3744_374475

theorem sqrt_144_squared_times_2 : 2 * (Real.sqrt 144)^2 = 288 := by sorry

end NUMINAMATH_CALUDE_sqrt_144_squared_times_2_l3744_374475


namespace NUMINAMATH_CALUDE_sin_150_degrees_l3744_374439

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l3744_374439


namespace NUMINAMATH_CALUDE_polynomial_coefficient_l3744_374476

theorem polynomial_coefficient (a : Fin 11 → ℝ) :
  (∀ x : ℝ, x^2 + x^10 = a 0 + a 1 * (x + 1) + a 2 * (x + 1)^2 + 
    a 3 * (x + 1)^3 + a 4 * (x + 1)^4 + a 5 * (x + 1)^5 + 
    a 6 * (x + 1)^6 + a 7 * (x + 1)^7 + a 8 * (x + 1)^8 + 
    a 9 * (x + 1)^9 + a 10 * (x + 1)^10) →
  a 9 = -10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_l3744_374476


namespace NUMINAMATH_CALUDE_min_distance_squared_l3744_374495

theorem min_distance_squared (a b c d : ℝ) 
  (h1 : Real.log a - Real.log 3 = Real.log c) 
  (h2 : b * d = -3) : 
  ∃ (min_val : ℝ), min_val = 18/5 ∧ 
    ∀ (x y : ℝ), (x - b)^2 + (y - c)^2 ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_squared_l3744_374495


namespace NUMINAMATH_CALUDE_variance_of_transformed_binomial_l3744_374418

/-- A random variable following a binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- The variance of a linear transformation of a random variable -/
def varianceLinearTransform (a b : ℝ) (v : ℝ) : ℝ := a^2 * v

theorem variance_of_transformed_binomial :
  let ξ : BinomialDistribution := ⟨100, 0.3, by norm_num⟩
  varianceLinearTransform 3 (-5) (variance ξ) = 189 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_transformed_binomial_l3744_374418


namespace NUMINAMATH_CALUDE_fraction_simplification_l3744_374421

theorem fraction_simplification :
  (21 : ℚ) / 16 * 48 / 35 * 80 / 63 = 48 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3744_374421


namespace NUMINAMATH_CALUDE_T_divisibility_l3744_374411

def T : Set ℕ := {s | ∃ n : ℕ, s = (n - 2)^2 + (n - 1)^2 + n^2 + (n + 1)^2}

theorem T_divisibility :
  (∀ s ∈ T, ¬(9 ∣ s)) ∧ (∃ s ∈ T, 4 ∣ s) := by sorry

end NUMINAMATH_CALUDE_T_divisibility_l3744_374411


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3744_374410

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (m n : ℝ) :
  a > 0 →
  b > 0 →
  c = (a^2 + b^2).sqrt →
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1 ↔ ((x, y) : ℝ × ℝ) ∈ {p | p.1^2 / a^2 - p.2^2 / b^2 = 1}) →
  (c, 0) ∈ {p | p.1^2 / a^2 - p.2^2 / b^2 = 1} →
  ((m + n) * c, (m - n) * b * c / a) ∈ {p | p.1^2 / a^2 - p.2^2 / b^2 = 1} →
  m * n = 2 / 9 →
  (a^2 + b^2) / a^2 = (3 * Real.sqrt 2 / 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3744_374410


namespace NUMINAMATH_CALUDE_tape_recorder_cost_l3744_374454

theorem tape_recorder_cost : ∃ (x y : ℕ), 
  x > 0 ∧ y > 0 ∧
  170 ≤ x * y ∧ x * y ≤ 195 ∧
  y = 2 * x + 2 ∧
  x * y = 180 := by
  sorry

end NUMINAMATH_CALUDE_tape_recorder_cost_l3744_374454


namespace NUMINAMATH_CALUDE_power_three_equality_l3744_374413

theorem power_three_equality : 3^2012 - 6 * 3^2013 + 2 * 3^2014 = 3^2012 := by
  sorry

end NUMINAMATH_CALUDE_power_three_equality_l3744_374413


namespace NUMINAMATH_CALUDE_arccos_one_half_eq_pi_third_l3744_374425

theorem arccos_one_half_eq_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_eq_pi_third_l3744_374425


namespace NUMINAMATH_CALUDE_furniture_cost_price_l3744_374474

theorem furniture_cost_price (computer_table_price chair_price bookshelf_price : ℝ)
  (h1 : computer_table_price = 8091)
  (h2 : chair_price = 5346)
  (h3 : bookshelf_price = 11700)
  (computer_table_markup : ℝ)
  (h4 : computer_table_markup = 0.24)
  (chair_markup : ℝ)
  (h5 : chair_markup = 0.18)
  (chair_discount : ℝ)
  (h6 : chair_discount = 0.05)
  (bookshelf_markup : ℝ)
  (h7 : bookshelf_markup = 0.30)
  (sales_tax : ℝ)
  (h8 : sales_tax = 0.045) :
  ∃ (computer_table_cost chair_cost bookshelf_cost : ℝ),
    computer_table_cost = computer_table_price / (1 + computer_table_markup) ∧
    chair_cost = chair_price / ((1 + chair_markup) * (1 - chair_discount)) ∧
    bookshelf_cost = bookshelf_price / (1 + bookshelf_markup) ∧
    computer_table_cost + chair_cost + bookshelf_cost = 20295 :=
by sorry

end NUMINAMATH_CALUDE_furniture_cost_price_l3744_374474


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3744_374461

theorem sum_of_numbers (a b c : ℝ) : 
  a = 0.8 → b = 1/2 → c = 0.5 → a < 2 ∧ b < 2 ∧ c < 2 → a + b + c = 1.8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3744_374461


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_864_l3744_374486

theorem sum_of_roots_equals_864 
  (p q r s : ℝ) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h_eq1 : ∀ x, x^2 - 8*p*x - 12*q = 0 ↔ x = r ∨ x = s)
  (h_eq2 : ∀ x, x^2 - 8*r*x - 12*s = 0 ↔ x = p ∨ x = q) :
  p + q + r + s = 864 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_864_l3744_374486


namespace NUMINAMATH_CALUDE_line_equation_proof_l3744_374494

theorem line_equation_proof (m c : ℝ) (h1 : c ≠ 0) (h2 : m = 4 + 2 * Real.sqrt 7) (h3 : c = 2 - 2 * Real.sqrt 7) :
  ∃ k : ℝ, 
    (∀ k' : ℝ, k' ≠ k → 
      (abs ((k'^2 + 4*k' + 3) - (m*k' + c)) ≠ 7 ∨ 
       ¬∃ y1 y2 : ℝ, y1 = k'^2 + 4*k' + 3 ∧ y2 = m*k' + c ∧ y1 ≠ y2)) ∧
    (∃ y1 y2 : ℝ, y1 = k^2 + 4*k + 3 ∧ y2 = m*k + c ∧ y1 ≠ y2 ∧ abs (y1 - y2) = 7) ∧
    m * 1 + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l3744_374494


namespace NUMINAMATH_CALUDE_incenter_coordinates_specific_triangle_l3744_374478

/-- Given a triangle PQR with side lengths p, q, r, this function returns the coordinates of the incenter I -/
def incenter_coordinates (p q r : ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating that for a triangle with side lengths 8, 10, and 6, the incenter coordinates are (1/3, 5/12, 1/4) -/
theorem incenter_coordinates_specific_triangle :
  let (x, y, z) := incenter_coordinates 8 10 6
  x = 1/3 ∧ y = 5/12 ∧ z = 1/4 ∧ x + y + z = 1 := by sorry

end NUMINAMATH_CALUDE_incenter_coordinates_specific_triangle_l3744_374478


namespace NUMINAMATH_CALUDE_function_inequality_l3744_374466

-- Define the function f
variable {f : ℝ → ℝ}

-- State the theorem
theorem function_inequality
  (h : ∀ x y : ℝ, f y - f x ≤ (y - x)^2)
  (n : ℕ)
  (hn : n > 0)
  (a b : ℝ) :
  |f b - f a| ≤ (1 / n : ℝ) * (b - a)^2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3744_374466


namespace NUMINAMATH_CALUDE_simplify_expression_l3744_374446

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = a + b) :
  a / b + b / a - 1 / (a * b) = 1 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l3744_374446


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l3744_374438

theorem largest_multiple_of_15_under_500 : 
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 → n ≤ 495 := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l3744_374438


namespace NUMINAMATH_CALUDE_blue_line_length_calculation_l3744_374431

-- Define the length of the white line
def white_line_length : ℝ := 7.666666666666667

-- Define the difference between the white and blue lines
def length_difference : ℝ := 4.333333333333333

-- Define the length of the blue line
def blue_line_length : ℝ := white_line_length - length_difference

-- Theorem statement
theorem blue_line_length_calculation : 
  blue_line_length = 3.333333333333334 := by sorry

end NUMINAMATH_CALUDE_blue_line_length_calculation_l3744_374431


namespace NUMINAMATH_CALUDE_club_selection_count_l3744_374465

theorem club_selection_count (n : ℕ) (h : n = 18) : 
  n * (Nat.choose (n - 1) 2) = 2448 := by
  sorry

end NUMINAMATH_CALUDE_club_selection_count_l3744_374465


namespace NUMINAMATH_CALUDE_estimate_fish_population_l3744_374405

/-- Represents the fish pond scenario --/
structure FishPond where
  totalFish : ℕ  -- Total number of fish in the pond
  markedFish : ℕ  -- Number of fish initially marked
  secondSampleSize : ℕ  -- Size of the second sample
  markedInSecondSample : ℕ  -- Number of marked fish in the second sample

/-- Theorem stating the estimated number of fish in the pond --/
theorem estimate_fish_population (pond : FishPond) 
  (h1 : pond.markedFish = 100)
  (h2 : pond.secondSampleSize = 120)
  (h3 : pond.markedInSecondSample = 15) :
  pond.totalFish = 800 := by
  sorry

#check estimate_fish_population

end NUMINAMATH_CALUDE_estimate_fish_population_l3744_374405


namespace NUMINAMATH_CALUDE_probability_all_red_or_all_white_l3744_374406

/-- The probability of drawing either all red marbles or all white marbles when drawing 3 marbles
    without replacement from a bag containing 5 red, 4 white, and 6 blue marbles -/
theorem probability_all_red_or_all_white (total_marbles : ℕ) (red_marbles : ℕ) (white_marbles : ℕ) 
    (blue_marbles : ℕ) (drawn_marbles : ℕ) :
  total_marbles = red_marbles + white_marbles + blue_marbles →
  total_marbles = 15 →
  red_marbles = 5 →
  white_marbles = 4 →
  blue_marbles = 6 →
  drawn_marbles = 3 →
  (red_marbles.choose drawn_marbles * (total_marbles - drawn_marbles).factorial / total_marbles.factorial +
   white_marbles.choose drawn_marbles * (total_marbles - drawn_marbles).factorial / total_marbles.factorial : ℚ) = 14 / 455 := by
  sorry

#check probability_all_red_or_all_white

end NUMINAMATH_CALUDE_probability_all_red_or_all_white_l3744_374406


namespace NUMINAMATH_CALUDE_lemon_pie_degrees_l3744_374414

/-- The number of degrees in a circle --/
def circle_degrees : ℕ := 360

/-- The total number of students in the class --/
def total_students : ℕ := 45

/-- The number of students preferring chocolate pie --/
def chocolate_pref : ℕ := 15

/-- The number of students preferring apple pie --/
def apple_pref : ℕ := 10

/-- The number of students preferring blueberry pie --/
def blueberry_pref : ℕ := 9

/-- Calculate the number of students preferring lemon pie --/
def lemon_pref : ℚ :=
  (total_students - (chocolate_pref + apple_pref + blueberry_pref)) / 2

/-- Theorem: The number of degrees for lemon pie on a pie chart is 44° --/
theorem lemon_pie_degrees : 
  (lemon_pref / total_students) * circle_degrees = 44 := by
  sorry

end NUMINAMATH_CALUDE_lemon_pie_degrees_l3744_374414


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3744_374452

theorem rectangle_circle_area_ratio :
  ∀ (w l r : ℝ),
  w > 0 → l > 0 → r > 0 →
  l = 2 * w →
  2 * l + 2 * w = 2 * π * r →
  (l * w) / (π * r^2) = 2 * π / 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3744_374452


namespace NUMINAMATH_CALUDE_unique_solution_positive_root_l3744_374401

theorem unique_solution_positive_root (x : ℝ) :
  x ≥ 0 ∧ 2021 * (x^2020)^(1/202) - 1 = 2020 * x ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_positive_root_l3744_374401


namespace NUMINAMATH_CALUDE_g_evaluation_and_derivative_l3744_374477

def g (x : ℝ) : ℝ := 3 * x^5 - 4 * x^4 + 2 * x^3 - 28 * x^2 + 15 * x - 90

theorem g_evaluation_and_derivative :
  g 6 = 17568 ∧ (deriv g) 6 = 15879 := by sorry

end NUMINAMATH_CALUDE_g_evaluation_and_derivative_l3744_374477


namespace NUMINAMATH_CALUDE_picture_area_l3744_374450

theorem picture_area (x y : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : 2 * x * y + 9 * x + 4 * y = 42) : x * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_l3744_374450


namespace NUMINAMATH_CALUDE_symmetry_yoz_proof_l3744_374467

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the yOz plane -/
def symmetryYOZ (p : Point3D) : Point3D :=
  ⟨-p.x, p.y, p.z⟩

/-- The original point -/
def originalPoint : Point3D :=
  ⟨1, -2, 3⟩

theorem symmetry_yoz_proof :
  symmetryYOZ originalPoint = Point3D.mk (-1) (-2) 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_yoz_proof_l3744_374467


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l3744_374468

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 3742) % 17 = 1578 % 17 ∧
  ∀ (y : ℕ), y > 0 ∧ (y + 3742) % 17 = 1578 % 17 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l3744_374468


namespace NUMINAMATH_CALUDE_equation_solution_exists_l3744_374455

theorem equation_solution_exists : ∃ (a b c d e : ℕ), 
  a ∈ ({1, 2, 3, 5, 6} : Set ℕ) ∧ 
  b ∈ ({1, 2, 3, 5, 6} : Set ℕ) ∧ 
  c ∈ ({1, 2, 3, 5, 6} : Set ℕ) ∧ 
  d ∈ ({1, 2, 3, 5, 6} : Set ℕ) ∧ 
  e ∈ ({1, 2, 3, 5, 6} : Set ℕ) ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧ 
  (a + b - c) * d / e = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l3744_374455


namespace NUMINAMATH_CALUDE_spade_equation_solution_l3744_374464

/-- The spade operation -/
def spade_op (X Y : ℝ) : ℝ := 4 * X - 3 * Y + 7

/-- Theorem stating that if X spade 5 = 23, then X = 7.75 -/
theorem spade_equation_solution :
  ∀ X : ℝ, spade_op X 5 = 23 → X = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_spade_equation_solution_l3744_374464


namespace NUMINAMATH_CALUDE_money_problem_l3744_374498

theorem money_problem (a b : ℝ) 
  (h1 : 5 * a + 2 * b > 100)
  (h2 : 4 * a - b = 40) : 
  a > 180 / 13 ∧ b > 200 / 13 := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l3744_374498


namespace NUMINAMATH_CALUDE_discount_and_increase_l3744_374407

theorem discount_and_increase (original_price : ℝ) (h : original_price > 0) :
  ∃ (x : ℝ), original_price = original_price * (1 - 0.2) * (1 + x / 100) ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_discount_and_increase_l3744_374407


namespace NUMINAMATH_CALUDE_anayet_driving_time_l3744_374481

/-- Proves that Anayet drove for 2 hours given the conditions of the problem -/
theorem anayet_driving_time 
  (total_distance : ℝ)
  (amoli_speed : ℝ)
  (amoli_time : ℝ)
  (anayet_speed : ℝ)
  (remaining_distance : ℝ)
  (h1 : total_distance = 369)
  (h2 : amoli_speed = 42)
  (h3 : amoli_time = 3)
  (h4 : anayet_speed = 61)
  (h5 : remaining_distance = 121)
  : ∃ (anayet_time : ℝ), anayet_time = 2 ∧ 
    total_distance = amoli_speed * amoli_time + anayet_speed * anayet_time + remaining_distance :=
by
  sorry


end NUMINAMATH_CALUDE_anayet_driving_time_l3744_374481


namespace NUMINAMATH_CALUDE_average_age_of_students_l3744_374462

theorem average_age_of_students (num_students : ℕ) (teacher_age : ℕ) (total_average : ℕ) 
  (h1 : num_students = 40)
  (h2 : teacher_age = 56)
  (h3 : total_average = 16) :
  (num_students * total_average - teacher_age) / num_students = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_students_l3744_374462


namespace NUMINAMATH_CALUDE_child_b_share_l3744_374472

theorem child_b_share (total_amount : ℕ) (ratio_a ratio_b ratio_c : ℕ) : 
  total_amount = 1800 →
  ratio_a = 2 →
  ratio_b = 3 →
  ratio_c = 4 →
  (ratio_b * total_amount) / (ratio_a + ratio_b + ratio_c) = 600 := by
  sorry

end NUMINAMATH_CALUDE_child_b_share_l3744_374472


namespace NUMINAMATH_CALUDE_tangent_line_and_bounds_l3744_374496

/-- The function f(x) = (ax+b)e^(-2x) -/
noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) * Real.exp (-2 * x)

/-- The function g(x) = f(x) + x * ln(x) -/
noncomputable def g (a b x : ℝ) : ℝ := f a b x + x * Real.log x

theorem tangent_line_and_bounds
  (a b : ℝ)
  (h1 : f a b 0 = 1)  -- f(0) = 1 from the tangent line equation
  (h2 : (deriv (f a b)) 0 = -1)  -- f'(0) = -1 from the tangent line equation
  : a = 1 ∧ b = 1 ∧ ∀ x, 0 < x → x < 1 → 2 * Real.exp (-2) - Real.exp (-1) < g a b x ∧ g a b x < 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_bounds_l3744_374496


namespace NUMINAMATH_CALUDE_prime_factorization_equality_l3744_374482

theorem prime_factorization_equality : 5 * 13 * 31 - 2 = 3 * 11 * 61 := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_equality_l3744_374482


namespace NUMINAMATH_CALUDE_probability_consecutive_numbers_l3744_374434

/-- The total number of lottery numbers --/
def total_numbers : ℕ := 90

/-- The number of drawn lottery numbers --/
def drawn_numbers : ℕ := 5

/-- The set of all possible combinations of drawn numbers --/
def all_combinations : ℕ := Nat.choose total_numbers drawn_numbers

/-- The set of combinations with at least one pair of consecutive numbers --/
def consecutive_combinations : ℕ := 9122966

/-- The probability of drawing at least one pair of consecutive numbers --/
theorem probability_consecutive_numbers :
  (consecutive_combinations : ℚ) / all_combinations = 9122966 / 43949268 := by
  sorry

end NUMINAMATH_CALUDE_probability_consecutive_numbers_l3744_374434


namespace NUMINAMATH_CALUDE_replaced_person_age_l3744_374445

/-- Given a group of 10 people, if replacing one person with a 14-year-old
    decreases the average age by 3 years, then the replaced person was 44 years old. -/
theorem replaced_person_age (group_size : ℕ) (new_person_age : ℕ) (avg_decrease : ℕ) :
  group_size = 10 →
  new_person_age = 14 →
  avg_decrease = 3 →
  ∃ (replaced_age : ℕ),
    (group_size * (replaced_age / group_size) - 
     (group_size * ((replaced_age / group_size) - avg_decrease))) =
    (replaced_age - new_person_age) ∧
    replaced_age = 44 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_age_l3744_374445


namespace NUMINAMATH_CALUDE_exists_a_for_f_with_real_domain_and_range_l3744_374409

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x + 1

-- State the theorem
theorem exists_a_for_f_with_real_domain_and_range :
  ∃ a : ℝ, (∀ x : ℝ, ∃ y : ℝ, f a y = x) ∧ (∀ y : ℝ, ∃ x : ℝ, f a x = y) := by
  sorry

end NUMINAMATH_CALUDE_exists_a_for_f_with_real_domain_and_range_l3744_374409


namespace NUMINAMATH_CALUDE_equation_solutions_l3744_374424

theorem equation_solutions : 
  {x : ℝ | (1 / (x^2 + 11*x + 12) + 1 / (x^2 + 2*x + 3) + 1 / (x^2 - 13*x + 14) = 0)} = 
  {-4, -3, 3, 4} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3744_374424


namespace NUMINAMATH_CALUDE_equation_solution_l3744_374412

theorem equation_solution :
  ∀ x : ℝ, Real.sqrt (x + 9) - Real.sqrt (x - 5) - 2 = 0 → x = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3744_374412


namespace NUMINAMATH_CALUDE_tangent_cotangent_identity_l3744_374400

theorem tangent_cotangent_identity (α : Real) 
  (h1 : 0 < α) (h2 : α < π/2) (h3 : α ≠ π/4) :
  (Real.sqrt (Real.tan α) + Real.sqrt (1 / Real.tan α)) / 
  (Real.sqrt (Real.tan α) - Real.sqrt (1 / Real.tan α)) = 
  1 / Real.tan (α - π/4) := by
  sorry

end NUMINAMATH_CALUDE_tangent_cotangent_identity_l3744_374400


namespace NUMINAMATH_CALUDE_angle_in_linear_pair_l3744_374471

/-- 
Given a line segment AB with three angles:
- ACD = 90°
- ECB = 52°
- DCE = x°
Prove that x = 38°
-/
theorem angle_in_linear_pair (x : ℝ) : 
  90 + x + 52 = 180 → x = 38 := by sorry

end NUMINAMATH_CALUDE_angle_in_linear_pair_l3744_374471


namespace NUMINAMATH_CALUDE_contrapositive_sine_not_piecewise_l3744_374449

-- Define the universe of functions
variable (F : Type) [Nonempty F]

-- Define predicates for sine function and piecewise function
variable (is_sine : F → Prop)
variable (is_piecewise : F → Prop)

-- State the theorem
theorem contrapositive_sine_not_piecewise :
  (∀ f : F, is_sine f → ¬ is_piecewise f) ↔
  (∀ f : F, is_piecewise f → ¬ is_sine f) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_sine_not_piecewise_l3744_374449


namespace NUMINAMATH_CALUDE_fraction_sum_non_negative_l3744_374479

theorem fraction_sum_non_negative (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  1 / (a - b) + 1 / (b - c) + 4 / (c - a) ≥ 0 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_non_negative_l3744_374479


namespace NUMINAMATH_CALUDE_percentage_excess_l3744_374423

theorem percentage_excess (x y : ℝ) (h : x = 0.38 * y) :
  (y - x) / x = 0.62 := by
  sorry

end NUMINAMATH_CALUDE_percentage_excess_l3744_374423


namespace NUMINAMATH_CALUDE_max_min_f_a4_range_a_inequality_l3744_374429

-- Define the function f
def f (a x : ℝ) : ℝ := x * abs (x - a) + 2 * x - 3

-- Theorem for part 1
theorem max_min_f_a4 :
  ∃ (max min : ℝ),
    (∀ x, 2 ≤ x ∧ x ≤ 5 → f 4 x ≤ max) ∧
    (∃ x, 2 ≤ x ∧ x ≤ 5 ∧ f 4 x = max) ∧
    (∀ x, 2 ≤ x ∧ x ≤ 5 → min ≤ f 4 x) ∧
    (∃ x, 2 ≤ x ∧ x ≤ 5 ∧ f 4 x = min) ∧
    max = 12 ∧ min = 5 :=
sorry

-- Theorem for part 2
theorem range_a_inequality :
  ∀ a : ℝ,
    (∀ x, 1 ≤ x ∧ x ≤ 2 → f a x ≤ 2 * x - 2) ↔
    (3 / 2 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_max_min_f_a4_range_a_inequality_l3744_374429


namespace NUMINAMATH_CALUDE_sales_tax_reduction_difference_l3744_374499

/-- The difference in sales tax between two rates for a given market price -/
def sales_tax_difference (original_rate new_rate market_price : ℝ) : ℝ :=
  market_price * original_rate - market_price * new_rate

/-- Theorem stating the difference in sales tax for the given problem -/
theorem sales_tax_reduction_difference :
  let original_rate : ℝ := 3.5 / 100
  let new_rate : ℝ := 10 / 3 / 100
  let market_price : ℝ := 7800
  abs (sales_tax_difference original_rate new_rate market_price - 13.26) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_sales_tax_reduction_difference_l3744_374499


namespace NUMINAMATH_CALUDE_parabola_intersection_l3744_374463

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Line structure -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Intersection points -/
structure IntersectionPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Main theorem -/
theorem parabola_intersection (C : Parabola) (l : Line) (I : IntersectionPoints) :
  l.slope = 2 ∧
  l.point = (C.p / 2, 0) ∧
  (I.A.1 - C.p / 2) * (I.A.1 - C.p / 2) + I.A.2 * I.A.2 = 20 ∧
  (I.B.1 - C.p / 2) * (I.B.1 - C.p / 2) + I.B.2 * I.B.2 = 20 ∧
  I.A.2 * I.A.2 = 2 * C.p * I.A.1 ∧
  I.B.2 * I.B.2 = 2 * C.p * I.B.1 →
  C.p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l3744_374463


namespace NUMINAMATH_CALUDE_function_equation_solver_l3744_374441

theorem function_equation_solver (f : ℝ → ℝ) :
  (∀ x, f (x + 1) = x^2 + 4*x + 1) →
  (∀ x, f x = x^2 + 2*x - 2) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solver_l3744_374441


namespace NUMINAMATH_CALUDE_seonhos_wallet_problem_l3744_374444

theorem seonhos_wallet_problem (initial_money : ℚ) : 
  (initial_money / 4) * (1 / 3) = 2500 → initial_money = 10000 := by sorry

end NUMINAMATH_CALUDE_seonhos_wallet_problem_l3744_374444


namespace NUMINAMATH_CALUDE_factorial_fraction_equals_zero_l3744_374453

theorem factorial_fraction_equals_zero : 
  (5 * Nat.factorial 7 - 35 * Nat.factorial 6) / Nat.factorial 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equals_zero_l3744_374453


namespace NUMINAMATH_CALUDE_water_fountain_problem_l3744_374427

/-- Represents the number of men needed to build a water fountain -/
def men_needed (length : ℕ) (days : ℕ) (men : ℕ) : Prop :=
  ∃ (k : ℚ), k * (men * days) = length

theorem water_fountain_problem :
  men_needed 56 42 60 ∧ men_needed 7 3 35 →
  (∀ l₁ d₁ m₁ l₂ d₂ m₂,
    men_needed l₁ d₁ m₁ → men_needed l₂ d₂ m₂ →
    (m₁ * d₁ : ℚ) / l₁ = (m₂ * d₂ : ℚ) / l₂) →
  60 = (35 * 3 * 56) / (7 * 42) :=
by sorry

end NUMINAMATH_CALUDE_water_fountain_problem_l3744_374427


namespace NUMINAMATH_CALUDE_parabola_linear_function_relationship_l3744_374415

-- Define the parabola
def parabola (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

-- Define the linear function
def linear_function (a b : ℝ) (x : ℝ) : ℝ := (a - b) * x + b

theorem parabola_linear_function_relationship 
  (a b m : ℝ) 
  (h1 : a < 0)  -- parabola opens downwards
  (h2 : m < 0)  -- P(-1, m) is in the third quadrant
  (h3 : parabola a b (-1) = m)  -- parabola passes through P(-1, m)
  (h4 : -b / (2*a) < 0)  -- axis of symmetry is negative (P and origin on opposite sides)
  : ∀ x y : ℝ, x > 0 ∧ y > 0 → linear_function a b x ≠ y :=
by sorry

end NUMINAMATH_CALUDE_parabola_linear_function_relationship_l3744_374415


namespace NUMINAMATH_CALUDE_height_difference_l3744_374408

/-- Given height differences between consecutive locations, prove that A-B = 0.4 --/
theorem height_difference (D_A E_D F_E G_F H_G B_H : ℝ) 
  (h1 : D_A = 3.3)
  (h2 : E_D = -4.2)
  (h3 : F_E = -0.5)
  (h4 : G_F = 2.7)
  (h5 : H_G = 3.9)
  (h6 : B_H = -5.6) : 
  A - B = 0.4 := by
  sorry


end NUMINAMATH_CALUDE_height_difference_l3744_374408


namespace NUMINAMATH_CALUDE_yanna_afternoon_biscuits_l3744_374447

/-- The number of butter cookies Yanna baked in the afternoon -/
def afternoon_butter_cookies : ℕ := 10

/-- The difference between biscuits and butter cookies baked in the afternoon -/
def biscuit_cookie_difference : ℕ := 30

/-- The number of biscuits Yanna baked in the afternoon -/
def afternoon_biscuits : ℕ := afternoon_butter_cookies + biscuit_cookie_difference

theorem yanna_afternoon_biscuits : afternoon_biscuits = 40 := by
  sorry

end NUMINAMATH_CALUDE_yanna_afternoon_biscuits_l3744_374447


namespace NUMINAMATH_CALUDE_decimal_expansion_3_11_l3744_374448

theorem decimal_expansion_3_11 : 
  ∃ (n : ℕ) (a b : ℕ), 
    (3 : ℚ) / 11 = (a : ℚ) / (10^n - 1) ∧ 
    b = 10^n - 1 ∧ 
    n = 2 ∧
    a < b := by sorry

end NUMINAMATH_CALUDE_decimal_expansion_3_11_l3744_374448


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3744_374490

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*y + k = 0 → y = x) → 
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3744_374490


namespace NUMINAMATH_CALUDE_cookies_left_l3744_374437

-- Define the number of cookies in a dozen
def cookies_per_dozen : ℕ := 12

-- Define the number of dozens John buys
def dozens_bought : ℕ := 2

-- Define the number of cookies John eats
def cookies_eaten : ℕ := 3

-- Theorem statement
theorem cookies_left : 
  dozens_bought * cookies_per_dozen - cookies_eaten = 21 := by
sorry

end NUMINAMATH_CALUDE_cookies_left_l3744_374437


namespace NUMINAMATH_CALUDE_expansion_coefficient_l3744_374473

/-- The coefficient of x^n in the expansion of (x-1/x)^m -/
def coeff (m n : ℕ) : ℤ :=
  if (m - n) % 2 = 0 
  then (-1)^((m - n) / 2) * (m.choose ((m - n) / 2))
  else 0

/-- The coefficient of x^6 in the expansion of (x^2+a)(x-1/x)^10 -/
def coeff_x6 (a : ℤ) : ℤ := coeff 10 6 + a * coeff 10 4

theorem expansion_coefficient (a : ℤ) : 
  coeff_x6 a = -30 → a = 2 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l3744_374473


namespace NUMINAMATH_CALUDE_g_neg_two_l3744_374480

def g (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem g_neg_two : g (-2) = -1 := by sorry

end NUMINAMATH_CALUDE_g_neg_two_l3744_374480


namespace NUMINAMATH_CALUDE_quadratic_a_value_l3744_374419

/-- A quadratic function with vertex (h, k) passing through point (x₀, y₀) -/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ
  x₀ : ℝ
  y₀ : ℝ
  vertex_form : ∀ x, a * (x - h)^2 + k = a * x^2 + (a * h * (-2)) * x + (a * h^2 + k)
  passes_through : a * (x₀ - h)^2 + k = y₀

/-- The theorem stating that for a quadratic function with vertex (3, 5) passing through (0, -20), a = -25/9 -/
theorem quadratic_a_value (f : QuadraticFunction) 
    (h_vertex : f.h = 3 ∧ f.k = 5) 
    (h_point : f.x₀ = 0 ∧ f.y₀ = -20) : 
    f.a = -25/9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_a_value_l3744_374419


namespace NUMINAMATH_CALUDE_amanda_stroll_time_l3744_374404

/-- Amanda's stroll to Kimberly's house -/
theorem amanda_stroll_time (speed : ℝ) (distance : ℝ) (h1 : speed = 2) (h2 : distance = 6) :
  distance / speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_amanda_stroll_time_l3744_374404


namespace NUMINAMATH_CALUDE_range_of_2x_minus_y_l3744_374435

theorem range_of_2x_minus_y (x y : ℝ) 
  (hx : 0 < x ∧ x < 4) 
  (hy : 0 < y ∧ y < 6) : 
  -6 < 2*x - y ∧ 2*x - y < 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_2x_minus_y_l3744_374435


namespace NUMINAMATH_CALUDE_increasing_function_implies_a_bound_l3744_374428

/-- A function f is increasing on an interval [a, +∞) if for any x₁, x₂ in the interval with x₁ < x₂, we have f(x₁) < f(x₂) -/
def IncreasingOnInterval (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x₁ x₂, a ≤ x₁ ∧ x₁ < x₂ → f x₁ < f x₂

/-- The main theorem stating that if f(x) = x^2 - 2ax + 2 is increasing on [3, +∞), then a ≤ 3 -/
theorem increasing_function_implies_a_bound (a : ℝ) :
  IncreasingOnInterval (fun x => x^2 - 2*a*x + 2) 3 → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_implies_a_bound_l3744_374428


namespace NUMINAMATH_CALUDE_max_product_with_sum_constraint_l3744_374489

theorem max_product_with_sum_constraint :
  ∃ (x : ℤ), 
    (∀ y : ℤ, x * (340 - x) ≥ y * (340 - y)) ∧ 
    (x * (340 - x) > 2000) ∧
    (x * (340 - x) = 28900) := by
  sorry

end NUMINAMATH_CALUDE_max_product_with_sum_constraint_l3744_374489


namespace NUMINAMATH_CALUDE_equation_b_is_quadratic_l3744_374417

/-- A quadratic equation in one variable is an equation that can be written in the form ax² + bx + c = 0, where a ≠ 0 and x is a variable. --/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(y) = 5y² - 5y represents the equation 5y = 5y². --/
def f (y : ℝ) : ℝ := 5 * y^2 - 5 * y

/-- Theorem: The equation 5y = 5y² is a quadratic equation. --/
theorem equation_b_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_equation_b_is_quadratic_l3744_374417


namespace NUMINAMATH_CALUDE_no_valid_tiling_exists_l3744_374457

/-- Represents a chessboard square --/
inductive Square
| Black
| White

/-- Represents a 2x1 domino --/
structure Domino :=
(first : Square)
(second : Square)

/-- Represents the modified 8x8 chessboard with corners removed --/
def ModifiedChessboard := Fin 62 → Square

/-- A tiling of the modified chessboard using dominos --/
def Tiling := Fin 31 → Domino

/-- Checks if a tiling is valid for the modified chessboard --/
def is_valid_tiling (board : ModifiedChessboard) (tiling : Tiling) : Prop :=
  ∀ i j : Fin 62, i ≠ j → 
    ∃ k : Fin 31, (tiling k).first = board i ∧ (tiling k).second = board j

/-- The main theorem stating that no valid tiling exists --/
theorem no_valid_tiling_exists :
  ¬∃ (board : ModifiedChessboard) (tiling : Tiling), is_valid_tiling board tiling :=
sorry

end NUMINAMATH_CALUDE_no_valid_tiling_exists_l3744_374457


namespace NUMINAMATH_CALUDE_min_shots_theorem_l3744_374403

/-- Represents a strategy for shooting at windows -/
def ShootingStrategy (n : ℕ) := ℕ → Fin n

/-- Determines if a shooting strategy is successful for all possible target positions -/
def is_successful_strategy (n : ℕ) (strategy : ShootingStrategy n) : Prop :=
  ∀ (start_pos : Fin n), ∃ (k : ℕ), strategy k = min (start_pos + k) (Fin.last n)

/-- The minimum number of shots needed to guarantee hitting the target -/
def min_shots_needed (n : ℕ) : ℕ := n / 2 + 1

/-- Theorem stating the minimum number of shots needed to guarantee hitting the target -/
theorem min_shots_theorem (n : ℕ) : 
  ∃ (strategy : ShootingStrategy n), is_successful_strategy n strategy ∧ 
  (∀ (other_strategy : ShootingStrategy n), 
    is_successful_strategy n other_strategy → 
    (∃ (k : ℕ), ∀ (i : ℕ), i < k → strategy i = other_strategy i) → 
    k ≥ min_shots_needed n) :=
sorry

end NUMINAMATH_CALUDE_min_shots_theorem_l3744_374403


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3744_374493

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x₀ : ℝ, x₀^2 - 2*x₀ + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3744_374493


namespace NUMINAMATH_CALUDE_limit_exp_sin_ratio_l3744_374487

theorem limit_exp_sin_ratio : 
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → 
    |((Real.exp (2*x) - Real.exp x) / (Real.sin (2*x) - Real.sin x)) - 1| < ε := by
sorry

end NUMINAMATH_CALUDE_limit_exp_sin_ratio_l3744_374487
