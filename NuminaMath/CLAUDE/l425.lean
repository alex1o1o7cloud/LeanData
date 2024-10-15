import Mathlib

namespace NUMINAMATH_CALUDE_free_throw_probability_convergence_l425_42506

/-- Represents the number of successful shots for a given total number of shots -/
def makes : ℕ → ℕ
| 50 => 28
| 100 => 49
| 150 => 78
| 200 => 102
| 300 => 153
| 400 => 208
| 500 => 255
| _ => 0  -- For any other number of shots, we don't have data

/-- Represents the total number of shots taken -/
def shots : List ℕ := [50, 100, 150, 200, 300, 400, 500]

/-- Calculate the make frequency for a given number of shots -/
def makeFrequency (n : ℕ) : ℚ :=
  (makes n : ℚ) / n

/-- The statement to be proved -/
theorem free_throw_probability_convergence :
  ∀ ε > 0, ∃ N, ∀ n ∈ shots, n ≥ N → |makeFrequency n - 51/100| < ε :=
sorry

end NUMINAMATH_CALUDE_free_throw_probability_convergence_l425_42506


namespace NUMINAMATH_CALUDE_handshake_count_l425_42595

/-- The number of handshakes in a convention of gremlins and imps -/
theorem handshake_count (num_gremlins num_imps : ℕ) : 
  num_gremlins = 20 →
  num_imps = 15 →
  (num_gremlins * (num_gremlins - 1)) / 2 + num_gremlins * num_imps = 490 := by
  sorry

#check handshake_count

end NUMINAMATH_CALUDE_handshake_count_l425_42595


namespace NUMINAMATH_CALUDE_sum_of_cubes_representable_l425_42502

theorem sum_of_cubes_representable (a b : ℤ) 
  (h1 : ∃ (x1 y1 : ℤ), a = x1^2 + 3*y1^2) 
  (h2 : ∃ (x2 y2 : ℤ), b = x2^2 + 3*y2^2) : 
  ∃ (x3 y3 : ℤ), a^3 + b^3 = x3^2 + 3*y3^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_representable_l425_42502


namespace NUMINAMATH_CALUDE_total_count_is_six_l425_42576

def problem (total_count : ℕ) (group1_count group2_count group3_count : ℕ) 
  (total_avg group1_avg group2_avg group3_avg : ℚ) : Prop :=
  total_count = group1_count + group2_count + group3_count ∧
  group1_count = 2 ∧
  group2_count = 2 ∧
  group3_count = 2 ∧
  total_avg = 3.95 ∧
  group1_avg = 3.6 ∧
  group2_avg = 3.85 ∧
  group3_avg = 4.400000000000001

theorem total_count_is_six :
  ∃ (total_count : ℕ) (group1_count group2_count group3_count : ℕ)
    (total_avg group1_avg group2_avg group3_avg : ℚ),
  problem total_count group1_count group2_count group3_count
    total_avg group1_avg group2_avg group3_avg ∧
  total_count = 6 :=
by sorry

end NUMINAMATH_CALUDE_total_count_is_six_l425_42576


namespace NUMINAMATH_CALUDE_carpet_border_area_l425_42524

/-- Calculates the area of a carpet border in a rectangular room -/
theorem carpet_border_area 
  (room_length : ℝ) 
  (room_width : ℝ) 
  (border_width : ℝ) 
  (h1 : room_length = 12) 
  (h2 : room_width = 10) 
  (h3 : border_width = 2) : 
  room_length * room_width - (room_length - 2 * border_width) * (room_width - 2 * border_width) = 72 := by
  sorry

#check carpet_border_area

end NUMINAMATH_CALUDE_carpet_border_area_l425_42524


namespace NUMINAMATH_CALUDE_sports_equipment_pricing_and_discount_l425_42553

theorem sports_equipment_pricing_and_discount (soccer_price basketball_price : ℝ)
  (h1 : 2 * soccer_price + 3 * basketball_price = 410)
  (h2 : 5 * soccer_price + 2 * basketball_price = 530)
  (h3 : ∃ discount_rate : ℝ, 
    discount_rate * (5 * soccer_price + 5 * basketball_price) = 680 ∧ 
    0 < discount_rate ∧ 
    discount_rate < 1) :
  soccer_price = 70 ∧ basketball_price = 90 ∧ 
  ∃ discount_rate : ℝ, discount_rate * (5 * 70 + 5 * 90) = 680 ∧ discount_rate = 0.85 := by
sorry

end NUMINAMATH_CALUDE_sports_equipment_pricing_and_discount_l425_42553


namespace NUMINAMATH_CALUDE_cube_volume_derivative_half_surface_area_l425_42569

-- Define a cube with edge length x
def cube_volume (x : ℝ) : ℝ := x^3
def cube_surface_area (x : ℝ) : ℝ := 6 * x^2

-- State the theorem
theorem cube_volume_derivative_half_surface_area :
  ∀ x : ℝ, (deriv cube_volume) x = (1/2) * cube_surface_area x :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_derivative_half_surface_area_l425_42569


namespace NUMINAMATH_CALUDE_inequality_solution_l425_42589

theorem inequality_solution (x : ℝ) : 
  1 / (x^3 + 1) > 4 / x + 2 / 5 ↔ -1 < x ∧ x < 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l425_42589


namespace NUMINAMATH_CALUDE_special_polynomial_value_l425_42518

/-- A polynomial function of degree n satisfying f(k) = k/(k+1) for k = 0, 1, ..., n -/
def SpecialPolynomial (n : ℕ) (f : ℝ → ℝ) : Prop :=
  (∃ p : Polynomial ℝ, Polynomial.degree p = n ∧ f = p.eval) ∧
  (∀ k : ℕ, k ≤ n → f k = k / (k + 1))

/-- The main theorem stating the value of f(n+1) for a SpecialPolynomial -/
theorem special_polynomial_value (n : ℕ) (f : ℝ → ℝ) 
  (h : SpecialPolynomial n f) : 
  f (n + 1) = (n + 1 + (-1)^(n + 1)) / (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_value_l425_42518


namespace NUMINAMATH_CALUDE_min_value_trigonometric_expression_l425_42531

theorem min_value_trigonometric_expression (θ φ : ℝ) :
  (3 * Real.cos θ + 4 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 4 * Real.cos φ - 20)^2 ≥ 235.97 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_expression_l425_42531


namespace NUMINAMATH_CALUDE_specific_arithmetic_series_sum_l425_42568

/-- Sum of an arithmetic series with given parameters -/
def arithmeticSeriesSum (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) : ℤ :=
  let n : ℤ := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem: The sum of the arithmetic series (-42) + (-40) + ⋯ + 0 is -462 -/
theorem specific_arithmetic_series_sum :
  arithmeticSeriesSum (-42) 0 2 = -462 := by
  sorry

end NUMINAMATH_CALUDE_specific_arithmetic_series_sum_l425_42568


namespace NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l425_42552

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The function f(x) = x^2 + (m-1)x - 3 -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  x^2 + (m-1)*x - 3

theorem even_function_implies_m_equals_one :
  ∀ m : ℝ, IsEven (f m) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l425_42552


namespace NUMINAMATH_CALUDE_art_club_election_l425_42564

theorem art_club_election (total_candidates : ℕ) (past_officers : ℕ) (positions : ℕ) 
  (h1 : total_candidates = 18) 
  (h2 : past_officers = 8) 
  (h3 : positions = 6) :
  (Nat.choose total_candidates positions) - 
  (Nat.choose (total_candidates - past_officers) positions) = 18354 := by
  sorry

end NUMINAMATH_CALUDE_art_club_election_l425_42564


namespace NUMINAMATH_CALUDE_sum_of_roots_l425_42577

theorem sum_of_roots (x y : ℝ) 
  (hx : x^3 - 3*x^2 + 2000*x = 1997)
  (hy : y^3 - 3*y^2 + 2000*y = 1999) : 
  x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l425_42577


namespace NUMINAMATH_CALUDE_employee_pay_percentage_l425_42529

/-- Proof that X is paid 120% of Y's pay given the conditions -/
theorem employee_pay_percentage (total pay_y : ℕ) (pay_x : ℕ) 
  (h1 : total = 880)
  (h2 : pay_y = 400)
  (h3 : pay_x + pay_y = total) :
  (pay_x : ℚ) / pay_y = 120 / 100 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_percentage_l425_42529


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_l425_42566

-- Define the functions f and g
def f (A B x : ℝ) : ℝ := A * x + B
def g (A B x : ℝ) : ℝ := B * x + A

-- State the theorem
theorem sum_of_A_and_B 
  (A B : ℝ) 
  (h1 : A ≠ B) 
  (h2 : B - A = 2) 
  (h3 : ∀ x, f A B (g A B x) - g A B (f A B x) = B^2 - A^2) : 
  A + B = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_l425_42566


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l425_42571

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36 →
  (a 5 + a 7 = 6 ∨ a 5 + a 7 = -6) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l425_42571


namespace NUMINAMATH_CALUDE_test_scores_l425_42598

theorem test_scores (keith_score : Real) (larry_multiplier : Real) (danny_difference : Real)
  (h1 : keith_score = 3.5)
  (h2 : larry_multiplier = 3.2)
  (h3 : danny_difference = 5.7) :
  let larry_score := keith_score * larry_multiplier
  let danny_score := larry_score + danny_difference
  keith_score + larry_score + danny_score = 31.6 := by
  sorry

end NUMINAMATH_CALUDE_test_scores_l425_42598


namespace NUMINAMATH_CALUDE_f_one_intersection_l425_42505

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (3*a - 2)*x + a - 1

/-- Theorem stating the condition for f(x) to have exactly one intersection with x-axis in (-1,3) -/
theorem f_one_intersection (a : ℝ) : 
  (∃! x : ℝ, x > -1 ∧ x < 3 ∧ f a x = 0) ↔ (a ≤ -1/5 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_f_one_intersection_l425_42505


namespace NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l425_42560

/-- The polynomial z^6 - z^3 + 1 -/
def f (z : ℂ) : ℂ := z^6 - z^3 + 1

/-- The set of roots of f(z) -/
def roots_of_f : Set ℂ := {z : ℂ | f z = 0}

/-- n-th roots of unity -/
def nth_roots_of_unity (n : ℕ) : Set ℂ := {z : ℂ | z^n = 1}

/-- Theorem: 9 is the smallest positive integer n such that all roots of z^6 - z^3 + 1 = 0 are n-th roots of unity -/
theorem smallest_n_for_roots_of_unity :
  ∃ (n : ℕ), n > 0 ∧ roots_of_f ⊆ nth_roots_of_unity n ∧
  ∀ (m : ℕ), m > 0 ∧ m < n → ¬(roots_of_f ⊆ nth_roots_of_unity m) ∧
  n = 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l425_42560


namespace NUMINAMATH_CALUDE_virus_reaches_64MB_l425_42530

/-- Represents the memory occupation of a virus over time -/
def virusMemory (t : ℕ) : ℕ :=
  2 * 2^t

/-- Represents the time in minutes since boot -/
def timeInMinutes (n : ℕ) : ℕ :=
  3 * n

/-- Theorem stating that the virus occupies 64MB after 45 minutes -/
theorem virus_reaches_64MB :
  ∃ n : ℕ, virusMemory n = 64 * 2^10 ∧ timeInMinutes n = 45 := by
  sorry

end NUMINAMATH_CALUDE_virus_reaches_64MB_l425_42530


namespace NUMINAMATH_CALUDE_tyler_saltwater_animals_l425_42556

/-- The number of aquariums Tyler has -/
def num_aquariums : ℕ := 8

/-- The number of animals in each aquarium -/
def animals_per_aquarium : ℕ := 64

/-- The total number of saltwater animals Tyler has -/
def total_animals : ℕ := num_aquariums * animals_per_aquarium

/-- Theorem stating that the total number of saltwater animals Tyler has is 512 -/
theorem tyler_saltwater_animals : total_animals = 512 := by
  sorry

end NUMINAMATH_CALUDE_tyler_saltwater_animals_l425_42556


namespace NUMINAMATH_CALUDE_integral_sqrt_minus_sin_equals_pi_l425_42526

open Set
open MeasureTheory
open Interval

theorem integral_sqrt_minus_sin_equals_pi :
  ∫ x in (Icc (-1) 1), (2 * Real.sqrt (1 - x^2) - Real.sin x) = π := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_minus_sin_equals_pi_l425_42526


namespace NUMINAMATH_CALUDE_cell_growth_proof_l425_42558

/-- The time interval between cell divisions in minutes -/
def division_interval : ℕ := 20

/-- The total time elapsed in minutes -/
def total_time : ℕ := 3 * 60 + 20

/-- The number of cells after one division -/
def cells_after_division : ℕ := 2

/-- The number of cells after a given number of divisions -/
def cells_after_divisions (n : ℕ) : ℕ := cells_after_division ^ n

theorem cell_growth_proof :
  cells_after_divisions (total_time / division_interval) = 1024 :=
by sorry

end NUMINAMATH_CALUDE_cell_growth_proof_l425_42558


namespace NUMINAMATH_CALUDE_marble_distribution_l425_42578

theorem marble_distribution (y : ℚ) : 
  (4 * y + 2) + (2 * y) + (y + 3) = 31 → y = 26 / 7 := by
sorry

end NUMINAMATH_CALUDE_marble_distribution_l425_42578


namespace NUMINAMATH_CALUDE_carmen_paint_area_l425_42547

/-- Represents the dimensions of a room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total wall area to be painted in Carmen's house -/
def total_paint_area (num_rooms : ℕ) (room_dims : RoomDimensions) (unpainted_area : ℝ) : ℝ :=
  let wall_area := 2 * (room_dims.length * room_dims.height + room_dims.width * room_dims.height)
  let paintable_area := wall_area - unpainted_area
  (num_rooms : ℝ) * paintable_area

/-- Theorem stating that the total area to be painted in Carmen's house is 1408 square feet -/
theorem carmen_paint_area :
  let room_dims : RoomDimensions := ⟨15, 12, 8⟩
  let num_rooms : ℕ := 4
  let unpainted_area : ℝ := 80
  total_paint_area num_rooms room_dims unpainted_area = 1408 := by
  sorry

end NUMINAMATH_CALUDE_carmen_paint_area_l425_42547


namespace NUMINAMATH_CALUDE_total_pets_is_54_l425_42540

/-- The number of pets owned by Teddy, Ben, and Dave -/
def total_pets : ℕ :=
  let teddy_dogs : ℕ := 7
  let teddy_cats : ℕ := 8
  let ben_extra_dogs : ℕ := 9
  let dave_extra_cats : ℕ := 13
  let dave_fewer_dogs : ℕ := 5

  let teddy_total : ℕ := teddy_dogs + teddy_cats
  let ben_total : ℕ := (teddy_dogs + ben_extra_dogs)
  let dave_total : ℕ := (teddy_cats + dave_extra_cats) + (teddy_dogs - dave_fewer_dogs)

  teddy_total + ben_total + dave_total

/-- Theorem stating that the total number of pets is 54 -/
theorem total_pets_is_54 : total_pets = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_pets_is_54_l425_42540


namespace NUMINAMATH_CALUDE_parabola_equation_l425_42593

/-- A parabola with vertex at the origin and axis of symmetry x = 2 -/
structure Parabola where
  /-- The equation of the parabola in the form y² = -2px -/
  equation : ℝ → ℝ → Prop
  /-- The vertex of the parabola is at the origin -/
  vertex_at_origin : equation 0 0
  /-- The axis of symmetry is x = 2 -/
  axis_of_symmetry : ∀ y, equation 2 y ↔ equation 2 (-y)

/-- The equation of the parabola is y² = -8x -/
theorem parabola_equation (p : Parabola) : 
  p.equation = fun x y => y^2 = -8*x :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l425_42593


namespace NUMINAMATH_CALUDE_part_one_part_two_l425_42501

-- Definition of balanced numbers
def balanced (a b n : ℤ) : Prop := a + b = n

-- Part 1
theorem part_one : balanced (-6) 8 2 := by sorry

-- Part 2
theorem part_two (k : ℤ) (h : ∀ x : ℤ, ∃ n : ℤ, balanced (6*x^2 - 4*k*x + 8) (-2*(3*x^2 - 2*x + k)) n) :
  ∃ n : ℤ, (∀ x : ℤ, balanced (6*x^2 - 4*k*x + 8) (-2*(3*x^2 - 2*x + k)) n) ∧ n = 6 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l425_42501


namespace NUMINAMATH_CALUDE_inequality_proof_l425_42582

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a * b + b * c + c * a = 3) :
  (a^2 / (1 + b * c)) + (b^2 / (1 + c * a)) + (c^2 / (1 + a * b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l425_42582


namespace NUMINAMATH_CALUDE_tulip_ratio_l425_42519

/-- Given the number of red tulips for eyes and smile, and the total number of tulips,
    prove that the ratio of yellow tulips in the background to red tulips in the smile is 9:1 -/
theorem tulip_ratio (red_tulips_per_eye : ℕ) (red_tulips_smile : ℕ) (total_tulips : ℕ) :
  red_tulips_per_eye = 8 →
  red_tulips_smile = 18 →
  total_tulips = 196 →
  (total_tulips - (2 * red_tulips_per_eye + red_tulips_smile)) / red_tulips_smile = 9 := by
  sorry

end NUMINAMATH_CALUDE_tulip_ratio_l425_42519


namespace NUMINAMATH_CALUDE_polynomial_factorization_l425_42543

theorem polynomial_factorization (x y : ℝ) : 
  x^3 - 4*x^2*y + 4*x*y^2 = x*(x - 2*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l425_42543


namespace NUMINAMATH_CALUDE_andys_future_age_ratio_l425_42521

def rahims_current_age : ℕ := 6
def age_difference : ℕ := 1
def years_in_future : ℕ := 5

theorem andys_future_age_ratio :
  (rahims_current_age + age_difference + years_in_future) / rahims_current_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_andys_future_age_ratio_l425_42521


namespace NUMINAMATH_CALUDE_max_students_equal_distribution_l425_42591

theorem max_students_equal_distribution (pens pencils : ℕ) 
  (h_pens : pens = 640) (h_pencils : pencils = 520) :
  Nat.gcd pens pencils = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_students_equal_distribution_l425_42591


namespace NUMINAMATH_CALUDE_odd_function_a_value_l425_42597

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x < 0 then 2^x - a*x else -(2^(-x)) - a*x

-- State the theorem
theorem odd_function_a_value :
  -- f is an odd function
  (∀ x, f a x = -(f a (-x))) →
  -- f(x) = 2^x - ax when x < 0
  (∀ x, x < 0 → f a x = 2^x - a*x) →
  -- f(2) = 2
  f a 2 = 2 →
  -- Then a = -9/8
  a = -9/8 :=
sorry

end NUMINAMATH_CALUDE_odd_function_a_value_l425_42597


namespace NUMINAMATH_CALUDE_binomial_cube_constant_l425_42586

theorem binomial_cube_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 27 * x^3 + 9 * x^2 + 36 * x + a = (3 * x + b)^3) → 
  a = 8 := by
sorry

end NUMINAMATH_CALUDE_binomial_cube_constant_l425_42586


namespace NUMINAMATH_CALUDE_liam_commute_speed_l425_42509

theorem liam_commute_speed (distance : ℝ) (actual_speed : ℝ) (early_time : ℝ) 
  (h1 : distance = 40)
  (h2 : actual_speed = 60)
  (h3 : early_time = 4/60) :
  let ideal_speed := actual_speed - 5
  let actual_time := distance / actual_speed
  let ideal_time := distance / ideal_speed
  ideal_time - actual_time = early_time := by sorry

end NUMINAMATH_CALUDE_liam_commute_speed_l425_42509


namespace NUMINAMATH_CALUDE_ball_selection_probabilities_l425_42581

/-- Represents a bag of balls -/
structure Bag where
  white : ℕ
  red : ℕ

/-- The probability of drawing a white ball from a bag -/
def probWhite (bag : Bag) : ℚ :=
  bag.white / (bag.white + bag.red)

/-- The probability of drawing a red ball from a bag -/
def probRed (bag : Bag) : ℚ :=
  bag.red / (bag.white + bag.red)

/-- The bags used in the problem -/
def bagA : Bag := ⟨8, 4⟩
def bagB : Bag := ⟨6, 6⟩

/-- The theorem to be proved -/
theorem ball_selection_probabilities :
  /- The probability of selecting two balls of the same color is 1/2 -/
  (probWhite bagA * probWhite bagB + probRed bagA * probRed bagB = 1/2) ∧
  /- The probability of selecting at least one red ball is 2/3 -/
  (1 - probWhite bagA * probWhite bagB = 2/3) := by
  sorry


end NUMINAMATH_CALUDE_ball_selection_probabilities_l425_42581


namespace NUMINAMATH_CALUDE_ceiling_distance_to_square_existence_l425_42583

theorem ceiling_distance_to_square_existence : 
  ∃ (A : ℝ), ∀ (n : ℕ), 
    ∃ (m : ℕ), (⌈A^n⌉ : ℝ) - (m^2 : ℝ) = 2 ∧ 
    ∀ (k : ℕ), k > m → (k^2 : ℝ) > ⌈A^n⌉ :=
by sorry

end NUMINAMATH_CALUDE_ceiling_distance_to_square_existence_l425_42583


namespace NUMINAMATH_CALUDE_chess_players_lost_to_ai_l425_42515

theorem chess_players_lost_to_ai (total_players : ℕ) (never_lost_fraction : ℚ) : 
  total_players = 120 →
  never_lost_fraction = 2 / 5 →
  (total_players : ℚ) * (1 - never_lost_fraction) = 72 := by
  sorry

end NUMINAMATH_CALUDE_chess_players_lost_to_ai_l425_42515


namespace NUMINAMATH_CALUDE_students_liking_both_subjects_l425_42546

theorem students_liking_both_subjects (total : ℕ) (math : ℕ) (english : ℕ) (neither : ℕ) 
  (h1 : total = 48)
  (h2 : math = 38)
  (h3 : english = 36)
  (h4 : neither = 4) :
  math + english - (total - neither) = 30 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_subjects_l425_42546


namespace NUMINAMATH_CALUDE_min_sum_xyz_l425_42575

theorem min_sum_xyz (x y z : ℤ) (h : (x - 10) * (y - 5) * (z - 2) = 1000) :
  ∀ (a b c : ℤ), (a - 10) * (b - 5) * (c - 2) = 1000 → x + y + z ≤ a + b + c :=
by sorry

end NUMINAMATH_CALUDE_min_sum_xyz_l425_42575


namespace NUMINAMATH_CALUDE_birds_on_fence_two_plus_four_birds_l425_42588

/-- The number of birds on a fence after more birds join -/
theorem birds_on_fence (initial : Nat) (joined : Nat) : 
  initial + joined = initial + joined :=
by sorry

/-- The specific case of 2 initial birds and 4 joined birds -/
theorem two_plus_four_birds : 2 + 4 = 6 :=
by sorry

end NUMINAMATH_CALUDE_birds_on_fence_two_plus_four_birds_l425_42588


namespace NUMINAMATH_CALUDE_exists_coprime_in_ten_consecutive_integers_l425_42513

theorem exists_coprime_in_ten_consecutive_integers (n : ℤ) :
  ∃ k ∈ Finset.range 10, ∀ m ∈ Finset.range 10, m ≠ k → Int.gcd (n + k) (n + m) = 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_coprime_in_ten_consecutive_integers_l425_42513


namespace NUMINAMATH_CALUDE_function_characterization_l425_42525

/-- A function from ℚ × ℚ to ℚ satisfying the given property -/
def FunctionProperty (f : ℚ × ℚ → ℚ) : Prop :=
  ∀ x y z : ℚ, f (x, y) + f (y, z) + f (z, x) = f (0, x + y + z)

/-- The theorem stating the form of functions satisfying the property -/
theorem function_characterization (f : ℚ × ℚ → ℚ) (h : FunctionProperty f) :
    ∃ a b : ℚ, ∀ x y : ℚ, f (x, y) = a * y^2 + 2 * a * x * y + b * y :=
  sorry

end NUMINAMATH_CALUDE_function_characterization_l425_42525


namespace NUMINAMATH_CALUDE_cheeseBalls35ozBarrel_l425_42557

/-- Calculates the number of cheese balls in a barrel given its size in ounces -/
def cheeseBallsInBarrel (barrelSize : ℕ) : ℕ :=
  let servingsIn24oz : ℕ := 60
  let cheeseBallsPerServing : ℕ := 12
  let cheeseBallsPer24oz : ℕ := servingsIn24oz * cheeseBallsPerServing
  let cheeseBallsPerOz : ℕ := cheeseBallsPer24oz / 24
  barrelSize * cheeseBallsPerOz

theorem cheeseBalls35ozBarrel :
  cheeseBallsInBarrel 35 = 1050 :=
by sorry

end NUMINAMATH_CALUDE_cheeseBalls35ozBarrel_l425_42557


namespace NUMINAMATH_CALUDE_circle_power_theorem_l425_42523

/-- Given a circle with center O and radius R, and points A and B on the circle,
    for any point P on line AB, PA * PB = OP^2 - R^2 in terms of algebraic lengths -/
theorem circle_power_theorem (O : ℝ × ℝ) (R : ℝ) (A B P : ℝ × ℝ) :
  (∀ X : ℝ × ℝ, dist O X = R → (X = A ∨ X = B)) →
  (∃ t : ℝ, P = (1 - t) • A + t • B) →
  (dist P A * dist P B : ℝ) = dist O P ^ 2 - R ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_power_theorem_l425_42523


namespace NUMINAMATH_CALUDE_winnie_lollipop_distribution_l425_42548

theorem winnie_lollipop_distribution (total_lollipops : ℕ) (friends : ℕ) 
  (h1 : total_lollipops = 37 + 108 + 8 + 254) 
  (h2 : friends = 13) : 
  total_lollipops % friends = 4 := by
  sorry

end NUMINAMATH_CALUDE_winnie_lollipop_distribution_l425_42548


namespace NUMINAMATH_CALUDE_socks_selection_with_red_l425_42580

def total_socks : ℕ := 10
def red_socks : ℕ := 1
def socks_to_choose : ℕ := 4

theorem socks_selection_with_red :
  (Nat.choose total_socks socks_to_choose) - 
  (Nat.choose (total_socks - red_socks) socks_to_choose) = 84 := by
  sorry

end NUMINAMATH_CALUDE_socks_selection_with_red_l425_42580


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l425_42508

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, m₁ * x - y + b₁ = 0 ↔ m₂ * x + y + b₂ = 0) ↔ m₁ = -m₂

/-- The value of a when two lines are parallel -/
theorem parallel_lines_a_value :
  (∀ x y : ℝ, 2 * x - y + 1 = 0 ↔ x + a * y + 2 = 0) → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l425_42508


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l425_42510

/-- The segment interval for systematic sampling given a population and sample size -/
def segment_interval (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

/-- Theorem: The segment interval for systematic sampling of 100 students from a population of 2400 is 24 -/
theorem systematic_sampling_interval :
  segment_interval 2400 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l425_42510


namespace NUMINAMATH_CALUDE_book_problem_solution_l425_42563

def book_problem (total_cost selling_price_1 cost_1 loss_percent : ℚ) : Prop :=
  let cost_2 := total_cost - cost_1
  let selling_price_2 := selling_price_1
  let gain_percent := (selling_price_2 - cost_2) / cost_2 * 100
  (total_cost = 540) ∧
  (cost_1 = 315) ∧
  (selling_price_1 = cost_1 * (1 - loss_percent / 100)) ∧
  (loss_percent = 15) ∧
  (gain_percent = 19)

theorem book_problem_solution :
  ∃ (total_cost selling_price_1 cost_1 loss_percent : ℚ),
    book_problem total_cost selling_price_1 cost_1 loss_percent :=
sorry

end NUMINAMATH_CALUDE_book_problem_solution_l425_42563


namespace NUMINAMATH_CALUDE_prob_missing_one_equals_two_prob_decreasing_sequence_l425_42551

-- Define the number of items in the collection
def n : ℕ := 10

-- Define the probability of finding each item
def p : ℝ := 0.1

-- Define the probability of missing exactly k items in the second set
-- when the first set is completed
noncomputable def p_k (k : ℕ) : ℝ := sorry

-- Theorem 1: p_1 = p_2
theorem prob_missing_one_equals_two : p_k 1 = p_k 2 := sorry

-- Theorem 2: p_2 > p_3 > p_4 > ... > p_10
theorem prob_decreasing_sequence : 
  ∀ k₁ k₂ : ℕ, 2 ≤ k₁ → k₁ < k₂ → k₂ ≤ n → p_k k₁ > p_k k₂ := sorry

end NUMINAMATH_CALUDE_prob_missing_one_equals_two_prob_decreasing_sequence_l425_42551


namespace NUMINAMATH_CALUDE_danny_shorts_count_l425_42545

/-- Represents the number of clothes washed by Cally and Danny -/
structure ClothesWashed where
  cally_white_shirts : Nat
  cally_colored_shirts : Nat
  cally_shorts : Nat
  cally_pants : Nat
  danny_white_shirts : Nat
  danny_colored_shirts : Nat
  danny_pants : Nat
  total_clothes : Nat

/-- Theorem stating that Danny washed 10 pairs of shorts -/
theorem danny_shorts_count (cw : ClothesWashed)
    (h1 : cw.cally_white_shirts = 10)
    (h2 : cw.cally_colored_shirts = 5)
    (h3 : cw.cally_shorts = 7)
    (h4 : cw.cally_pants = 6)
    (h5 : cw.danny_white_shirts = 6)
    (h6 : cw.danny_colored_shirts = 8)
    (h7 : cw.danny_pants = 6)
    (h8 : cw.total_clothes = 58) :
    ∃ (danny_shorts : Nat), danny_shorts = 10 ∧
    cw.total_clothes = cw.cally_white_shirts + cw.cally_colored_shirts + cw.cally_shorts + cw.cally_pants +
                       cw.danny_white_shirts + cw.danny_colored_shirts + danny_shorts + cw.danny_pants :=
  by sorry


end NUMINAMATH_CALUDE_danny_shorts_count_l425_42545


namespace NUMINAMATH_CALUDE_combined_share_A_and_C_l425_42528

def total_amount : ℚ := 15800
def charity_percentage : ℚ := 10 / 100
def savings_percentage : ℚ := 8 / 100
def distribution_ratio : List ℚ := [5, 9, 6, 5]

def remaining_amount : ℚ := total_amount * (1 - charity_percentage - savings_percentage)

def share (ratio : ℚ) : ℚ := (ratio / (distribution_ratio.sum)) * remaining_amount

theorem combined_share_A_and_C : 
  share (distribution_ratio[0]!) + share (distribution_ratio[2]!) = 5700.64 := by
  sorry

end NUMINAMATH_CALUDE_combined_share_A_and_C_l425_42528


namespace NUMINAMATH_CALUDE_sin_cos_fourth_power_sum_l425_42544

theorem sin_cos_fourth_power_sum (θ : Real) (h : Real.sin (2 * θ) = 1 / 2) :
  Real.sin θ ^ 4 + Real.cos θ ^ 4 = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_fourth_power_sum_l425_42544


namespace NUMINAMATH_CALUDE_det_cyclic_matrix_zero_l425_42573

theorem det_cyclic_matrix_zero (p q r : ℝ) (a b c d : ℝ) : 
  (a^4 + p*a^2 + q*a + r = 0) →
  (b^4 + p*b^2 + q*b + r = 0) →
  (c^4 + p*c^2 + q*c + r = 0) →
  (d^4 + p*d^2 + q*d + r = 0) →
  Matrix.det (
    ![![a, b, c, d],
      ![b, c, d, a],
      ![c, d, a, b],
      ![d, a, b, c]]
  ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_det_cyclic_matrix_zero_l425_42573


namespace NUMINAMATH_CALUDE_brandon_cash_sales_l425_42520

theorem brandon_cash_sales (total_sales : ℝ) (credit_ratio : ℝ) (cash_sales : ℝ) : 
  total_sales = 80 →
  credit_ratio = 2/5 →
  cash_sales = total_sales * (1 - credit_ratio) →
  cash_sales = 48 := by
sorry

end NUMINAMATH_CALUDE_brandon_cash_sales_l425_42520


namespace NUMINAMATH_CALUDE_pool_supply_problem_l425_42594

theorem pool_supply_problem (x : ℕ) (h1 : x + 3 * x = 800) : x = 266 := by
  sorry

end NUMINAMATH_CALUDE_pool_supply_problem_l425_42594


namespace NUMINAMATH_CALUDE_intersection_distance_squared_l425_42512

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 5)^2 = 9
def circle3 (x y : ℝ) : Prop := (x - 5)^2 + (y - 2)^2 = 16

-- Define the intersection points
def intersection (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y

-- Theorem statement
theorem intersection_distance_squared :
  ∃ (x1 y1 x2 y2 : ℝ),
    intersection x1 y1 ∧
    intersection x2 y2 ∧
    circle3 x1 y1 ∧
    circle3 x2 y2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 224 / 9 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_l425_42512


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l425_42584

/-- Given a circle with equation (x-1)^2+(y+1)^2=4, its symmetric circle with respect to the origin has the equation (x+1)^2+(y-1)^2=4 -/
theorem symmetric_circle_equation (x y : ℝ) : 
  (∀ x y, (x - 1)^2 + (y + 1)^2 = 4) →
  (∀ x y, (x + 1)^2 + (y - 1)^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l425_42584


namespace NUMINAMATH_CALUDE_marbles_lost_found_difference_l425_42503

/-- Given Josh's marble collection scenario, prove the difference between lost and found marbles. -/
theorem marbles_lost_found_difference (initial : ℕ) (lost : ℕ) (found : ℕ) 
  (h1 : initial = 4)
  (h2 : lost = 16)
  (h3 : found = 8) :
  lost - found = 8 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_found_difference_l425_42503


namespace NUMINAMATH_CALUDE_unique_solution_l425_42539

def satisfiesConditions (n : ℕ) : Prop :=
  ∃ k m p : ℕ, n = 2 * k + 1 ∧ n = 3 * m - 1 ∧ n = 5 * p + 2

theorem unique_solution : 
  satisfiesConditions 47 ∧ 
  (¬ satisfiesConditions 39) ∧ 
  (¬ satisfiesConditions 40) ∧ 
  (¬ satisfiesConditions 49) ∧ 
  (¬ satisfiesConditions 53) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l425_42539


namespace NUMINAMATH_CALUDE_soccer_ball_max_height_l425_42599

/-- The height function of a soccer ball's path -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

/-- The maximum height reached by the soccer ball -/
def max_height : ℝ := 40

theorem soccer_ball_max_height :
  ∀ t : ℝ, h t ≤ max_height :=
sorry

end NUMINAMATH_CALUDE_soccer_ball_max_height_l425_42599


namespace NUMINAMATH_CALUDE_correct_number_value_l425_42500

/-- Given 10 numbers with an initial average of 21, where one number was wrongly read as 26,
    and the correct average is 22, prove that the correct value of the wrongly read number is 36. -/
theorem correct_number_value (n : ℕ) (initial_avg correct_avg wrong_value : ℚ) :
  n = 10 ∧ 
  initial_avg = 21 ∧ 
  correct_avg = 22 ∧ 
  wrong_value = 26 →
  ∃ (correct_value : ℚ), 
    n * correct_avg - (n * initial_avg - wrong_value) = correct_value ∧
    correct_value = 36 :=
by sorry

end NUMINAMATH_CALUDE_correct_number_value_l425_42500


namespace NUMINAMATH_CALUDE_power_of_product_l425_42516

theorem power_of_product (a b : ℝ) : (-5 * a^3 * b)^2 = 25 * a^6 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l425_42516


namespace NUMINAMATH_CALUDE_optimal_selling_price_l425_42538

/-- Represents the problem of finding the optimal selling price for a product --/
structure PricingProblem where
  costPrice : ℝ        -- Cost price per kilogram
  initialPrice : ℝ     -- Initial selling price per kilogram
  initialSales : ℝ     -- Initial monthly sales in kilograms
  salesDecrease : ℝ    -- Decrease in sales per 1 yuan price increase
  availableCapital : ℝ -- Available capital
  targetProfit : ℝ     -- Target profit

/-- Calculates the profit for a given selling price --/
def calculateProfit (p : PricingProblem) (sellingPrice : ℝ) : ℝ :=
  let salesVolume := p.initialSales - (sellingPrice - p.initialPrice) * p.salesDecrease
  (sellingPrice - p.costPrice) * salesVolume

/-- Checks if the capital required for a given selling price is within the available capital --/
def isCapitalSufficient (p : PricingProblem) (sellingPrice : ℝ) : Prop :=
  let salesVolume := p.initialSales - (sellingPrice - p.initialPrice) * p.salesDecrease
  p.costPrice * salesVolume ≤ p.availableCapital

/-- Theorem stating that the optimal selling price is 80 yuan --/
theorem optimal_selling_price (p : PricingProblem) 
  (h1 : p.costPrice = 40)
  (h2 : p.initialPrice = 50)
  (h3 : p.initialSales = 500)
  (h4 : p.salesDecrease = 10)
  (h5 : p.availableCapital = 10000)
  (h6 : p.targetProfit = 8000) :
  ∃ (x : ℝ), x = 80 ∧ 
    calculateProfit p x = p.targetProfit ∧ 
    isCapitalSufficient p x ∧
    ∀ (y : ℝ), y ≠ x → calculateProfit p y = p.targetProfit → ¬(isCapitalSufficient p y) := by
  sorry


end NUMINAMATH_CALUDE_optimal_selling_price_l425_42538


namespace NUMINAMATH_CALUDE_min_distance_C₁_C₂_l425_42587

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop :=
  (x - Real.sqrt 3 / 2)^2 + (y - 1/2)^2 = 1

-- Define the line C₂
def C₂ (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + y - 8 = 0

-- State the theorem
theorem min_distance_C₁_C₂ :
  ∃ d : ℝ, d = 2 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    C₁ x₁ y₁ → C₂ x₂ y₂ →
    d ≤ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) :=
sorry

end NUMINAMATH_CALUDE_min_distance_C₁_C₂_l425_42587


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l425_42559

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 3 > 0} = {x : ℝ | x > 3/2 ∨ x < -1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l425_42559


namespace NUMINAMATH_CALUDE_product_of_fractions_l425_42567

theorem product_of_fractions : (2 : ℚ) / 9 * 5 / 11 = 10 / 99 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l425_42567


namespace NUMINAMATH_CALUDE_constant_speed_motion_not_correlation_l425_42549

/-- Definition of a correlation relationship -/
def correlation_relationship (X Y : Type) (f : X → Y) :=
  ∃ (pattern : X → Set Y), ∀ x : X, f x ∈ pattern x ∧ ¬ (∃ y : Y, pattern x = {y})

/-- Definition of a functional relationship -/
def functional_relationship (X Y : Type) (f : X → Y) :=
  ∀ x : X, ∃! y : Y, f x = y

/-- Distance as a function of time for constant speed motion -/
def distance (v : ℝ) (t : ℝ) : ℝ := v * t

theorem constant_speed_motion_not_correlation :
  ∀ v : ℝ, v > 0 → ¬ (correlation_relationship ℝ ℝ (distance v)) :=
sorry

end NUMINAMATH_CALUDE_constant_speed_motion_not_correlation_l425_42549


namespace NUMINAMATH_CALUDE_paths_from_A_to_D_l425_42555

/-- Represents a point in the network -/
inductive Point : Type
| A : Point
| B : Point
| C : Point
| D : Point

/-- Represents the number of direct paths between two points -/
def direct_paths (p q : Point) : ℕ :=
  match p, q with
  | Point.A, Point.B => 2
  | Point.B, Point.C => 2
  | Point.C, Point.D => 2
  | Point.A, Point.C => 1
  | _, _ => 0

/-- The total number of paths from A to D -/
def total_paths : ℕ := 10

theorem paths_from_A_to_D :
  total_paths = 
    (direct_paths Point.A Point.B * direct_paths Point.B Point.C * direct_paths Point.C Point.D) +
    (direct_paths Point.A Point.C * direct_paths Point.C Point.D) :=
by sorry

end NUMINAMATH_CALUDE_paths_from_A_to_D_l425_42555


namespace NUMINAMATH_CALUDE_qian_receives_23_yuan_l425_42504

/-- Represents the amount of money paid by each person for each meal -/
structure MealPayments where
  zhao_lunch : ℕ
  qian_lunch : ℕ
  sun_lunch : ℕ
  zhao_dinner : ℕ
  qian_dinner : ℕ

/-- Calculates the amount Qian should receive from Li -/
def amount_qian_receives (payments : MealPayments) : ℕ :=
  let total_cost := payments.zhao_lunch + payments.qian_lunch + payments.sun_lunch +
                    payments.zhao_dinner + payments.qian_dinner
  let cost_per_person := total_cost / 4
  let qian_paid := payments.qian_lunch + payments.qian_dinner
  qian_paid - cost_per_person

/-- The main theorem stating that Qian should receive 23 yuan from Li -/
theorem qian_receives_23_yuan (payments : MealPayments) 
  (h1 : payments.zhao_lunch = 23)
  (h2 : payments.qian_lunch = 41)
  (h3 : payments.sun_lunch = 56)
  (h4 : payments.zhao_dinner = 48)
  (h5 : payments.qian_dinner = 32) :
  amount_qian_receives payments = 23 := by
  sorry


end NUMINAMATH_CALUDE_qian_receives_23_yuan_l425_42504


namespace NUMINAMATH_CALUDE_sum_abs_difference_l425_42561

theorem sum_abs_difference : ∀ (a b : ℤ), a = -5 ∧ b = -4 → abs a + abs b - (a + b) = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_abs_difference_l425_42561


namespace NUMINAMATH_CALUDE_vector_angle_difference_l425_42535

theorem vector_angle_difference (α β : Real) (a b : Fin 2 → Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π)
  (ha : a = λ i => if i = 0 then Real.cos α else Real.sin α)
  (hb : b = λ i => if i = 0 then Real.cos β else Real.sin β)
  (h_eq : ‖(2 : Real) • a + b‖ = ‖a - (2 : Real) • b‖) :
  β - α = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_angle_difference_l425_42535


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l425_42550

theorem arithmetic_sequence_problem (a : ℚ) : 
  a > 0 ∧ 
  (∃ d : ℚ, 140 + d = a ∧ a + d = 45/28) → 
  a = 3965/56 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l425_42550


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_difference_1004_1001_l425_42507

/-- Given an arithmetic sequence with first term 3 and common difference 7,
    the positive difference between the 1004th term and the 1001st term is 21. -/
theorem arithmetic_sequence_difference : ℕ → ℕ :=
  fun n => 3 + (n - 1) * 7

#check arithmetic_sequence_difference 1004 - arithmetic_sequence_difference 1001 = 21

theorem difference_1004_1001 :
  arithmetic_sequence_difference 1004 - arithmetic_sequence_difference 1001 = 21 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_difference_1004_1001_l425_42507


namespace NUMINAMATH_CALUDE_cat_and_mouse_positions_l425_42511

/-- Represents the position of the cat or mouse -/
inductive Position
| TopLeft
| TopMiddle
| TopRight
| RightMiddle
| BottomRight
| BottomMiddle
| BottomLeft
| LeftMiddle

/-- The number of moves in the problem -/
def totalMoves : ℕ := 315

/-- The length of the cat's movement cycle -/
def catCycleLength : ℕ := 4

/-- The length of the mouse's movement cycle -/
def mouseCycleLength : ℕ := 8

/-- Function to determine the cat's position after a given number of moves -/
def catPosition (moves : ℕ) : Position :=
  match moves % catCycleLength with
  | 0 => Position.TopLeft
  | 1 => Position.TopRight
  | 2 => Position.BottomRight
  | 3 => Position.BottomLeft
  | _ => Position.TopLeft  -- This case should never occur due to the modulo operation

/-- Function to determine the mouse's position after a given number of moves -/
def mousePosition (moves : ℕ) : Position :=
  match moves % mouseCycleLength with
  | 0 => Position.TopMiddle
  | 1 => Position.TopRight
  | 2 => Position.RightMiddle
  | 3 => Position.BottomRight
  | 4 => Position.BottomMiddle
  | 5 => Position.BottomLeft
  | 6 => Position.LeftMiddle
  | 7 => Position.TopLeft
  | _ => Position.TopMiddle  -- This case should never occur due to the modulo operation

theorem cat_and_mouse_positions : 
  catPosition totalMoves = Position.BottomRight ∧ 
  mousePosition totalMoves = Position.RightMiddle := by
  sorry

end NUMINAMATH_CALUDE_cat_and_mouse_positions_l425_42511


namespace NUMINAMATH_CALUDE_intersection_point_k_value_l425_42570

theorem intersection_point_k_value (k : ℝ) : 
  (∃ x y : ℝ, x - 2*y - 2*k = 0 ∧ 2*x - 3*y - k = 0 ∧ 3*x - y = 0) → k = 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_k_value_l425_42570


namespace NUMINAMATH_CALUDE_equation_represents_pair_of_straight_lines_l425_42541

/-- The equation representing the graph -/
def equation (x y : ℝ) : Prop := 9 * x^2 - y^2 - 6 * x = 0

/-- Definition of a straight line in slope-intercept form -/
def is_straight_line (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

/-- The theorem stating that the equation represents a pair of straight lines -/
theorem equation_represents_pair_of_straight_lines :
  ∃ f g : ℝ → ℝ, 
    (is_straight_line f ∧ is_straight_line g) ∧
    (∀ x y : ℝ, equation x y ↔ (y = f x ∨ y = g x)) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_pair_of_straight_lines_l425_42541


namespace NUMINAMATH_CALUDE_toms_weekly_fluid_intake_l425_42572

/-- The number of cans of soda Tom drinks per day -/
def soda_cans_per_day : ℕ := 5

/-- The number of ounces in each can of soda -/
def oz_per_soda_can : ℕ := 12

/-- The number of ounces of water Tom drinks per day -/
def water_oz_per_day : ℕ := 64

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Tom's weekly fluid intake in ounces -/
def weekly_fluid_intake : ℕ := 
  (soda_cans_per_day * oz_per_soda_can + water_oz_per_day) * days_in_week

theorem toms_weekly_fluid_intake : weekly_fluid_intake = 868 := by
  sorry

end NUMINAMATH_CALUDE_toms_weekly_fluid_intake_l425_42572


namespace NUMINAMATH_CALUDE_sum_of_pairwise_products_of_cubic_roots_l425_42592

theorem sum_of_pairwise_products_of_cubic_roots (p q r : ℝ) : 
  (6 * p^3 - 9 * p^2 + 17 * p - 12 = 0) →
  (6 * q^3 - 9 * q^2 + 17 * q - 12 = 0) →
  (6 * r^3 - 9 * r^2 + 17 * r - 12 = 0) →
  p * q + q * r + r * p = 17 / 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_pairwise_products_of_cubic_roots_l425_42592


namespace NUMINAMATH_CALUDE_intersection_points_concyclic_and_share_radical_axis_l425_42554

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Given two circles and two lines, returns the new circle formed by the intersection of chords -/
def newCircle (C₁ C₂ : Circle) (L₁ L₂ : Line) : Circle :=
  sorry

/-- Checks if three circles share a common radical axis -/
def shareCommonRadicalAxis (C₁ C₂ C₃ : Circle) : Prop :=
  sorry

/-- Main theorem: The four intersection points of chords lie on a new circle that shares
    a common radical axis with the original two circles -/
theorem intersection_points_concyclic_and_share_radical_axis
  (C₁ C₂ : Circle) (L₁ L₂ : Line) :
  let C := newCircle C₁ C₂ L₁ L₂
  shareCommonRadicalAxis C₁ C₂ C :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_points_concyclic_and_share_radical_axis_l425_42554


namespace NUMINAMATH_CALUDE_parallel_line_slope_l425_42585

/-- Given a line parallel to 3x - 6y = 21, its slope is 1/2 -/
theorem parallel_line_slope (a b c : ℝ) (h : a * x - b * y = c) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ (a, b, c) = (3 * k, 6 * k, 21 * k)) :
  (a / b : ℝ) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l425_42585


namespace NUMINAMATH_CALUDE_clock_angle_at_four_l425_42514

/-- The number of degrees in a complete circle -/
def circle_degrees : ℕ := 360

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees between each hour mark on a clock -/
def degrees_per_hour : ℕ := circle_degrees / clock_hours

/-- The hour we're considering -/
def target_hour : ℕ := 4

/-- The smaller angle formed by the clock hands at the target hour -/
def smaller_angle (h : ℕ) : ℕ := min (h * degrees_per_hour) (circle_degrees - h * degrees_per_hour)

theorem clock_angle_at_four :
  smaller_angle target_hour = 120 := by sorry

end NUMINAMATH_CALUDE_clock_angle_at_four_l425_42514


namespace NUMINAMATH_CALUDE_go_board_sales_solution_l425_42522

/-- Represents the sales data for a month -/
structure MonthlySales where
  typeA : ℕ
  typeB : ℕ
  revenue : ℕ

/-- Represents the Go board sales problem -/
structure GoBoardSales where
  purchasePriceA : ℕ
  purchasePriceB : ℕ
  month1 : MonthlySales
  month2 : MonthlySales
  totalBudget : ℕ
  totalSets : ℕ

/-- Theorem stating the solution to the Go board sales problem -/
theorem go_board_sales_solution (sales : GoBoardSales)
  (h1 : sales.purchasePriceA = 200)
  (h2 : sales.purchasePriceB = 170)
  (h3 : sales.month1 = ⟨3, 5, 1800⟩)
  (h4 : sales.month2 = ⟨4, 10, 3100⟩)
  (h5 : sales.totalBudget = 5400)
  (h6 : sales.totalSets = 30) :
  ∃ (sellingPriceA sellingPriceB maxTypeA : ℕ),
    sellingPriceA = 250 ∧
    sellingPriceB = 210 ∧
    maxTypeA = 10 ∧
    maxTypeA * sales.purchasePriceA + (sales.totalSets - maxTypeA) * sales.purchasePriceB ≤ sales.totalBudget ∧
    maxTypeA * (sellingPriceA - sales.purchasePriceA) + (sales.totalSets - maxTypeA) * (sellingPriceB - sales.purchasePriceB) = 1300 :=
by sorry


end NUMINAMATH_CALUDE_go_board_sales_solution_l425_42522


namespace NUMINAMATH_CALUDE_seashells_total_l425_42565

theorem seashells_total (sally_shells tom_shells jessica_shells : ℕ) 
  (h1 : sally_shells = 9)
  (h2 : tom_shells = 7)
  (h3 : jessica_shells = 5) :
  sally_shells + tom_shells + jessica_shells = 21 := by
  sorry

end NUMINAMATH_CALUDE_seashells_total_l425_42565


namespace NUMINAMATH_CALUDE_grid_toothpick_count_l425_42537

/-- Calculates the number of toothpicks in a rectangular grid with a missing row and column -/
def toothpick_count (height : ℕ) (width : ℕ) : ℕ :=
  let horizontal_lines := height
  let vertical_lines := width
  let horizontal_toothpicks := horizontal_lines * width
  let vertical_toothpicks := vertical_lines * (height - 1)
  horizontal_toothpicks + vertical_toothpicks

/-- Theorem stating that a 25x15 grid with a missing row and column uses 735 toothpicks -/
theorem grid_toothpick_count : toothpick_count 25 15 = 735 := by
  sorry

#eval toothpick_count 25 15

end NUMINAMATH_CALUDE_grid_toothpick_count_l425_42537


namespace NUMINAMATH_CALUDE_sum_of_three_squares_l425_42542

theorem sum_of_three_squares (square triangle : ℚ) : 
  (square + triangle + 2 * square + triangle = 34) →
  (triangle + square + triangle + 3 * square = 40) →
  (3 * square = 66 / 7) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_l425_42542


namespace NUMINAMATH_CALUDE_orange_price_is_60_cents_l425_42562

/-- Represents the price and quantity of fruits -/
structure FruitInfo where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  final_avg_price : ℚ
  removed_oranges : ℕ

/-- Theorem stating that given the conditions, the price of each orange is 60 cents -/
theorem orange_price_is_60_cents (info : FruitInfo) 
    (h1 : info.apple_price = 40/100)
    (h2 : info.total_fruits = 10)
    (h3 : info.initial_avg_price = 54/100)
    (h4 : info.final_avg_price = 48/100)
    (h5 : info.removed_oranges = 5) :
    info.orange_price = 60/100 := by
  sorry

#check orange_price_is_60_cents

end NUMINAMATH_CALUDE_orange_price_is_60_cents_l425_42562


namespace NUMINAMATH_CALUDE_ray_gave_ratio_l425_42527

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The initial amount Ray has in cents -/
def initial_amount : ℕ := 95

/-- The amount Ray gives to Peter in cents -/
def amount_to_peter : ℕ := 25

/-- The number of nickels Ray has left after giving to both Peter and Randi -/
def nickels_left : ℕ := 4

/-- The ratio of the amount Ray gave to Randi to the amount he gave to Peter -/
def ratio_randi_to_peter : ℚ := 2 / 1

theorem ray_gave_ratio :
  let initial_nickels := initial_amount / nickel_value
  let nickels_to_peter := amount_to_peter / nickel_value
  let nickels_to_randi := initial_nickels - nickels_to_peter - nickels_left
  let amount_to_randi := nickels_to_randi * nickel_value
  (amount_to_randi : ℚ) / amount_to_peter = ratio_randi_to_peter :=
by sorry

end NUMINAMATH_CALUDE_ray_gave_ratio_l425_42527


namespace NUMINAMATH_CALUDE_k_range_l425_42574

theorem k_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 2 * k * x - (k + 2) < 0) → 
  -1 < k ∧ k < 0 := by
sorry

end NUMINAMATH_CALUDE_k_range_l425_42574


namespace NUMINAMATH_CALUDE_no_half_parallel_diagonals_l425_42532

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- n ≥ 3 for a valid polygon
  sides_ge_three : n ≥ 3

/-- The number of diagonals in a regular polygon -/
def num_diagonals (p : RegularPolygon n) : ℕ :=
  n * (n - 3) / 2

/-- The number of diagonals parallel to sides in a regular polygon -/
def num_parallel_diagonals (p : RegularPolygon n) : ℕ :=
  if n % 2 = 0 then
    (n / 2 - 1)
  else
    0

/-- Theorem: No regular polygon has exactly half of its diagonals parallel to its sides -/
theorem no_half_parallel_diagonals (n : ℕ) (p : RegularPolygon n) :
  2 * (num_parallel_diagonals p) ≠ num_diagonals p :=
sorry

end NUMINAMATH_CALUDE_no_half_parallel_diagonals_l425_42532


namespace NUMINAMATH_CALUDE_square_neg_sqrt_three_eq_three_l425_42536

theorem square_neg_sqrt_three_eq_three : (-Real.sqrt 3)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_neg_sqrt_three_eq_three_l425_42536


namespace NUMINAMATH_CALUDE_mans_swimming_speed_l425_42533

/-- The speed of a man swimming in still water, given his downstream and upstream performances. -/
theorem mans_swimming_speed (downstream_distance upstream_distance : ℝ) 
  (time : ℝ) (h1 : downstream_distance = 42) (h2 : upstream_distance = 18) (h3 : time = 3) : 
  ∃ (v_m v_s : ℝ), v_m = 10 ∧ 
    downstream_distance = (v_m + v_s) * time ∧ 
    upstream_distance = (v_m - v_s) * time :=
by
  sorry

#check mans_swimming_speed

end NUMINAMATH_CALUDE_mans_swimming_speed_l425_42533


namespace NUMINAMATH_CALUDE_garden_flowers_equality_l425_42534

/-- Given a garden with white and red flowers, calculate the number of additional red flowers needed to make their quantities equal. -/
def additional_red_flowers (white : ℕ) (red : ℕ) : ℕ :=
  if white > red then white - red else 0

/-- Theorem: In a garden with 555 white flowers and 347 red flowers, 208 additional red flowers are needed to make their quantities equal. -/
theorem garden_flowers_equality : additional_red_flowers 555 347 = 208 := by
  sorry

end NUMINAMATH_CALUDE_garden_flowers_equality_l425_42534


namespace NUMINAMATH_CALUDE_phil_quarters_l425_42590

def initial_amount : ℚ := 40
def pizza_cost : ℚ := 2.75
def soda_cost : ℚ := 1.5
def jeans_cost : ℚ := 11.5

def remaining_amount : ℚ := initial_amount - (pizza_cost + soda_cost + jeans_cost)

def quarters_in_dollar : ℕ := 4

theorem phil_quarters : 
  ⌊remaining_amount * quarters_in_dollar⌋ = 97 := by
  sorry

end NUMINAMATH_CALUDE_phil_quarters_l425_42590


namespace NUMINAMATH_CALUDE_quadratic_sum_l425_42596

/-- A quadratic function f(x) = ax^2 + bx + c with vertex (3, -2) and f(0) = 0 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c x = a * (x - 3)^2 - 2) →  -- vertex form
  QuadraticFunction a b c 0 = 0 →                         -- passes through (0, 0)
  a + b + c = -10/9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l425_42596


namespace NUMINAMATH_CALUDE_fourth_sample_is_31_l425_42517

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : Nat
  sample_size : Nat
  known_samples : Finset Nat

/-- Calculates the sample interval for a systematic sampling. -/
def sample_interval (s : SystematicSampling) : Nat :=
  s.total_students / s.sample_size

/-- Theorem: In a systematic sampling of 4 from 56 students, if 3, 17, and 45 are sampled, the fourth sample is 31. -/
theorem fourth_sample_is_31 (s : SystematicSampling) 
  (h1 : s.total_students = 56)
  (h2 : s.sample_size = 4)
  (h3 : s.known_samples = {3, 17, 45}) :
  ∃ (fourth_sample : Nat), fourth_sample ∈ s.known_samples ∪ {31} ∧ 
  (s.known_samples ∪ {fourth_sample}).card = s.sample_size :=
by
  sorry


end NUMINAMATH_CALUDE_fourth_sample_is_31_l425_42517


namespace NUMINAMATH_CALUDE_side_c_values_simplify_expression_l425_42579

-- Define a triangle with side lengths a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the specific triangle with given conditions
def SpecificTriangle : Triangle → Prop
  | t => t.a = 4 ∧ t.b = 6 ∧ t.a + t.b + t.c < 18 ∧ Even (t.a + t.b + t.c)

-- Theorem 1: If the perimeter is less than 18 and even, then c = 4 or c = 6
theorem side_c_values (t : Triangle) (h : SpecificTriangle t) :
  t.c = 4 ∨ t.c = 6 := by
  sorry

-- Theorem 2: Simplification of |a+b-c|+|c-a-b|
theorem simplify_expression (t : Triangle) :
  |t.a + t.b - t.c| + |t.c - t.a - t.b| = 2*t.a + 2*t.b - 2*t.c := by
  sorry

end NUMINAMATH_CALUDE_side_c_values_simplify_expression_l425_42579
