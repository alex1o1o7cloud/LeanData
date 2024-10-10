import Mathlib

namespace money_saved_monthly_payment_l1352_135241

/-- Calculates the money saved by paying monthly instead of weekly for a hotel stay. -/
theorem money_saved_monthly_payment (weekly_rate : ℕ) (monthly_rate : ℕ) (num_months : ℕ) 
  (h1 : weekly_rate = 280)
  (h2 : monthly_rate = 1000)
  (h3 : num_months = 3) :
  weekly_rate * 4 * num_months - monthly_rate * num_months = 360 := by
  sorry

#check money_saved_monthly_payment

end money_saved_monthly_payment_l1352_135241


namespace complex_number_quadrant_l1352_135257

theorem complex_number_quadrant : 
  let z : ℂ := (1 + 2*I) / (1 - I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end complex_number_quadrant_l1352_135257


namespace expand_product_l1352_135295

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12*x + 27 := by
  sorry

end expand_product_l1352_135295


namespace inverse_of_i_power_2023_l1352_135253

theorem inverse_of_i_power_2023 : ∃ z : ℂ, z = (Complex.I : ℂ) ^ 2023 ∧ z⁻¹ = Complex.I := by
  sorry

end inverse_of_i_power_2023_l1352_135253


namespace solve_for_x_l1352_135269

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := by
  sorry

end solve_for_x_l1352_135269


namespace total_pets_is_54_l1352_135259

/-- The number of pets owned by Teddy, Ben, and Dave -/
def total_pets : ℕ :=
  let teddy_dogs : ℕ := 7
  let teddy_cats : ℕ := 8
  let ben_extra_dogs : ℕ := 9
  let dave_extra_cats : ℕ := 13
  let dave_fewer_dogs : ℕ := 5

  let teddy_pets : ℕ := teddy_dogs + teddy_cats
  let ben_pets : ℕ := (teddy_dogs + ben_extra_dogs)
  let dave_pets : ℕ := (teddy_cats + dave_extra_cats) + (teddy_dogs - dave_fewer_dogs)

  teddy_pets + ben_pets + dave_pets

theorem total_pets_is_54 : total_pets = 54 := by
  sorry

end total_pets_is_54_l1352_135259


namespace x_squared_coefficient_l1352_135289

def expansion (x : ℝ) := (2*x + 1) * (x - 2)^3

theorem x_squared_coefficient : 
  (∃ a b c d : ℝ, expansion x = a*x^3 + b*x^2 + c*x + d) → 
  (∃ a c d : ℝ, expansion x = a*x^3 + 18*x^2 + c*x + d) :=
by sorry

end x_squared_coefficient_l1352_135289


namespace digit_sum_problem_l1352_135216

/-- Given that P, Q, and R are single digits and PQR + QR = 1012, prove that P + Q + R = 20 -/
theorem digit_sum_problem (P Q R : ℕ) : 
  P < 10 → Q < 10 → R < 10 → 
  100 * P + 10 * Q + R + 10 * Q + R = 1012 →
  P + Q + R = 20 := by
sorry

end digit_sum_problem_l1352_135216


namespace not_square_of_two_pow_minus_one_l1352_135207

theorem not_square_of_two_pow_minus_one (n : ℕ) (h : n > 1) :
  ¬ ∃ k : ℕ, 2^n - 1 = k^2 := by
  sorry

end not_square_of_two_pow_minus_one_l1352_135207


namespace factory_sampling_probability_l1352_135220

/-- Represents a district with a number of factories -/
structure District where
  name : String
  factories : ℕ

/-- Represents the sampling result -/
structure SamplingResult where
  districtA : ℕ
  districtB : ℕ
  districtC : ℕ

/-- The stratified sampling function -/
def stratifiedSampling (districts : List District) (totalSample : ℕ) : SamplingResult :=
  sorry

/-- The probability calculation function -/
def probabilityAtLeastOneFromC (sample : SamplingResult) : ℚ :=
  sorry

theorem factory_sampling_probability :
  let districts := [
    { name := "A", factories := 9 },
    { name := "B", factories := 18 },
    { name := "C", factories := 18 }
  ]
  let sample := stratifiedSampling districts 5
  sample.districtA = 1 ∧
  sample.districtB = 2 ∧
  sample.districtC = 2 ∧
  probabilityAtLeastOneFromC sample = 7/10 :=
by sorry

end factory_sampling_probability_l1352_135220


namespace parabola_shift_theorem_l1352_135225

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift_theorem :
  let original := Parabola.mk 1 0 0  -- y = x^2
  let shifted := shift_parabola original 1 2
  shifted = Parabola.mk 1 (-2) 3  -- y = (x-1)^2 + 2
  := by sorry

end parabola_shift_theorem_l1352_135225


namespace coefficient_expansion_l1352_135292

theorem coefficient_expansion (a : ℝ) : 
  (∃ c : ℝ, c = 9 ∧ c = 1 + 4 * a) → a = 2 := by sorry

end coefficient_expansion_l1352_135292


namespace total_books_is_14_l1352_135288

/-- The number of books a librarian takes away. -/
def librarian_books : ℕ := 2

/-- The number of books that can fit on a shelf. -/
def books_per_shelf : ℕ := 3

/-- The number of shelves Roger needs. -/
def shelves_needed : ℕ := 4

/-- The total number of books to put away. -/
def total_books : ℕ := librarian_books + books_per_shelf * shelves_needed

theorem total_books_is_14 : total_books = 14 := by
  sorry

end total_books_is_14_l1352_135288


namespace sqrt_22_greater_than_4_l1352_135231

theorem sqrt_22_greater_than_4 : Real.sqrt 22 > 4 := by
  sorry

end sqrt_22_greater_than_4_l1352_135231


namespace wilsons_theorem_l1352_135266

theorem wilsons_theorem (p : ℕ) (hp : p > 1) :
  (p.factorial - 1) % p = 0 ↔ Nat.Prime p := by sorry

end wilsons_theorem_l1352_135266


namespace circle_center_problem_l1352_135232

/-- A circle tangent to two parallel lines with its center on a third line --/
theorem circle_center_problem (x y : ℝ) :
  (6 * x - 5 * y = 15) ∧ 
  (3 * x + 2 * y = 0) →
  x = 10 / 3 ∧ y = -5 := by
  sorry

#check circle_center_problem

end circle_center_problem_l1352_135232


namespace derivative_symmetric_points_l1352_135234

/-- Given a function f(x) = ax^4 + bx^2 + c where f'(1) = 2, prove that f'(-1) = -2 -/
theorem derivative_symmetric_points 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^4 + b * x^2 + c)
  (h2 : deriv f 1 = 2) :
  deriv f (-1) = -2 := by
  sorry

end derivative_symmetric_points_l1352_135234


namespace smallest_n_with_three_pairs_l1352_135236

/-- The function g(n) returns the number of distinct ordered pairs of positive integers (a, b) such that a^2 + b^2 + ab = n -/
def g (n : ℕ) : ℕ := (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2^2 + p.1 * p.2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 48 is the smallest positive integer n for which g(n) = 3 -/
theorem smallest_n_with_three_pairs : (∀ m : ℕ, m > 0 ∧ m < 48 → g m ≠ 3) ∧ g 48 = 3 := by
  sorry

end smallest_n_with_three_pairs_l1352_135236


namespace point_on_linear_graph_l1352_135298

/-- Given that the point (a, -1) lies on the graph of y = -2x + 1, prove that a = 1 -/
theorem point_on_linear_graph (a : ℝ) : 
  -1 = -2 * a + 1 → a = 1 := by
  sorry

end point_on_linear_graph_l1352_135298


namespace equation_solution_l1352_135291

theorem equation_solution :
  ∀ y : ℝ, (3 + 1.5 * y^2 = 0.5 * y^2 + 16) ↔ (y = Real.sqrt 13 ∨ y = -Real.sqrt 13) :=
by sorry

end equation_solution_l1352_135291


namespace worker_a_time_l1352_135208

/-- Proves that Worker A takes 10 hours to do a job alone, given the conditions of the problem -/
theorem worker_a_time (time_b time_together : ℝ) : 
  time_b = 15 → 
  time_together = 6 → 
  (1 / 10 : ℝ) + (1 / time_b) = (1 / time_together) := by
  sorry

end worker_a_time_l1352_135208


namespace total_bread_slices_l1352_135246

/-- The number of sandwiches Ryan wants to make -/
def num_sandwiches : ℕ := 5

/-- The number of bread slices needed for each sandwich -/
def slices_per_sandwich : ℕ := 3

/-- Theorem: The total number of bread slices needed for Ryan's sandwiches is 15 -/
theorem total_bread_slices : num_sandwiches * slices_per_sandwich = 15 := by
  sorry

end total_bread_slices_l1352_135246


namespace shortest_ribbon_length_l1352_135235

theorem shortest_ribbon_length (a b c d : ℕ) (ha : a = 2) (hb : b = 5) (hc : c = 7) (hd : d = 11) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 770 :=
by sorry

end shortest_ribbon_length_l1352_135235


namespace eggs_bought_l1352_135293

def initial_eggs : ℕ := 98
def final_eggs : ℕ := 106

theorem eggs_bought : final_eggs - initial_eggs = 8 := by
  sorry

end eggs_bought_l1352_135293


namespace divisor_ratio_of_M_l1352_135286

def M : ℕ := 36 * 45 * 98 * 160

/-- Sum of odd divisors of a natural number -/
def sum_odd_divisors (n : ℕ) : ℕ := sorry

/-- Sum of even divisors of a natural number -/
def sum_even_divisors (n : ℕ) : ℕ := sorry

/-- The ratio of sum of odd divisors to sum of even divisors -/
def divisor_ratio (n : ℕ) : ℚ :=
  (sum_odd_divisors n : ℚ) / (sum_even_divisors n : ℚ)

theorem divisor_ratio_of_M :
  divisor_ratio M = 1 / 510 := by sorry

end divisor_ratio_of_M_l1352_135286


namespace distance_to_center_l1352_135229

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 98}

-- Define the properties of the points
def PointProperties (A B C : ℝ × ℝ) : Prop :=
  A ∈ Circle ∧ B ∈ Circle ∧ C ∈ Circle ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64 ∧  -- AB = 8
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 9 ∧   -- BC = 3
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0  -- Angle ABC is right

-- The theorem
theorem distance_to_center (A B C : ℝ × ℝ) :
  PointProperties A B C → B.1^2 + B.2^2 = 50 := by sorry

end distance_to_center_l1352_135229


namespace two_digit_number_puzzle_l1352_135206

theorem two_digit_number_puzzle :
  ∀ x y : ℕ,
  x < 10 ∧ y < 10 ∧  -- Ensuring x and y are single digits
  x + y = 7 ∧  -- Sum of digits is 7
  (x + 2) + 10 * (y + 2) = 2 * (10 * y + x) - 3  -- Condition after adding 2 to each digit
  → 10 * y + x = 25 :=  -- The original number is 25
by
  sorry

end two_digit_number_puzzle_l1352_135206


namespace stating_fifteenth_term_is_43_l1352_135221

/-- 
Given an arithmetic sequence where:
- a₁ is the first term
- d is the common difference
- n is the term number
This function calculates the nth term of the sequence.
-/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- 
Theorem stating that the 15th term of the arithmetic sequence
with first term 1 and common difference 3 is 43.
-/
theorem fifteenth_term_is_43 : 
  arithmeticSequenceTerm 1 3 15 = 43 := by
sorry

end stating_fifteenth_term_is_43_l1352_135221


namespace final_price_after_discounts_l1352_135277

def original_price : ℝ := 15
def first_discount_rate : ℝ := 0.20
def second_discount_rate : ℝ := 0.25

theorem final_price_after_discounts :
  (original_price * (1 - first_discount_rate) * (1 - second_discount_rate)) = 9 := by
  sorry

end final_price_after_discounts_l1352_135277


namespace two_power_minus_three_power_eq_one_solutions_l1352_135278

theorem two_power_minus_three_power_eq_one_solutions :
  ∀ m n : ℕ, 2^m - 3^n = 1 ↔ (m = 1 ∧ n = 0) ∨ (m = 2 ∧ n = 1) := by
  sorry

end two_power_minus_three_power_eq_one_solutions_l1352_135278


namespace race_speed_factor_l1352_135201

/-- Represents the race scenario described in the problem -/
structure RaceScenario where
  k : ℝ  -- Factor by which A is faster than B
  startAdvantage : ℝ  -- Head start given to B in meters
  totalDistance : ℝ  -- Total race distance in meters

/-- Theorem stating that under the given conditions, A must be 4 times faster than B -/
theorem race_speed_factor (race : RaceScenario) 
  (h1 : race.startAdvantage = 72)
  (h2 : race.totalDistance = 96)
  (h3 : race.totalDistance / race.k = (race.totalDistance - race.startAdvantage)) :
  race.k = 4 := by
  sorry


end race_speed_factor_l1352_135201


namespace tension_in_rope_l1352_135287

/-- A system of pulleys and masses as described in the problem -/
structure PulleySystem (m : ℝ) where
  /-- The acceleration due to gravity -/
  g : ℝ
  /-- The tension in the rope connecting the bodies m and 2m through the upper pulley -/
  tension : ℝ

/-- The theorem stating the tension in the rope connecting the bodies m and 2m -/
theorem tension_in_rope (m : ℝ) (sys : PulleySystem m) (hm : m > 0) :
  sys.tension = (10 / 3) * m * sys.g := by
  sorry


end tension_in_rope_l1352_135287


namespace no_base_for_square_l1352_135222

theorem no_base_for_square (b : ℤ) : b > 4 → ¬∃ (k : ℤ), b^2 + 4*b + 3 = k^2 := by
  sorry

end no_base_for_square_l1352_135222


namespace rectangle_triangle_area_ratio_l1352_135290

theorem rectangle_triangle_area_ratio :
  ∀ (L W : ℝ), L > 0 → W > 0 →
  (L * W) / ((1/2) * L * W) = 2 :=
by
  sorry

end rectangle_triangle_area_ratio_l1352_135290


namespace kyler_won_one_game_l1352_135285

/-- Represents a chess tournament between Peter, Emma, and Kyler -/
structure ChessTournament where
  total_games : ℕ
  peter_wins : ℕ
  peter_losses : ℕ
  emma_wins : ℕ
  emma_losses : ℕ
  kyler_losses : ℕ

/-- Calculates Kyler's wins in the chess tournament -/
def kyler_wins (t : ChessTournament) : ℕ :=
  t.total_games - (t.peter_wins + t.peter_losses + t.emma_wins + t.emma_losses + t.kyler_losses)

/-- Theorem stating that Kyler won 1 game in the given tournament conditions -/
theorem kyler_won_one_game (t : ChessTournament) 
  (h1 : t.total_games = 15)
  (h2 : t.peter_wins = 5)
  (h3 : t.peter_losses = 3)
  (h4 : t.emma_wins = 2)
  (h5 : t.emma_losses = 4)
  (h6 : t.kyler_losses = 4) :
  kyler_wins t = 1 := by
  sorry

end kyler_won_one_game_l1352_135285


namespace sum_abcd_equals_negative_ten_thirds_l1352_135267

theorem sum_abcd_equals_negative_ten_thirds 
  (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 6) : 
  a + b + c + d = -10/3 := by
sorry

end sum_abcd_equals_negative_ten_thirds_l1352_135267


namespace length_BC_in_triangle_l1352_135299

/-- Parabola function -/
def parabola (x : ℝ) : ℝ := 2 * x^2

/-- Triangle ABC -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Theorem: Length of BC in triangle ABC -/
theorem length_BC_in_triangle (t : Triangle) : 
  (t.A.1 = 0 ∧ t.A.2 = 0) →  -- A is at origin
  (t.B.2 = parabola t.B.1) →  -- B is on parabola
  (t.C.2 = parabola t.C.1) →  -- C is on parabola
  (t.B.2 = t.C.2) →  -- BC is parallel to x-axis
  (1/2 * (t.C.1 - t.B.1) * t.B.2 = 128) →  -- Area of triangle is 128
  (t.C.1 - t.B.1 = 8) :=  -- Length of BC is 8
by sorry

end length_BC_in_triangle_l1352_135299


namespace function_decomposition_l1352_135281

-- Define a type for the domain that is symmetric with respect to the origin
structure SymmetricDomain where
  X : Type
  symm : X → X
  symm_involutive : ∀ x, symm (symm x) = x

-- Define a function on the symmetric domain
def Function (D : SymmetricDomain) := D.X → ℝ

-- Define an even function
def IsEven (D : SymmetricDomain) (f : Function D) : Prop :=
  ∀ x, f (D.symm x) = f x

-- Define an odd function
def IsOdd (D : SymmetricDomain) (f : Function D) : Prop :=
  ∀ x, f (D.symm x) = -f x

-- State the theorem
theorem function_decomposition (D : SymmetricDomain) (f : Function D) :
  ∃! (e o : Function D), (∀ x, f x = e x + o x) ∧ IsEven D e ∧ IsOdd D o := by
  sorry

end function_decomposition_l1352_135281


namespace equality_implies_equation_l1352_135264

theorem equality_implies_equation (x y : ℝ) (h : x = y) : -1/3 * x + 1 = -1/3 * y + 1 := by
  sorry

end equality_implies_equation_l1352_135264


namespace negative_a_squared_sum_l1352_135265

theorem negative_a_squared_sum (a : ℝ) : -3 * a^2 - 5 * a^2 = -8 * a^2 := by
  sorry

end negative_a_squared_sum_l1352_135265


namespace julia_remaining_money_l1352_135213

def initial_amount : ℚ := 40
def game_fraction : ℚ := 1/2
def in_game_purchase_fraction : ℚ := 1/4

theorem julia_remaining_money :
  let amount_after_game := initial_amount * (1 - game_fraction)
  let final_amount := amount_after_game * (1 - in_game_purchase_fraction)
  final_amount = 15 := by sorry

end julia_remaining_money_l1352_135213


namespace helen_cookies_proof_l1352_135279

/-- The number of cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 31

/-- The number of cookies Helen baked the day before yesterday -/
def cookies_day_before_yesterday : ℕ := 419

/-- The total number of cookies Helen baked until last night -/
def total_cookies : ℕ := cookies_yesterday + cookies_day_before_yesterday

theorem helen_cookies_proof : total_cookies = 450 := by
  sorry

end helen_cookies_proof_l1352_135279


namespace specific_pyramid_surface_area_l1352_135210

/-- Represents a pyramid with a parallelogram base -/
structure Pyramid where
  base_side1 : ℝ
  base_side2 : ℝ
  base_diagonal : ℝ
  height : ℝ

/-- Calculates the total surface area of the pyramid -/
def totalSurfaceArea (p : Pyramid) : ℝ :=
  sorry

/-- Theorem stating the total surface area of the specific pyramid -/
theorem specific_pyramid_surface_area :
  let p : Pyramid := { base_side1 := 10, base_side2 := 8, base_diagonal := 6, height := 4 }
  totalSurfaceArea p = 8 * (11 + Real.sqrt 34) := by
  sorry

end specific_pyramid_surface_area_l1352_135210


namespace rug_area_is_48_l1352_135203

/-- Calculates the area of a rug with specific dimensions -/
def rugArea (rect_length rect_width para_base para_height : ℝ) : ℝ :=
  let rect_area := rect_length * rect_width
  let para_area := para_base * para_height
  rect_area + 2 * para_area

/-- Theorem stating that a rug with given dimensions has an area of 48 square meters -/
theorem rug_area_is_48 :
  rugArea 6 4 3 4 = 48 := by
  sorry

end rug_area_is_48_l1352_135203


namespace lcm_gcf_ratio_l1352_135284

theorem lcm_gcf_ratio : (Nat.lcm 256 162) / (Nat.gcd 256 162) = 10368 := by
  sorry

end lcm_gcf_ratio_l1352_135284


namespace robin_ate_twelve_cupcakes_l1352_135237

/-- The number of cupcakes Robin ate with chocolate sauce -/
def chocolate_cupcakes : ℕ := 4

/-- The number of cupcakes Robin ate with buttercream frosting -/
def buttercream_cupcakes : ℕ := 2 * chocolate_cupcakes

/-- The total number of cupcakes Robin ate -/
def total_cupcakes : ℕ := chocolate_cupcakes + buttercream_cupcakes

theorem robin_ate_twelve_cupcakes : total_cupcakes = 12 := by
  sorry

end robin_ate_twelve_cupcakes_l1352_135237


namespace john_index_cards_l1352_135244

/-- Given that John buys 2 packs for each student, has 6 classes, and each class has 30 students,
    prove that the total number of packs John bought is 360. -/
theorem john_index_cards (packs_per_student : ℕ) (num_classes : ℕ) (students_per_class : ℕ)
  (h1 : packs_per_student = 2)
  (h2 : num_classes = 6)
  (h3 : students_per_class = 30) :
  packs_per_student * num_classes * students_per_class = 360 := by
  sorry

end john_index_cards_l1352_135244


namespace complex_division_simplification_l1352_135297

theorem complex_division_simplification :
  (3 + Complex.I) / (1 + Complex.I) = 2 - Complex.I :=
by sorry

end complex_division_simplification_l1352_135297


namespace henry_twice_jill_age_l1352_135202

/-- Represents the number of years ago when Henry was twice Jill's age -/
def years_ago (henry_age : ℕ) (jill_age : ℕ) : ℕ :=
  henry_age - jill_age

/-- Theorem stating that Henry was twice Jill's age 7 years ago -/
theorem henry_twice_jill_age (henry_age jill_age : ℕ) : 
  henry_age = 25 → 
  jill_age = 16 → 
  henry_age + jill_age = 41 → 
  years_ago henry_age jill_age = 7 := by
sorry

end henry_twice_jill_age_l1352_135202


namespace minimum_value_of_f_plus_f_prime_l1352_135242

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

-- Define the derivative of f
def f_prime (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

theorem minimum_value_of_f_plus_f_prime (a : ℝ) :
  (∃ x, f_prime a x = 0 ∧ x = 2) →
  (∀ m n, m ∈ Set.Icc (-1 : ℝ) 1 → n ∈ Set.Icc (-1 : ℝ) 1 →
    f a m + f_prime a n ≥ -13) ∧
  (∃ m n, m ∈ Set.Icc (-1 : ℝ) 1 ∧ n ∈ Set.Icc (-1 : ℝ) 1 ∧
    f a m + f_prime a n = -13) :=
by sorry

end minimum_value_of_f_plus_f_prime_l1352_135242


namespace most_accurate_reading_is_10_45_l1352_135247

/-- Represents a scientific weighing scale --/
structure ScientificScale where
  smallest_division : ℝ
  lower_bound : ℝ
  upper_bound : ℝ
  marker_position : ℝ

/-- Determines if a given reading is the most accurate for a scientific scale --/
def is_most_accurate_reading (s : ScientificScale) (reading : ℝ) : Prop :=
  s.lower_bound < reading ∧ 
  reading < s.upper_bound ∧ 
  reading % s.smallest_division = 0 ∧
  ∀ r, s.lower_bound < r ∧ r < s.upper_bound ∧ r % s.smallest_division = 0 → 
    |s.marker_position - reading| ≤ |s.marker_position - r|

/-- The theorem stating the most accurate reading for the given scale --/
theorem most_accurate_reading_is_10_45 (s : ScientificScale) 
  (h_division : s.smallest_division = 0.01)
  (h_lower : s.lower_bound = 10.41)
  (h_upper : s.upper_bound = 10.55)
  (h_marker : s.lower_bound < s.marker_position ∧ s.marker_position < (s.lower_bound + s.upper_bound) / 2) :
  is_most_accurate_reading s 10.45 :=
sorry

end most_accurate_reading_is_10_45_l1352_135247


namespace greatest_integer_absolute_value_l1352_135274

theorem greatest_integer_absolute_value (y : ℤ) : (∀ z : ℤ, |3*z - 4| ≤ 21 → z ≤ y) ↔ y = 8 := by sorry

end greatest_integer_absolute_value_l1352_135274


namespace smallest_square_containing_circle_l1352_135249

theorem smallest_square_containing_circle (r : ℝ) (h : r = 7) : 
  (2 * r) ^ 2 = 196 := by
  sorry

end smallest_square_containing_circle_l1352_135249


namespace irrational_difference_representation_l1352_135250

theorem irrational_difference_representation (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  ∃ (α β : ℝ), Irrational α ∧ Irrational β ∧ 0 < α ∧ α < 1 ∧ 0 < β ∧ β < 1 ∧ x = α - β := by
  sorry

end irrational_difference_representation_l1352_135250


namespace log_reciprocal_l1352_135200

theorem log_reciprocal (M : ℝ) (a : ℤ) (b : ℝ) 
  (h_pos : M > 0) 
  (h_log : Real.log M / Real.log 10 = a + b) 
  (h_b : 0 < b ∧ b < 1) : 
  Real.log (1 / M) / Real.log 10 = (-a - 1) + (1 - b) := by
  sorry

end log_reciprocal_l1352_135200


namespace line_through_point_with_equal_intercepts_l1352_135212

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) is on the line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if a line has equal intercepts on both axes -/
def Line.hasEqualIntercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c ≠ 0 ∧ l.c / l.a = - l.c / l.b

theorem line_through_point_with_equal_intercepts :
  ∃ (l : Line), l.contains (-1) 2 ∧ l.hasEqualIntercepts ∧
  ((l.a = 2 ∧ l.b = 1 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -1)) :=
sorry

end line_through_point_with_equal_intercepts_l1352_135212


namespace pole_wire_distance_l1352_135223

/-- Given a vertical pole and three equally spaced wires, calculate the distance between anchor points -/
theorem pole_wire_distance (pole_height : ℝ) (wire_length : ℝ) (anchor_distance : ℝ) : 
  pole_height = 70 →
  wire_length = 490 →
  (pole_height ^ 2 + (anchor_distance / (3 ^ (1/2))) ^ 2 = wire_length ^ 2) →
  anchor_distance = 840 := by
  sorry

end pole_wire_distance_l1352_135223


namespace smallest_area_right_triangle_l1352_135283

/-- The smallest possible area of a right triangle with two sides measuring 6 and 8 units is 24 square units. -/
theorem smallest_area_right_triangle : ℝ := by
  -- Let a and b be the two given sides of the right triangle
  let a : ℝ := 6
  let b : ℝ := 8
  
  -- Define the function to calculate the area of a right triangle
  let area (x y : ℝ) : ℝ := (1 / 2) * x * y
  
  -- State that the smallest area is 24
  let smallest_area : ℝ := 24
  
  sorry

end smallest_area_right_triangle_l1352_135283


namespace negation_of_forall_inequality_negation_of_inequality_negation_of_proposition_l1352_135255

theorem negation_of_forall_inequality (P : ℝ → Prop) :
  (¬ ∀ x < 0, P x) ↔ (∃ x < 0, ¬ P x) := by sorry

theorem negation_of_inequality (x : ℝ) :
  ¬(1 - x > Real.exp x) ↔ (1 - x ≤ Real.exp x) := by sorry

theorem negation_of_proposition :
  (¬ ∀ x < 0, 1 - x > Real.exp x) ↔ (∃ x < 0, 1 - x ≤ Real.exp x) := by sorry

end negation_of_forall_inequality_negation_of_inequality_negation_of_proposition_l1352_135255


namespace four_wheeler_wheels_l1352_135271

theorem four_wheeler_wheels (num_four_wheelers : ℕ) (wheels_per_four_wheeler : ℕ) : 
  num_four_wheelers = 17 → wheels_per_four_wheeler = 4 → num_four_wheelers * wheels_per_four_wheeler = 68 := by
  sorry

end four_wheeler_wheels_l1352_135271


namespace bus_passengers_l1352_135280

theorem bus_passengers (men women : ℕ) : 
  women = men / 3 →
  men - 24 = women + 12 →
  men + women = 72 :=
by sorry

end bus_passengers_l1352_135280


namespace fred_grew_38_cantaloupes_l1352_135248

/-- The number of cantaloupes Tim grew -/
def tims_cantaloupes : ℕ := 44

/-- The total number of cantaloupes Fred and Tim grew together -/
def total_cantaloupes : ℕ := 82

/-- The number of cantaloupes Fred grew -/
def freds_cantaloupes : ℕ := total_cantaloupes - tims_cantaloupes

theorem fred_grew_38_cantaloupes : freds_cantaloupes = 38 := by
  sorry

end fred_grew_38_cantaloupes_l1352_135248


namespace complex_number_quadrant_l1352_135243

theorem complex_number_quadrant (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end complex_number_quadrant_l1352_135243


namespace sum_of_integers_with_given_difference_and_product_l1352_135239

theorem sum_of_integers_with_given_difference_and_product :
  ∀ x y : ℕ+, 
    (x : ℝ) - (y : ℝ) = 10 →
    (x : ℝ) * (y : ℝ) = 56 →
    (x : ℝ) + (y : ℝ) = 18 := by
  sorry

end sum_of_integers_with_given_difference_and_product_l1352_135239


namespace second_order_de_solution_l1352_135270

/-- Given a second-order linear homogeneous differential equation with constant coefficients:
    y'' - 5y' - 6y = 0, prove that y = C₁e^(6x) + C₂e^(-x) is the general solution. -/
theorem second_order_de_solution (y : ℝ → ℝ) (C₁ C₂ : ℝ) :
  (∀ x, (deriv^[2] y) x - 5 * (deriv y) x - 6 * y x = 0) ↔
  (∃ C₁ C₂, ∀ x, y x = C₁ * Real.exp (6 * x) + C₂ * Real.exp (-x)) :=
sorry


end second_order_de_solution_l1352_135270


namespace distance_A_to_B_l1352_135296

def point_A : Fin 3 → ℝ := ![2, 3, 5]
def point_B : Fin 3 → ℝ := ![3, 1, 7]

theorem distance_A_to_B :
  Real.sqrt ((point_B 0 - point_A 0)^2 + (point_B 1 - point_A 1)^2 + (point_B 2 - point_A 2)^2) = 3 := by
  sorry

end distance_A_to_B_l1352_135296


namespace expected_heads_is_60_l1352_135238

/-- The number of coins -/
def num_coins : ℕ := 64

/-- The maximum number of flips per coin -/
def max_flips : ℕ := 4

/-- The probability of getting heads on a single flip -/
def p_heads : ℚ := 1/2

/-- The probability of getting heads after up to four flips -/
def p_heads_total : ℚ := 1 - (1 - p_heads)^max_flips

/-- The expected number of coins showing heads after up to four flips -/
def expected_heads : ℚ := num_coins * p_heads_total

theorem expected_heads_is_60 : expected_heads = 60 := by
  sorry

end expected_heads_is_60_l1352_135238


namespace hyperbola_equation_l1352_135275

/-- Given a hyperbola with the following properties:
    1) Standard form equation: x²/a² - y²/b² = 1
    2) a > 0 and b > 0
    3) Focal length is 2√5
    4) One asymptote is perpendicular to the line 2x + y = 0
    Prove that the equation of the hyperbola is x²/4 - y² = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_focal : (2 * Real.sqrt 5 : ℝ) = 2 * Real.sqrt (a^2 + b^2))
  (h_asymptote : b / a = 1 / 2) :
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 4 - y^2 = 1 := by
sorry

end hyperbola_equation_l1352_135275


namespace total_lockers_is_399_l1352_135273

/-- Represents the position of Minyoung's locker in the classroom -/
structure LockerPosition where
  front : ℕ
  back : ℕ
  left : ℕ
  right : ℕ

/-- Calculates the total number of lockers in the classroom based on Minyoung's locker position -/
def total_lockers (pos : LockerPosition) : ℕ :=
  (pos.front + pos.back - 1) * (pos.left + pos.right - 1)

/-- Theorem stating that the total number of lockers is 399 given Minyoung's locker position -/
theorem total_lockers_is_399 (pos : LockerPosition) 
  (h_front : pos.front = 8)
  (h_back : pos.back = 14)
  (h_left : pos.left = 7)
  (h_right : pos.right = 13) : 
  total_lockers pos = 399 := by
  sorry

#eval total_lockers ⟨8, 14, 7, 13⟩

end total_lockers_is_399_l1352_135273


namespace ABABCDCD_square_theorem_l1352_135219

/-- Represents an 8-digit number in the form ABABCDCD -/
def ABABCDCD (A B C D : Nat) : Nat :=
  A * 10000000 + B * 1000000 + A * 100000 + B * 10000 + C * 1000 + D * 100 + C * 10 + D

/-- Checks if four numbers are distinct digits -/
def areDistinctDigits (A B C D : Nat) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10

/-- The main theorem stating that only two sets of digits satisfy the conditions -/
theorem ABABCDCD_square_theorem :
  ∀ A B C D : Nat,
    (ABABCDCD A B C D)^2 = ABABCDCD A B C D ∧ areDistinctDigits A B C D →
    ((A = 9 ∧ B = 7 ∧ C = 0 ∧ D = 4) ∨ (A = 8 ∧ B = 0 ∧ C = 2 ∧ D = 1)) :=
by sorry

end ABABCDCD_square_theorem_l1352_135219


namespace right_triangle_existence_unique_non_right_triangle_l1352_135258

theorem right_triangle_existence (a b c : ℝ) : Bool :=
  a * a + b * b = c * c

theorem unique_non_right_triangle : 
  right_triangle_existence 3 4 5 = true ∧
  right_triangle_existence 1 1 (Real.sqrt 2) = true ∧
  right_triangle_existence 8 15 18 = false ∧
  right_triangle_existence 5 12 13 = true ∧
  right_triangle_existence 6 8 10 = true :=
by sorry

end right_triangle_existence_unique_non_right_triangle_l1352_135258


namespace arithmetic_sequence_sum_of_cubes_l1352_135218

/-- Given an arithmetic sequence with first term x and common difference 2,
    this function returns the sum of cubes of the first n+1 terms. -/
def sumOfCubes (x : ℤ) (n : ℕ) : ℤ :=
  (Finset.range (n+1)).sum (fun i => (x + 2 * i)^3)

/-- Theorem stating that for an arithmetic sequence with integer first term,
    if the sum of cubes of its terms is -6859 and the number of terms is greater than 6,
    then the number of terms is exactly 7 (i.e., n = 6). -/
theorem arithmetic_sequence_sum_of_cubes (x : ℤ) (n : ℕ) 
    (h1 : sumOfCubes x n = -6859)
    (h2 : n > 5) : n = 6 := by
  sorry

end arithmetic_sequence_sum_of_cubes_l1352_135218


namespace diophantine_equation_solution_l1352_135252

theorem diophantine_equation_solution :
  ∃ (m n : ℕ), 26019 * m - 649 * n = 118 ∧ m = 2 ∧ n = 80 := by
  sorry

end diophantine_equation_solution_l1352_135252


namespace point_in_fourth_quadrant_l1352_135230

def point : ℝ × ℝ := (8, -3)

theorem point_in_fourth_quadrant :
  let (x, y) := point
  x > 0 ∧ y < 0 :=
by sorry

end point_in_fourth_quadrant_l1352_135230


namespace wayne_shrimp_guests_l1352_135217

/-- Given Wayne's shrimp appetizer scenario, prove the number of guests he can serve. -/
theorem wayne_shrimp_guests :
  ∀ (shrimp_per_guest : ℕ) 
    (cost_per_pound : ℚ) 
    (shrimp_per_pound : ℕ) 
    (total_spent : ℚ),
  shrimp_per_guest = 5 →
  cost_per_pound = 17 →
  shrimp_per_pound = 20 →
  total_spent = 170 →
  (total_spent / cost_per_pound * shrimp_per_pound) / shrimp_per_guest = 40 :=
by
  sorry

end wayne_shrimp_guests_l1352_135217


namespace binary_11101_equals_29_l1352_135261

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11101_equals_29 :
  binary_to_decimal [true, false, true, true, true] = 29 := by
  sorry

end binary_11101_equals_29_l1352_135261


namespace handbag_discount_proof_l1352_135282

theorem handbag_discount_proof (initial_price : ℝ) (regular_discount : ℝ) (monday_discount : ℝ) :
  initial_price = 250 →
  regular_discount = 0.4 →
  monday_discount = 0.1 →
  let price_after_regular_discount := initial_price * (1 - regular_discount)
  let final_price := price_after_regular_discount * (1 - monday_discount)
  final_price = 135 :=
by sorry

end handbag_discount_proof_l1352_135282


namespace tom_theater_expenditure_l1352_135263

/-- Calculates Tom's expenditure for opening a theater --/
theorem tom_theater_expenditure
  (cost_per_sq_ft : ℝ)
  (space_per_seat : ℝ)
  (num_seats : ℕ)
  (construction_cost_multiplier : ℝ)
  (partner_contribution_percentage : ℝ)
  (h1 : cost_per_sq_ft = 5)
  (h2 : space_per_seat = 12)
  (h3 : num_seats = 500)
  (h4 : construction_cost_multiplier = 2)
  (h5 : partner_contribution_percentage = 0.4)
  : tom_expenditure = 54000 := by
  sorry

where
  tom_expenditure : ℝ :=
    let total_sq_ft := space_per_seat * num_seats
    let land_cost := cost_per_sq_ft * total_sq_ft
    let construction_cost := construction_cost_multiplier * land_cost
    let total_cost := land_cost + construction_cost
    (1 - partner_contribution_percentage) * total_cost

end tom_theater_expenditure_l1352_135263


namespace expression_value_l1352_135240

theorem expression_value (x y z : ℤ) (hx : x = 3) (hy : y = 2) (hz : z = -1) :
  3 * x^2 - 4 * y + 5 * z = 14 := by
  sorry

end expression_value_l1352_135240


namespace x_value_l1352_135228

theorem x_value (x : ℝ) (h1 : x > 0) (h2 : Real.sqrt ((8 * x) / 3) = x) : x = 8 / 3 := by
  sorry

end x_value_l1352_135228


namespace equal_age_regression_five_year_difference_regression_ten_percent_older_regression_l1352_135227

-- Define the type for a couple's ages
structure CoupleAges where
  bride_age : ℝ
  groom_age : ℝ

-- Define the dataset
def dataset : List CoupleAges := sorry

-- Define the regression line
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

-- Function to calculate regression line
def calculate_regression_line (data : List CoupleAges) : RegressionLine := sorry

-- Theorems to prove

theorem equal_age_regression :
  (∀ couple ∈ dataset, couple.bride_age = couple.groom_age) →
  let reg_line := calculate_regression_line dataset
  reg_line.slope = 1 ∧ reg_line.intercept = 0 := by sorry

theorem five_year_difference_regression :
  (∀ couple ∈ dataset, couple.groom_age = couple.bride_age + 5) →
  let reg_line := calculate_regression_line dataset
  reg_line.slope = 1 ∧ reg_line.intercept = 5 := by sorry

theorem ten_percent_older_regression :
  (∀ couple ∈ dataset, couple.groom_age = 1.1 * couple.bride_age) →
  let reg_line := calculate_regression_line dataset
  reg_line.slope = 1.1 ∧ reg_line.intercept = 0 := by sorry

end equal_age_regression_five_year_difference_regression_ten_percent_older_regression_l1352_135227


namespace concert_drive_distance_l1352_135224

/-- Calculates the remaining distance to drive given the total distance and the distance already driven. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Theorem stating that given a total distance of 78 miles and a distance already driven of 32 miles, 
    the remaining distance to drive is 46 miles. -/
theorem concert_drive_distance : remaining_distance 78 32 = 46 := by
  sorry

end concert_drive_distance_l1352_135224


namespace max_value_is_57_l1352_135233

/-- Represents a type of rock with its weight and value -/
structure Rock where
  weight : Nat
  value : Nat

/-- The problem setup -/
def rockTypes : List Rock := [
  { weight := 6, value := 18 },
  { weight := 3, value := 9 },
  { weight := 2, value := 3 }
]

/-- The maximum weight Carl can carry -/
def maxWeight : Nat := 20

/-- The minimum number of rocks available for each type -/
def minRocksPerType : Nat := 15

/-- A function to calculate the total value of a collection of rocks -/
def totalValue (rocks : List (Rock × Nat)) : Nat :=
  rocks.foldl (fun acc (rock, count) => acc + rock.value * count) 0

/-- A function to calculate the total weight of a collection of rocks -/
def totalWeight (rocks : List (Rock × Nat)) : Nat :=
  rocks.foldl (fun acc (rock, count) => acc + rock.weight * count) 0

/-- The main theorem stating that the maximum value Carl can carry is $57 -/
theorem max_value_is_57 :
  ∃ (rocks : List (Rock × Nat)),
    (∀ r ∈ rocks, r.1 ∈ rockTypes) ∧
    (∀ r ∈ rocks, r.2 ≤ minRocksPerType) ∧
    totalWeight rocks ≤ maxWeight ∧
    totalValue rocks = 57 ∧
    (∀ (other_rocks : List (Rock × Nat)),
      (∀ r ∈ other_rocks, r.1 ∈ rockTypes) →
      (∀ r ∈ other_rocks, r.2 ≤ minRocksPerType) →
      totalWeight other_rocks ≤ maxWeight →
      totalValue other_rocks ≤ 57) :=
by sorry


end max_value_is_57_l1352_135233


namespace parabola_focal_chord_inclination_l1352_135214

theorem parabola_focal_chord_inclination (x y : ℝ) (α : ℝ) : 
  y^2 = 6*x →  -- parabola equation
  12 = 6 / (Real.sin α)^2 →  -- focal chord length condition
  α = π/4 ∨ α = 3*π/4 :=  -- conclusion
by sorry

end parabola_focal_chord_inclination_l1352_135214


namespace cousins_distribution_l1352_135211

/-- The number of ways to distribute n indistinguishable objects into k distinct boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 5 cousins -/
def num_cousins : ℕ := 5

/-- There are 4 rooms -/
def num_rooms : ℕ := 4

/-- The number of ways to distribute the cousins into the rooms -/
def num_distributions : ℕ := distribute num_cousins num_rooms

theorem cousins_distribution :
  num_distributions = 66 := by sorry

end cousins_distribution_l1352_135211


namespace infinitely_many_special_numbers_l1352_135276

/-- The false derived function -/
noncomputable def false_derived (n : ℕ) : ℕ :=
  sorry

/-- The set of natural numbers n > 1 such that f(n) = f(n-1) + 1 -/
def special_set : Set ℕ :=
  {n : ℕ | n > 1 ∧ false_derived n = false_derived (n - 1) + 1}

/-- Theorem: There are infinitely many natural numbers n such that f(n) = f(n-1) + 1 -/
theorem infinitely_many_special_numbers : Set.Infinite special_set := by
  sorry

end infinitely_many_special_numbers_l1352_135276


namespace spade_problem_l1352_135254

/-- Custom operation ⊙ for real numbers -/
def spade (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

/-- Theorem stating that 2 ⊙ (3 ⊙ 4) = 384 -/
theorem spade_problem : spade 2 (spade 3 4) = 384 := by
  sorry

end spade_problem_l1352_135254


namespace count_hens_and_cows_l1352_135226

theorem count_hens_and_cows (total_animals : ℕ) (total_feet : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) : 
  total_animals = 44 → 
  total_feet = 128 → 
  hen_feet = 2 → 
  cow_feet = 4 → 
  ∃ (hens cows : ℕ), 
    hens + cows = total_animals ∧ 
    hen_feet * hens + cow_feet * cows = total_feet ∧ 
    hens = 24 := by
  sorry

end count_hens_and_cows_l1352_135226


namespace distance_A_B_min_value_expression_solutions_equation_max_product_mn_l1352_135272

-- Define the distance function on a number line
def distance (a b : ℝ) : ℝ := |a - b|

-- Statement 1
theorem distance_A_B : distance (-10) 8 = 18 := by sorry

-- Statement 2
theorem min_value_expression : 
  ∀ x : ℝ, |x - 3| + |x + 2| ≥ 5 := by sorry

-- Statement 3
theorem solutions_equation : 
  ∀ y : ℝ, |y - 3| + |y + 1| = 8 ↔ y = 5 ∨ y = -3 := by sorry

-- Statement 4
theorem max_product_mn : 
  ∀ m n : ℤ, (|m + 1| + |2 - m|) * (|n - 1| + |n + 3|) = 12 → 
  m * n ≤ 3 := by sorry

end distance_A_B_min_value_expression_solutions_equation_max_product_mn_l1352_135272


namespace find_n_l1352_135205

theorem find_n : ∃ n : ℤ, (5 : ℝ) ^ (2 * n) = (1 / 5 : ℝ) ^ (n - 12) ∧ n = 4 := by
  sorry

end find_n_l1352_135205


namespace find_heaviest_and_lightest_in_13_weighings_l1352_135209

/-- Represents a coin with a unique weight -/
structure Coin where
  weight : ℕ

/-- Represents the result of weighing two coins -/
inductive WeighResult
  | Left  : WeighResult  -- left coin is heavier
  | Right : WeighResult  -- right coin is heavier
  | Equal : WeighResult  -- coins have equal weight

/-- A function that simulates weighing two coins -/
def weigh (a b : Coin) : WeighResult :=
  if a.weight > b.weight then WeighResult.Left
  else if a.weight < b.weight then WeighResult.Right
  else WeighResult.Equal

/-- Theorem stating that it's possible to find the heaviest and lightest coins in 13 weighings -/
theorem find_heaviest_and_lightest_in_13_weighings
  (coins : List Coin)
  (h_distinct : ∀ i j, i ≠ j → (coins.get i).weight ≠ (coins.get j).weight)
  (h_count : coins.length = 10) :
  ∃ (heaviest lightest : Coin) (steps : List (Coin × Coin)),
    heaviest ∈ coins ∧
    lightest ∈ coins ∧
    (∀ c ∈ coins, c.weight ≤ heaviest.weight) ∧
    (∀ c ∈ coins, c.weight ≥ lightest.weight) ∧
    steps.length ≤ 13 ∧
    (∀ step ∈ steps, ∃ a b, step = (a, b) ∧ a ∈ coins ∧ b ∈ coins) :=
by sorry

end find_heaviest_and_lightest_in_13_weighings_l1352_135209


namespace four_roots_iff_a_in_range_l1352_135251

-- Define the function f(x) = |x^2 + 3x|
def f (x : ℝ) : ℝ := |x^2 + 3*x|

-- Define the equation f(x) - a|x-1| = 0
def equation (a : ℝ) (x : ℝ) : Prop := f x - a * |x - 1| = 0

-- Define the property of having exactly 4 distinct real roots
def has_four_distinct_roots (a : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    equation a x₁ ∧ equation a x₂ ∧ equation a x₃ ∧ equation a x₄ ∧
    ∀ (x : ℝ), equation a x → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄

-- Theorem statement
theorem four_roots_iff_a_in_range :
  ∀ a : ℝ, has_four_distinct_roots a ↔ (a ∈ Set.Ioo 0 1 ∪ Set.Ioi 9) :=
sorry

end four_roots_iff_a_in_range_l1352_135251


namespace employee_payment_l1352_135260

theorem employee_payment (total : ℝ) (x y : ℝ) (h1 : total = 528) (h2 : x = 1.2 * y) (h3 : total = x + y) : y = 240 := by
  sorry

end employee_payment_l1352_135260


namespace steve_pie_ratio_l1352_135215

/-- Steve's weekly pie baking schedule -/
structure PieSchedule where
  monday_apple : ℕ
  monday_blueberry : ℕ
  tuesday_cherry : ℕ
  tuesday_blueberry : ℕ
  wednesday_apple : ℕ
  wednesday_blueberry : ℕ
  thursday_cherry : ℕ
  thursday_blueberry : ℕ
  friday_apple : ℕ
  friday_blueberry : ℕ
  saturday_apple : ℕ
  saturday_cherry : ℕ
  saturday_blueberry : ℕ
  sunday_apple : ℕ
  sunday_cherry : ℕ
  sunday_blueberry : ℕ

/-- Calculate the total number of each type of pie baked in a week -/
def total_pies (schedule : PieSchedule) : ℕ × ℕ × ℕ :=
  let apple := schedule.monday_apple + schedule.wednesday_apple + schedule.friday_apple + 
                schedule.saturday_apple + schedule.sunday_apple
  let cherry := schedule.tuesday_cherry + schedule.thursday_cherry + 
                 schedule.saturday_cherry + schedule.sunday_cherry
  let blueberry := schedule.monday_blueberry + schedule.tuesday_blueberry + 
                   schedule.wednesday_blueberry + schedule.thursday_blueberry + 
                   schedule.friday_blueberry + schedule.saturday_blueberry + 
                   schedule.sunday_blueberry
  (apple, cherry, blueberry)

/-- Steve's actual weekly pie baking schedule -/
def steve_schedule : PieSchedule := {
  monday_apple := 16, monday_blueberry := 10,
  tuesday_cherry := 14, tuesday_blueberry := 8,
  wednesday_apple := 20, wednesday_blueberry := 12,
  thursday_cherry := 18, thursday_blueberry := 10,
  friday_apple := 16, friday_blueberry := 10,
  saturday_apple := 10, saturday_cherry := 8, saturday_blueberry := 6,
  sunday_apple := 6, sunday_cherry := 12, sunday_blueberry := 4
}

theorem steve_pie_ratio : 
  ∃ (k : ℕ), k > 0 ∧ total_pies steve_schedule = (17 * k, 13 * k, 15 * k) := by
  sorry

end steve_pie_ratio_l1352_135215


namespace min_area_theorem_l1352_135245

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with given side length -/
structure Square where
  side : ℝ

/-- Represents the configuration of shapes within the larger square -/
structure ShapeConfiguration where
  largeSquare : Square
  rectangle1 : Rectangle
  square1 : Square
  rectangleR : Rectangle

/-- The theorem statement -/
theorem min_area_theorem (config : ShapeConfiguration) : 
  config.rectangle1.width = 1 ∧ 
  config.rectangle1.height = 4 ∧
  config.square1.side = 1 ∧
  config.largeSquare.side ≥ 4 →
  config.largeSquare.side ^ 2 ≥ 16 ∧
  config.rectangleR.width * config.rectangleR.height = 11 := by
  sorry

end min_area_theorem_l1352_135245


namespace lost_card_number_l1352_135262

theorem lost_card_number (n : ℕ) (h1 : n > 0) (h2 : (n * (n + 1)) / 2 - 101 ∈ Finset.range (n + 1)) : 
  (n * (n + 1)) / 2 - 101 = 4 := by
sorry

end lost_card_number_l1352_135262


namespace savings_account_interest_rate_l1352_135294

theorem savings_account_interest_rate (initial_deposit : ℝ) (balance_after_first_year : ℝ) (total_increase_percentage : ℝ) : 
  initial_deposit = 5000 →
  balance_after_first_year = 5500 →
  total_increase_percentage = 21 →
  let total_balance := initial_deposit * (1 + total_increase_percentage / 100)
  let increase_second_year := total_balance - balance_after_first_year
  let percentage_increase_second_year := (increase_second_year / balance_after_first_year) * 100
  percentage_increase_second_year = 10 := by
sorry

end savings_account_interest_rate_l1352_135294


namespace intersected_cubes_in_4x4x4_cube_l1352_135268

/-- Represents a cube composed of unit cubes -/
structure UnitCube where
  side : ℕ

/-- Represents a plane in 3D space -/
structure Plane

/-- Predicate to check if a plane is perpendicular to and bisects an internal diagonal of a cube -/
def is_perpendicular_bisector (c : UnitCube) (p : Plane) : Prop :=
  sorry

/-- Counts the number of unit cubes intersected by a plane in a given cube -/
def intersected_cubes (c : UnitCube) (p : Plane) : ℕ :=
  sorry

/-- Theorem stating that a plane perpendicular to and bisecting an internal diagonal
    of a 4x4x4 cube intersects exactly 40 unit cubes -/
theorem intersected_cubes_in_4x4x4_cube (c : UnitCube) (p : Plane) :
  c.side = 4 → is_perpendicular_bisector c p → intersected_cubes c p = 40 :=
by sorry

end intersected_cubes_in_4x4x4_cube_l1352_135268


namespace equidistant_implies_d_squared_l1352_135256

/-- A complex function g that scales by a complex number c+di -/
def g (c d : ℝ) (z : ℂ) : ℂ := (c + d * Complex.I) * z

/-- The property that g(z) is equidistant from z and the origin for all z -/
def equidistant (c d : ℝ) : Prop :=
  ∀ z : ℂ, Complex.abs (g c d z - z) = Complex.abs (g c d z)

theorem equidistant_implies_d_squared (c d : ℝ) 
  (h1 : equidistant c d) 
  (h2 : Complex.abs (c + d * Complex.I) = 5) : 
  d^2 = 99/4 := by sorry

end equidistant_implies_d_squared_l1352_135256


namespace greatest_k_value_l1352_135204

theorem greatest_k_value (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + k*x₁ + 8 = 0 ∧ 
    x₂^2 + k*x₂ + 8 = 0 ∧ 
    |x₁ - x₂| = Real.sqrt 89) →
  k ≤ 11 :=
by sorry

end greatest_k_value_l1352_135204
