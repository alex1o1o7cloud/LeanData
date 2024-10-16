import Mathlib

namespace NUMINAMATH_CALUDE_min_shift_value_l2122_212288

open Real

noncomputable def f (x : ℝ) : ℝ := sin x * cos x - Real.sqrt 3 * cos x ^ 2

noncomputable def g (x : ℝ) : ℝ := sin (2 * x + π / 3) - Real.sqrt 3 / 2

theorem min_shift_value (k : ℝ) (h : k > 0) :
  (∀ x, f x = g (x - k)) ↔ k ≥ π / 3 :=
sorry

end NUMINAMATH_CALUDE_min_shift_value_l2122_212288


namespace NUMINAMATH_CALUDE_interval_intersection_l2122_212266

theorem interval_intersection (x : ℝ) : 
  (2 < 3*x ∧ 3*x < 5 ∧ 2 < 4*x ∧ 4*x < 5) ↔ (2/3 < x ∧ x < 5/4) :=
by sorry

end NUMINAMATH_CALUDE_interval_intersection_l2122_212266


namespace NUMINAMATH_CALUDE_farm_animal_count_l2122_212205

/-- Given a farm with cows and ducks, calculate the total number of animals --/
theorem farm_animal_count (total_legs : ℕ) (num_cows : ℕ) : total_legs = 42 → num_cows = 6 → ∃ (num_ducks : ℕ), num_cows + num_ducks = 15 := by
  sorry

end NUMINAMATH_CALUDE_farm_animal_count_l2122_212205


namespace NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l2122_212274

theorem quadratic_inequality_no_solution (m : ℝ) (h : m ≤ 1) :
  ¬∃ x : ℝ, x^2 + 2*x + 2 - m < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l2122_212274


namespace NUMINAMATH_CALUDE_exactly_four_sets_l2122_212267

-- Define the set S1 as {-1, 0, 1}
def S1 : Set Int := {-1, 0, 1}

-- Define the set S2 as {-2, 0, 2}
def S2 : Set Int := {-2, 0, 2}

-- Define the set R as {-2, 0, 1, 2}
def R : Set Int := {-2, 0, 1, 2}

-- Define the conditions for set A
def satisfiesConditions (A : Set Int) : Prop :=
  (A ∩ S1 = {0, 1}) ∧ (A ∪ S2 = R)

-- Theorem stating that there are exactly 4 sets satisfying the conditions
theorem exactly_four_sets :
  ∃! (s : Finset (Set Int)), (∀ A ∈ s, satisfiesConditions A) ∧ s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_exactly_four_sets_l2122_212267


namespace NUMINAMATH_CALUDE_hexagon_triangle_ratio_l2122_212213

theorem hexagon_triangle_ratio (s_h s_t : ℝ) (h : s_h > 0) (t : s_t > 0) :
  (3 * s_h^2 * Real.sqrt 3) / 2 = (s_t^2 * Real.sqrt 3) / 4 →
  s_t / s_h = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_triangle_ratio_l2122_212213


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2122_212285

theorem diophantine_equation_solution (x y z : ℤ) :
  x ≠ 0 → y ≠ 0 → z ≠ 0 → x + y + z ≠ 0 →
  (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = (1 : ℚ) / (x + y + z) →
  (z = -x - y) ∨ (y = -x - z) ∨ (x = -y - z) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2122_212285


namespace NUMINAMATH_CALUDE_table_tennis_racket_sales_l2122_212252

/-- Profit function for table tennis racket sales -/
def profit_function (c : ℝ) (x : ℝ) : ℝ :=
  let y := -10 * x + 900
  y * (x - c)

/-- Problem statement for table tennis racket sales -/
theorem table_tennis_racket_sales 
  (c : ℝ) 
  (max_price : ℝ) 
  (min_profit : ℝ) 
  (h1 : c = 50) 
  (h2 : max_price = 75) 
  (h3 : min_profit = 3000) :
  ∃ (optimal_price : ℝ) (max_profit : ℝ) (price_range : Set ℝ),
    -- 1. The monthly profit function
    (∀ x, profit_function c x = -10 * x^2 + 1400 * x - 45000) ∧
    -- 2. The optimal price and maximum profit
    (optimal_price = 70 ∧ 
     max_profit = profit_function c optimal_price ∧
     max_profit = 4000 ∧
     ∀ x, profit_function c x ≤ max_profit) ∧
    -- 3. The range of acceptable selling prices
    (price_range = {x | 60 ≤ x ∧ x ≤ 75} ∧
     ∀ x ∈ price_range, 
       x ≤ max_price ∧ 
       profit_function c x ≥ min_profit) :=
by sorry

end NUMINAMATH_CALUDE_table_tennis_racket_sales_l2122_212252


namespace NUMINAMATH_CALUDE_quadratic_with_integral_roots_exist_l2122_212217

theorem quadratic_with_integral_roots_exist : 
  ∃ (b c : ℝ), 
    (∃ (p q : ℤ), ∀ (x : ℝ), x^2 + b*x + c = 0 ↔ x = p ∨ x = q) ∧ 
    (∃ (r s : ℤ), ∀ (x : ℝ), x^2 + (b+1)*x + (c+1) = 0 ↔ x = r ∨ x = s) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_with_integral_roots_exist_l2122_212217


namespace NUMINAMATH_CALUDE_bug_probability_after_8_meters_l2122_212222

/-- Probability of the bug being at vertex A after n meters -/
def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/3 * (1 - P n)

/-- The probability of the bug being at vertex A after 8 meters is 547/2187 -/
theorem bug_probability_after_8_meters : P 8 = 547/2187 := by
  sorry

end NUMINAMATH_CALUDE_bug_probability_after_8_meters_l2122_212222


namespace NUMINAMATH_CALUDE_swaps_theorem_l2122_212206

/-- Represents a mode of letter swapping -/
inductive SwapMode
| Adjacent : SwapMode
| Any : SwapMode

/-- Represents a string of letters -/
def Text : Type := List Char

/-- Calculate the minimum number of swaps required to transform one text into another -/
def minSwaps (original : Text) (target : Text) (mode : SwapMode) : Nat :=
  match mode with
  | SwapMode.Adjacent => sorry
  | SwapMode.Any => sorry

/-- The original text -/
def originalText : Text := ['M', 'E', 'G', 'Y', 'E', 'I', ' ', 'T', 'A', 'K', 'A', 'R', 'É', 'K', 'P', 'É', 'N', 'Z', 'T', 'Á', 'R', ' ', 'R', '.', ' ', 'T', '.']

/-- The target text -/
def targetText : Text := ['T', 'A', 'T', 'Á', 'R', ' ', 'G', 'Y', 'E', 'R', 'M', 'E', 'K', ' ', 'A', ' ', 'P', 'É', 'N', 'Z', 'T', ' ', 'K', 'É', 'R', 'I', '.']

theorem swaps_theorem :
  (minSwaps originalText targetText SwapMode.Adjacent = 85) ∧
  (minSwaps originalText targetText SwapMode.Any = 11) :=
sorry

end NUMINAMATH_CALUDE_swaps_theorem_l2122_212206


namespace NUMINAMATH_CALUDE_domain_of_g_l2122_212251

-- Define the function f with domain [0,2]
def f : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- Define the function g(x) = f(x²)
def g (x : ℝ) : Prop := x^2 ∈ f

-- Theorem statement
theorem domain_of_g : 
  {x : ℝ | g x} = {x : ℝ | -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2} :=
by sorry

end NUMINAMATH_CALUDE_domain_of_g_l2122_212251


namespace NUMINAMATH_CALUDE_remainder_theorem_l2122_212270

theorem remainder_theorem (P D D' D'' Q Q' Q'' R R' R'' : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R')
  (h3 : Q' = Q'' * D'' + R'')
  (h4 : R < D)
  (h5 : R' < D')
  (h6 : R'' < D'') :
  P % (D * D' * D'') = R'' * D * D' + R' * D + R := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2122_212270


namespace NUMINAMATH_CALUDE_added_amount_l2122_212209

theorem added_amount (n : ℝ) (x : ℝ) (h1 : n = 12) (h2 : n / 2 + x = 11) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_added_amount_l2122_212209


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_intersection_l2122_212260

/-- Given a hyperbola, its asymptote, a parabola, and a circle, prove the value of a parameter. -/
theorem hyperbola_asymptote_intersection (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) →  -- Hyperbola equation
  (∃ x₀ : ℝ, y = 2*x₀*x - x₀^2 + 1) →      -- Asymptote equation (tangent to parabola)
  (∃ x y : ℝ, x^2 + (y - a)^2 = 1) →       -- Circle equation
  (∃ x₁ y₁ x₂ y₂ : ℝ,                      -- Chord endpoints
    x₁^2 + (y₁ - a)^2 = 1 ∧ 
    x₂^2 + (y₂ - a)^2 = 1 ∧ 
    y₁ = 2*x₀*x₁ - x₀^2 + 1 ∧ 
    y₂ = 2*x₀*x₂ - x₀^2 + 1 ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 2) →       -- Chord length
  a = Real.sqrt 10 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_intersection_l2122_212260


namespace NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l2122_212241

/-- The area of a circle with diameter endpoints C(-2,3) and D(4,-1) is 13π square units -/
theorem circle_area_from_diameter_endpoints : 
  let C : ℝ × ℝ := (-2, 3)
  let D : ℝ × ℝ := (4, -1)
  let diameter_length := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  let radius := diameter_length / 2
  let circle_area := π * radius^2
  circle_area = 13 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l2122_212241


namespace NUMINAMATH_CALUDE_conference_hall_tables_l2122_212273

theorem conference_hall_tables (chairs_per_table : ℕ) (chair_legs : ℕ) (table_legs : ℕ) (total_legs : ℕ) :
  chairs_per_table = 8 →
  chair_legs = 3 →
  table_legs = 5 →
  total_legs = 580 →
  ∃ (num_tables : ℕ), num_tables = 20 ∧ 
    chairs_per_table * num_tables * chair_legs + num_tables * table_legs = total_legs :=
by sorry

end NUMINAMATH_CALUDE_conference_hall_tables_l2122_212273


namespace NUMINAMATH_CALUDE_equation_solution_l2122_212219

theorem equation_solution : ∃! x : ℝ, x + ((2 / 3 * 3 / 8) + 4) - 8 / 16 = 4.25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2122_212219


namespace NUMINAMATH_CALUDE_police_catch_time_police_catch_rogue_l2122_212287

/-- The time it takes for a police spaceship to catch up with a rogue spaceship -/
theorem police_catch_time (rogue_speed : ℝ) (head_start_minutes : ℝ) (police_speed_increase : ℝ) : ℝ :=
  let head_start_hours := head_start_minutes / 60
  let police_speed := rogue_speed * (1 + police_speed_increase)
  let distance_traveled := rogue_speed * head_start_hours
  let relative_speed := police_speed - rogue_speed
  let catch_up_time_hours := distance_traveled / relative_speed
  catch_up_time_hours * 60

/-- The police will catch up with the rogue spaceship in 450 minutes -/
theorem police_catch_rogue : 
  ∀ (rogue_speed : ℝ), rogue_speed > 0 → police_catch_time rogue_speed 54 0.12 = 450 :=
by
  sorry


end NUMINAMATH_CALUDE_police_catch_time_police_catch_rogue_l2122_212287


namespace NUMINAMATH_CALUDE_smallest_positive_leading_coeff_l2122_212291

/-- A quadratic polynomial that takes integer values for all integer inputs. -/
def IntegerValuedQuadratic (a b c : ℚ) : ℤ → ℤ :=
  fun x => ⌊a * x^2 + b * x + c⌋

/-- The property that a quadratic polynomial takes integer values for all integer inputs. -/
def IsIntegerValued (a b c : ℚ) : Prop :=
  ∀ x : ℤ, (IntegerValuedQuadratic a b c x : ℚ) = a * x^2 + b * x + c

/-- The smallest positive leading coefficient of an integer-valued quadratic polynomial is 1/2. -/
theorem smallest_positive_leading_coeff :
  (∃ a b c : ℚ, a > 0 ∧ IsIntegerValued a b c) ∧
  (∀ a b c : ℚ, a > 0 → IsIntegerValued a b c → a ≥ 1/2) ∧
  (∃ b c : ℚ, IsIntegerValued (1/2) b c) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_leading_coeff_l2122_212291


namespace NUMINAMATH_CALUDE_abs_equation_solution_l2122_212232

theorem abs_equation_solution :
  ∃! x : ℝ, |x + 4| = 3 - x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_l2122_212232


namespace NUMINAMATH_CALUDE_candy_left_l2122_212258

theorem candy_left (houses : ℕ) (candies_per_house : ℕ) (people : ℕ) (candies_eaten_per_person : ℕ) : 
  houses = 15 → 
  candies_per_house = 8 → 
  people = 3 → 
  candies_eaten_per_person = 6 → 
  houses * candies_per_house - people * candies_eaten_per_person = 102 := by
sorry

end NUMINAMATH_CALUDE_candy_left_l2122_212258


namespace NUMINAMATH_CALUDE_salary_increase_after_employee_reduction_l2122_212240

theorem salary_increase_after_employee_reduction (E : ℝ) (S : ℝ) (h1 : E > 0) (h2 : S > 0) :
  let new_E := 0.9 * E
  let new_S := (E * S) / new_E
  (new_S - S) / S = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_salary_increase_after_employee_reduction_l2122_212240


namespace NUMINAMATH_CALUDE_sqrt_inequality_range_l2122_212242

theorem sqrt_inequality_range (x : ℝ) : 
  x > 0 → (Real.sqrt (2 * x) < 3 * x - 4 ↔ x > 2) := by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_range_l2122_212242


namespace NUMINAMATH_CALUDE_vector_c_value_l2122_212218

theorem vector_c_value (a b c : ℝ × ℝ) : 
  a = (-1, 2) → 
  b = (2, -3) → 
  a.1 * c.1 + a.2 * c.2 = -7 → 
  b.1 * c.1 + b.2 * c.2 = 12 → 
  c = (3, -2) := by
  sorry

end NUMINAMATH_CALUDE_vector_c_value_l2122_212218


namespace NUMINAMATH_CALUDE_weight_lifting_competition_l2122_212208

theorem weight_lifting_competition (total_weight first_lift : ℕ) 
  (h1 : total_weight = 1800)
  (h2 : first_lift = 700) :
  2 * first_lift - (total_weight - first_lift) = 300 := by
  sorry

end NUMINAMATH_CALUDE_weight_lifting_competition_l2122_212208


namespace NUMINAMATH_CALUDE_remainder_difference_l2122_212221

theorem remainder_difference (d r : ℕ) : 
  d > 1 → 
  2023 % d = r → 
  2459 % d = r → 
  3571 % d = r → 
  d - r = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_difference_l2122_212221


namespace NUMINAMATH_CALUDE_bird_stork_difference_is_one_l2122_212279

/-- Given an initial number of birds on a fence, and additional birds and storks that join,
    calculate the difference between the final number of storks and birds. -/
def fence_bird_stork_difference (initial_birds : ℕ) (joined_birds : ℕ) (joined_storks : ℕ) : ℤ :=
  (joined_storks : ℤ) - ((initial_birds + joined_birds) : ℤ)

/-- Theorem stating that with 3 initial birds, 2 joined birds, and 6 joined storks,
    there is 1 more stork than birds on the fence. -/
theorem bird_stork_difference_is_one :
  fence_bird_stork_difference 3 2 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_bird_stork_difference_is_one_l2122_212279


namespace NUMINAMATH_CALUDE_total_points_noa_and_phillip_l2122_212263

/-- 
Given that Noa scored 30 points and Phillip scored twice as many points as Noa,
prove that the total number of points scored by Noa and Phillip is 90.
-/
theorem total_points_noa_and_phillip : 
  let noa_points : ℕ := 30
  let phillip_points : ℕ := 2 * noa_points
  noa_points + phillip_points = 90 := by
sorry


end NUMINAMATH_CALUDE_total_points_noa_and_phillip_l2122_212263


namespace NUMINAMATH_CALUDE_vanessa_album_pictures_l2122_212280

/-- Represents the number of pictures in an album -/
def pictures_per_album (phone_pics camera_pics total_albums : ℕ) : ℚ :=
  (phone_pics + camera_pics : ℚ) / total_albums

/-- Theorem stating the number of pictures per album given the conditions -/
theorem vanessa_album_pictures :
  pictures_per_album 56 28 8 = 21/2 := by sorry

end NUMINAMATH_CALUDE_vanessa_album_pictures_l2122_212280


namespace NUMINAMATH_CALUDE_comic_book_arrangements_l2122_212228

def spiderman_comics : ℕ := 6
def archie_comics : ℕ := 5
def garfield_comics : ℕ := 4
def batman_comics : ℕ := 7

def total_arrangements : ℕ := 59536691200

theorem comic_book_arrangements :
  (spiderman_comics.factorial * archie_comics.factorial * garfield_comics.factorial * batman_comics.factorial) *
  (spiderman_comics + archie_comics + garfield_comics + batman_comics - 3).factorial = total_arrangements := by
  sorry

end NUMINAMATH_CALUDE_comic_book_arrangements_l2122_212228


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2122_212238

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop := y = (4/3) * x ∨ y = -(4/3) * x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 0) ↔ (y = (4/3) * x ∨ y = -(4/3) * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2122_212238


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2122_212230

/-- The equation 9x^2 + nx + 1 = 0 has exactly one solution in x if and only if n = 6 -/
theorem unique_solution_quadratic (n : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + n * x + 1 = 0) ↔ n = 6 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2122_212230


namespace NUMINAMATH_CALUDE_sheela_monthly_income_l2122_212234

theorem sheela_monthly_income (deposit : ℝ) (percentage : ℝ) (monthly_income : ℝ) :
  deposit = 4500 →
  percentage = 28 →
  deposit = percentage / 100 * monthly_income →
  monthly_income = 16071.43 := by
  sorry

end NUMINAMATH_CALUDE_sheela_monthly_income_l2122_212234


namespace NUMINAMATH_CALUDE_range_of_a_l2122_212268

def has_real_roots (m : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁^2 - m*x₁ - 1 = 0 ∧ x₂^2 - m*x₂ - 1 = 0

def roots_difference_bound (a : ℝ) : Prop :=
  ∀ (m : ℝ), has_real_roots m → 
    ∃ (x₁ x₂ : ℝ), x₁^2 - m*x₁ - 1 = 0 ∧ x₂^2 - m*x₂ - 1 = 0 ∧ a^2 + 4*a - 3 ≤ |x₁ - x₂|

def quadratic_has_solution (a : ℝ) : Prop :=
  ∃ (x : ℝ), x^2 + 2*x + a < 0

def proposition_p (a : ℝ) : Prop :=
  has_real_roots 0 ∧ roots_difference_bound a

def proposition_q (a : ℝ) : Prop :=
  quadratic_has_solution a

theorem range_of_a :
  ∀ (a : ℝ), (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) →
    a = 1 ∨ a < -5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2122_212268


namespace NUMINAMATH_CALUDE_cos_sin_identity_l2122_212284

open Real

theorem cos_sin_identity : 
  cos (89 * π / 180) * cos (π / 180) + sin (91 * π / 180) * sin (181 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_identity_l2122_212284


namespace NUMINAMATH_CALUDE_wages_problem_l2122_212246

/-- Represents the wages of a group of people -/
structure Wages where
  men : ℕ
  women : ℕ
  boys : ℕ
  menWage : ℚ
  womenWage : ℚ
  boysWage : ℚ

/-- The problem statement -/
theorem wages_problem (w : Wages) (h1 : w.men = 5) (h2 : w.boys = 8) 
    (h3 : w.men * w.menWage = w.women * w.womenWage) 
    (h4 : w.women * w.womenWage = w.boys * w.boysWage)
    (h5 : w.men * w.menWage + w.women * w.womenWage + w.boys * w.boysWage = 60) :
  w.men * w.menWage = 30 := by
  sorry

end NUMINAMATH_CALUDE_wages_problem_l2122_212246


namespace NUMINAMATH_CALUDE_range_of_function_l2122_212201

theorem range_of_function (x : ℝ) : 
  1/3 ≤ (2 - Real.cos x) / (2 + Real.cos x) ∧ (2 - Real.cos x) / (2 + Real.cos x) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l2122_212201


namespace NUMINAMATH_CALUDE_picture_book_shelves_l2122_212294

theorem picture_book_shelves (total_books : ℕ) (mystery_shelves : ℕ) (books_per_shelf : ℕ) :
  total_books = 32 →
  mystery_shelves = 5 →
  books_per_shelf = 4 →
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 3 :=
by sorry

end NUMINAMATH_CALUDE_picture_book_shelves_l2122_212294


namespace NUMINAMATH_CALUDE_equivalent_discount_l2122_212212

/-- Proves that a single discount of 40.5% on a $50 item results in the same final price
    as applying a 30% discount followed by a 15% discount on the discounted price. -/
theorem equivalent_discount (original_price : ℝ) (first_discount second_discount single_discount : ℝ) :
  original_price = 50 ∧
  first_discount = 0.3 ∧
  second_discount = 0.15 ∧
  single_discount = 0.405 →
  original_price * (1 - single_discount) =
  original_price * (1 - first_discount) * (1 - second_discount) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_discount_l2122_212212


namespace NUMINAMATH_CALUDE_inequality_proof_l2122_212297

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2122_212297


namespace NUMINAMATH_CALUDE_exists_square_function_l2122_212278

theorem exists_square_function : ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_square_function_l2122_212278


namespace NUMINAMATH_CALUDE_simplify_product_of_square_roots_l2122_212214

theorem simplify_product_of_square_roots (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (32 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) * Real.sqrt (72 * y) = 960 * y^2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_square_roots_l2122_212214


namespace NUMINAMATH_CALUDE_halloween_candy_theorem_l2122_212253

/-- Calculates the remaining candy after Debby and her sister combine their Halloween candy and eat some. -/
def remaining_candy (debby_candy : ℕ) (sister_candy : ℕ) (eaten_candy : ℕ) : ℕ :=
  debby_candy + sister_candy - eaten_candy

/-- Theorem stating that the remaining candy is correct given the initial conditions. -/
theorem halloween_candy_theorem (debby_candy sister_candy eaten_candy : ℕ) :
  remaining_candy debby_candy sister_candy eaten_candy = debby_candy + sister_candy - eaten_candy :=
by sorry

end NUMINAMATH_CALUDE_halloween_candy_theorem_l2122_212253


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2122_212264

-- Define the quadratic function
def f (k x : ℝ) := k * x^2 - 2 * x + 6 * k

-- Define the solution set
def solution_set (k : ℝ) := {x : ℝ | f k x < 0}

-- Define the interval (2, 3)
def interval := {x : ℝ | 2 < x ∧ x < 3}

theorem quadratic_inequality_theorem (k : ℝ) (h : k > 0) :
  (solution_set k = interval → k = 2/5) ∧
  (∀ x ∈ interval, f k x < 0 → 0 < k ∧ k ≤ 2/5) ∧
  (solution_set k ⊆ interval → 2/5 ≤ k) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2122_212264


namespace NUMINAMATH_CALUDE_quadratic_inequality_minimum_l2122_212211

theorem quadratic_inequality_minimum (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → x^2 + 2*x + a ≥ 0) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_minimum_l2122_212211


namespace NUMINAMATH_CALUDE_min_value_theorem_l2122_212272

theorem min_value_theorem (x y z : ℝ) (h : (1 / x) + (2 / y) + (3 / z) = 1) :
  x + y / 2 + z / 3 ≥ 9 ∧
  (x + y / 2 + z / 3 = 9 ↔ x = y / 2 ∧ y / 2 = z / 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2122_212272


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2122_212210

theorem right_triangle_hypotenuse (PQ PR : ℝ) (h1 : PQ = 15) (h2 : PR = 20) :
  Real.sqrt (PQ^2 + PR^2) = 25 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2122_212210


namespace NUMINAMATH_CALUDE_initial_cost_calculation_l2122_212286

/-- Represents the car rental cost structure and usage --/
structure CarRental where
  initialCost : ℝ
  costPerMile : ℝ
  milesDriven : ℝ
  totalCost : ℝ

/-- Theorem stating the initial cost of the car rental --/
theorem initial_cost_calculation (rental : CarRental) 
    (h1 : rental.costPerMile = 0.50)
    (h2 : rental.milesDriven = 1364)
    (h3 : rental.totalCost = 832) :
    rental.initialCost = 150 := by
  sorry


end NUMINAMATH_CALUDE_initial_cost_calculation_l2122_212286


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_trigonometric_fraction_simplification_l2122_212225

-- Problem 1
theorem trigonometric_expression_equality : 
  Real.cos (2/3 * Real.pi) - Real.tan (-Real.pi/4) + 3/4 * Real.tan (Real.pi/6) - Real.sin (-31/6 * Real.pi) = Real.sqrt 3 / 4 := by
  sorry

-- Problem 2
theorem trigonometric_fraction_simplification (α : Real) : 
  (Real.sin (Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.cos (-α + 3/2 * Real.pi)) / 
  (Real.cos (Real.pi/2 - α) * Real.sin (-Real.pi - α)) = -Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_trigonometric_fraction_simplification_l2122_212225


namespace NUMINAMATH_CALUDE_percentage_problem_l2122_212200

/-- The problem statement --/
theorem percentage_problem (P : ℝ) : 
  (P / 100) * 200 = (60 / 100) * 50 + 30 → P = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2122_212200


namespace NUMINAMATH_CALUDE_quadratic_minimum_zero_l2122_212257

/-- Given a quadratic function y = (1+a)x^2 + px + q with a minimum value of zero,
    where a is a positive constant, prove that q = p^2 / (4(1+a)). -/
theorem quadratic_minimum_zero (a p q : ℝ) (ha : a > 0) :
  (∃ (k : ℝ), ∀ (x : ℝ), (1 + a) * x^2 + p * x + q ≥ k) ∧ 
  (∃ (x : ℝ), (1 + a) * x^2 + p * x + q = 0) →
  q = p^2 / (4 * (1 + a)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_zero_l2122_212257


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_constraint_l2122_212277

/-- Given vectors a and b, if the magnitude of their sum does not exceed 5,
    then the second component of b is in the range [-6, 2]. -/
theorem vector_sum_magnitude_constraint (a b : ℝ × ℝ) (h : ‖a + b‖ ≤ 5) :
  a = (-2, 2) → b.1 = 5 → -6 ≤ b.2 ∧ b.2 ≤ 2 := by
  sorry

#check vector_sum_magnitude_constraint

end NUMINAMATH_CALUDE_vector_sum_magnitude_constraint_l2122_212277


namespace NUMINAMATH_CALUDE_sweater_markup_l2122_212276

theorem sweater_markup (wholesale_cost : ℝ) (retail_price : ℝ) :
  retail_price > 0 →
  wholesale_cost > 0 →
  (retail_price * 0.4 = wholesale_cost * 1.35) →
  ((retail_price - wholesale_cost) / wholesale_cost) * 100 = 237.5 := by
sorry

end NUMINAMATH_CALUDE_sweater_markup_l2122_212276


namespace NUMINAMATH_CALUDE_complex_number_problem_l2122_212249

theorem complex_number_problem (a : ℝ) : 
  let z : ℂ := (a / Complex.I) + ((1 - Complex.I) / 2) * Complex.I
  (z.re = 0 ∨ z.im = 0) ∧ (z.re + z.im = 0) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2122_212249


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2122_212202

theorem rationalize_denominator :
  (Real.sqrt 18 - Real.sqrt 8) / (Real.sqrt 8 + Real.sqrt 2) = (1 + Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2122_212202


namespace NUMINAMATH_CALUDE_special_line_equation_l2122_212237

/-- A line passing through a point and at a fixed distance from the origin -/
structure SpecialLine where
  a : ℝ  -- x-coordinate of the point
  b : ℝ  -- y-coordinate of the point
  d : ℝ  -- distance from the origin

/-- The equation of the special line -/
def lineEquation (l : SpecialLine) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = l.a ∨ 3 * p.1 + 4 * p.2 - 5 = 0}

/-- Theorem: The equation of the line passing through (-1, 2) and at a distance of 1 from the origin -/
theorem special_line_equation :
  let l : SpecialLine := ⟨-1, 2, 1⟩
  lineEquation l = {p : ℝ × ℝ | p.1 = -1 ∨ 3 * p.1 + 4 * p.2 - 5 = 0} := by
  sorry


end NUMINAMATH_CALUDE_special_line_equation_l2122_212237


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2122_212283

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a4 : a 4 = 3)
  (h_sum : a 3 + a 11 = 18) :
  (∀ n : ℕ, a n = 2 * n - 5) ∧
  (a 55 = 105) ∧
  (∃ n : ℕ, n * (2 * n - 8) / 2 = 32 ∧ n = 8) ∧
  (∀ n : ℕ, n * (2 * n - 8) / 2 ≥ 2 * (2 * 2 - 8) / 2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2122_212283


namespace NUMINAMATH_CALUDE_sum_parts_is_24_l2122_212229

/-- A rectangular prism with two opposite corners colored red -/
structure ColoredRectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ
  red_corners : ℕ
  h_red_corners : red_corners = 2

/-- The sum of edges, non-red corners, and faces of a colored rectangular prism -/
def sum_parts (prism : ColoredRectangularPrism) : ℕ :=
  12 + (8 - prism.red_corners) + 6

theorem sum_parts_is_24 (prism : ColoredRectangularPrism) :
  sum_parts prism = 24 :=
sorry

end NUMINAMATH_CALUDE_sum_parts_is_24_l2122_212229


namespace NUMINAMATH_CALUDE_min_set_size_l2122_212275

theorem min_set_size (n : ℕ) 
  (h1 : ∃ (s : Finset ℝ), s.card = 2*n + 1)
  (h2 : ∃ (s1 s2 : Finset ℝ), s1.card = n + 1 ∧ s2.card = n ∧ 
        (∀ x ∈ s1, x ≥ 10) ∧ (∀ x ∈ s2, x ≥ 1))
  (h3 : ∃ (s : Finset ℝ), s.card = 2*n + 1 ∧ 
        (Finset.sum s id) / (2*n + 1 : ℝ) = 6) :
  n ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_set_size_l2122_212275


namespace NUMINAMATH_CALUDE_systematic_sampling_elimination_l2122_212231

/-- The number of individuals randomly eliminated in a systematic sampling -/
def individuals_eliminated (population : ℕ) (sample_size : ℕ) : ℕ :=
  population % sample_size

/-- Theorem: The number of individuals randomly eliminated in a systematic sampling
    of 50 students from a population of 1252 is equal to 2 -/
theorem systematic_sampling_elimination :
  individuals_eliminated 1252 50 = 2 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_elimination_l2122_212231


namespace NUMINAMATH_CALUDE_simplify_fraction_l2122_212243

theorem simplify_fraction (a : ℝ) (h : a ≠ 2) :
  (a - 2) * ((a^2 - 4) / (a^2 - 4*a + 4)) = a + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2122_212243


namespace NUMINAMATH_CALUDE_opposite_men_exist_l2122_212203

/-- A circular arrangement of people -/
structure CircularArrangement where
  total : ℕ
  men : ℕ
  isMoreThanHalfMen : men > total / 2

/-- Two people are opposite if they are half the total distance apart -/
def areOpposite (n : ℕ) (a b : ℕ) : Prop :=
  (a - b) % n = n / 2 ∨ (b - a) % n = n / 2

theorem opposite_men_exist (arr : CircularArrangement) 
    (h : arr.total = 100) : 
    ∃ (a b : ℕ), a < arr.total ∧ b < arr.total ∧ 
                  a ≠ b ∧ 
                  areOpposite arr.total a b ∧ 
                  (a < arr.men ∧ b < arr.men) := by
  sorry

end NUMINAMATH_CALUDE_opposite_men_exist_l2122_212203


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2122_212255

theorem polynomial_factorization (a : ℝ) :
  (a^2 + 2*a)*(a^2 + 2*a + 2) + 1 = (a + 1)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2122_212255


namespace NUMINAMATH_CALUDE_tangent_slope_exponential_l2122_212295

theorem tangent_slope_exponential (x : ℝ) :
  let f : ℝ → ℝ := fun x => Real.exp x
  (deriv f) 1 = Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_exponential_l2122_212295


namespace NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l2122_212224

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^4 * a = a^5 := by sorry

end NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l2122_212224


namespace NUMINAMATH_CALUDE_expression_simplification_l2122_212223

theorem expression_simplification (m n : ℚ) (hm : m = 1) (hn : n = -3) :
  2/3 * (6*m - 9*m*n) - (n^2 - 6*m*n) = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2122_212223


namespace NUMINAMATH_CALUDE_divides_implies_equal_l2122_212227

theorem divides_implies_equal (a b : ℕ+) : 
  (4 * a * b - 1) ∣ ((4 * a^2 - 1)^2) → a = b := by
  sorry

end NUMINAMATH_CALUDE_divides_implies_equal_l2122_212227


namespace NUMINAMATH_CALUDE_exists_number_divisible_by_2n_with_only_1_and_2_l2122_212290

/-- A function that checks if a natural number only contains digits 1 and 2 in its decimal representation -/
def onlyOneAndTwo (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 1 ∨ d = 2

/-- For every natural number n, there exists a number divisible by 2^n 
    whose decimal representation uses only the digits 1 and 2 -/
theorem exists_number_divisible_by_2n_with_only_1_and_2 :
  ∀ n : ℕ, ∃ N : ℕ, 2^n ∣ N ∧ onlyOneAndTwo N :=
by sorry

end NUMINAMATH_CALUDE_exists_number_divisible_by_2n_with_only_1_and_2_l2122_212290


namespace NUMINAMATH_CALUDE_cube_sum_problem_l2122_212239

theorem cube_sum_problem (a b c : ℝ) 
  (sum_eq : a + b + c = 5)
  (sum_prod_eq : b * c + c * a + a * b = 7)
  (prod_eq : a * b * c = 2) :
  a^3 + b^3 + c^3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l2122_212239


namespace NUMINAMATH_CALUDE_derivative_sin_cos_product_l2122_212250

theorem derivative_sin_cos_product (x : ℝ) :
  deriv (fun x => 2 * Real.sin x * Real.cos x) x = 2 * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_cos_product_l2122_212250


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2122_212265

theorem quadratic_factorization (h b c d : ℤ) : 
  (∀ x : ℝ, 6 * x^2 + x - 12 = (h * x + b) * (c * x + d)) → 
  |h| + |b| + |c| + |d| = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2122_212265


namespace NUMINAMATH_CALUDE_selling_price_calculation_l2122_212204

theorem selling_price_calculation (cost_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  cost_price = 22500 →
  discount_rate = 0.1 →
  profit_rate = 0.08 →
  (1 - discount_rate) * (cost_price * (1 + profit_rate)) = 24300 →
  cost_price * (1 + profit_rate) / (1 - discount_rate) = 27000 := by
  sorry

#check selling_price_calculation

end NUMINAMATH_CALUDE_selling_price_calculation_l2122_212204


namespace NUMINAMATH_CALUDE_rectangle_width_l2122_212289

/-- Given a rectangle with length 3 inches and unknown width, and a square with width 5 inches,
    if the difference in area between the square and the rectangle is 7 square inches,
    then the width of the rectangle is 6 inches. -/
theorem rectangle_width (w : ℝ) : 
  (5 * 5 : ℝ) - (3 * w) = 7 → w = 6 := by sorry

end NUMINAMATH_CALUDE_rectangle_width_l2122_212289


namespace NUMINAMATH_CALUDE_multiplication_with_fraction_l2122_212281

theorem multiplication_with_fraction : 8 * (1 / 7) * 14 = 16 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_with_fraction_l2122_212281


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_l2122_212292

/-- Given a quadratic function f(x) = 2x^2 - x + 7, when shifted 6 units to the right,
    the resulting function g(x) = ax^2 + bx + c satisfies a + b + c = 62 -/
theorem quadratic_shift_sum (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 2 * x^2 - x + 7) →
  (∀ x, g x = f (x - 6)) →
  (∀ x, g x = a * x^2 + b * x + c) →
  a + b + c = 62 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_sum_l2122_212292


namespace NUMINAMATH_CALUDE_officer_selection_theorem_l2122_212233

def total_candidates : ℕ := 20
def officer_positions : ℕ := 6
def past_officers : ℕ := 5

theorem officer_selection_theorem :
  (Nat.choose past_officers 1 * Nat.choose (total_candidates - past_officers) (officer_positions - 1)) +
  (Nat.choose past_officers 2 * Nat.choose (total_candidates - past_officers) (officer_positions - 2)) +
  (Nat.choose past_officers 3 * Nat.choose (total_candidates - past_officers) (officer_positions - 3)) = 33215 := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_theorem_l2122_212233


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2122_212235

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - (1/2) * a 8 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2122_212235


namespace NUMINAMATH_CALUDE_intersection_and_length_l2122_212248

-- Define the coordinate system
variable (O : ℝ × ℝ)
variable (A : ℝ × ℝ)
variable (B : ℝ × ℝ)

-- Define lines l₁ and l₂
def l₁ (p : ℝ × ℝ) : Prop := p.1 + p.2 = 4
def l₂ (p : ℝ × ℝ) : Prop := p.2 = 2 * p.1

-- Define the conditions
axiom O_origin : O = (0, 0)
axiom A_on_l₁ : l₁ A
axiom A_on_l₂ : l₂ A
axiom B_on_l₁ : l₁ B
axiom OA_perp_OB : (A.1 * B.1 + A.2 * B.2) = 0

-- State the theorem
theorem intersection_and_length :
  A = (4/3, 8/3) ∧
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 20 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_and_length_l2122_212248


namespace NUMINAMATH_CALUDE_problem_solution_l2122_212236

def p (x : ℝ) : Prop := x^2 ≤ 5*x - 4

def q (x a : ℝ) : Prop := x^2 - (a + 2)*x + 2*a ≤ 0

theorem problem_solution :
  (∀ x : ℝ, ¬(p x) ↔ (x < 1 ∨ x > 4)) ∧
  (∀ a : ℝ, (∀ x : ℝ, q x a → p x) ∧ (∃ x : ℝ, p x ∧ ¬(q x a)) ↔ (1 ≤ a ∧ a ≤ 4)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2122_212236


namespace NUMINAMATH_CALUDE_tire_purchase_l2122_212299

theorem tire_purchase (cost_per_tire : ℚ) (total_cost : ℚ) (num_tires : ℕ) : 
  cost_per_tire = 1/2 →
  total_cost = 4 →
  num_tires = (total_cost / cost_per_tire).num →
  num_tires = 8 := by
sorry

end NUMINAMATH_CALUDE_tire_purchase_l2122_212299


namespace NUMINAMATH_CALUDE_dot_product_theorem_l2122_212220

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 2)

theorem dot_product_theorem (c : ℝ × ℝ) 
  (h : c = (3 * a.1 + 2 * b.1 - a.1, 3 * a.2 + 2 * b.2 - a.2)) :
  a.1 * c.1 + a.2 * c.2 = 4 := by sorry

end NUMINAMATH_CALUDE_dot_product_theorem_l2122_212220


namespace NUMINAMATH_CALUDE_sum_product_inequality_l2122_212226

theorem sum_product_inequality (a b c d : ℝ) (h : a + b + c + d = 0) :
  5 * (a * b + b * c + c * d) + 8 * (a * c + a * d + b * d) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l2122_212226


namespace NUMINAMATH_CALUDE_complex_real_condition_l2122_212298

theorem complex_real_condition (a : ℝ) : 
  (Complex.mk (1 / (a + 5)) (a^2 + 2*a - 15)).im = 0 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l2122_212298


namespace NUMINAMATH_CALUDE_joe_trading_cards_l2122_212244

theorem joe_trading_cards (cards_per_box : ℕ) (num_boxes : ℕ) (h1 : cards_per_box = 8) (h2 : num_boxes = 11) :
  cards_per_box * num_boxes = 88 := by
sorry

end NUMINAMATH_CALUDE_joe_trading_cards_l2122_212244


namespace NUMINAMATH_CALUDE_least_common_multiple_7_6_4_l2122_212261

theorem least_common_multiple_7_6_4 : ∃ (n : ℕ), n > 0 ∧ 7 ∣ n ∧ 6 ∣ n ∧ 4 ∣ n ∧ ∀ (m : ℕ), m > 0 → 7 ∣ m → 6 ∣ m → 4 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_common_multiple_7_6_4_l2122_212261


namespace NUMINAMATH_CALUDE_percentage_fraction_equality_l2122_212262

theorem percentage_fraction_equality : 
  (85 / 100 * 40) - (4 / 5 * 25) = 14 := by
sorry

end NUMINAMATH_CALUDE_percentage_fraction_equality_l2122_212262


namespace NUMINAMATH_CALUDE_employed_males_percentage_l2122_212215

/-- Given a population where 60% are employed and 20% of the employed are females,
    prove that 48% of the population are employed males. -/
theorem employed_males_percentage
  (total_population : ℕ) 
  (employed_percentage : ℚ) 
  (employed_females_percentage : ℚ) 
  (h1 : employed_percentage = 60 / 100)
  (h2 : employed_females_percentage = 20 / 100) :
  (employed_percentage - employed_percentage * employed_females_percentage) * 100 = 48 := by
  sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l2122_212215


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l2122_212216

theorem shaded_region_perimeter (r : ℝ) : 
  r > 0 →
  2 * Real.pi * r = 24 →
  (3 : ℝ) * (1 / 6 : ℝ) * (2 * Real.pi * r) = 12 :=
by sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l2122_212216


namespace NUMINAMATH_CALUDE_min_frames_for_18x15_grid_l2122_212256

/-- Represents a grid with given dimensions -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a square frame with side length 1 -/
structure Frame :=
  (side_length : ℕ := 1)

/-- Calculates the minimum number of frames needed to cover the grid -/
def min_frames_needed (g : Grid) : ℕ :=
  g.rows * g.cols - ((g.rows - 2) / 2 * (g.cols - 2))

/-- The theorem stating the minimum number of frames needed for an 18x15 grid -/
theorem min_frames_for_18x15_grid :
  let g : Grid := ⟨18, 15⟩
  min_frames_needed g = 166 := by
  sorry

#eval min_frames_needed ⟨18, 15⟩

end NUMINAMATH_CALUDE_min_frames_for_18x15_grid_l2122_212256


namespace NUMINAMATH_CALUDE_pizza_division_l2122_212254

theorem pizza_division (total_pizza : ℚ) (num_employees : ℕ) :
  total_pizza = 5 / 8 ∧ num_employees = 4 →
  total_pizza / num_employees = 5 / 32 := by
  sorry

end NUMINAMATH_CALUDE_pizza_division_l2122_212254


namespace NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l2122_212293

def f (a x : ℝ) : ℝ := x^2 - 2*(a+1)*x + 1

theorem even_function_implies_a_eq_neg_one (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l2122_212293


namespace NUMINAMATH_CALUDE_cubic_root_squared_l2122_212269

theorem cubic_root_squared (r : ℝ) : 
  r^3 - r + 3 = 0 → (r^2)^3 - 2*(r^2)^2 + r^2 - 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_squared_l2122_212269


namespace NUMINAMATH_CALUDE_y_value_when_x_is_one_l2122_212259

-- Define the inverse square relationship between x and y
def inverse_square_relation (k : ℝ) (x y : ℝ) : Prop :=
  x = k / (y ^ 2)

-- Theorem statement
theorem y_value_when_x_is_one 
  (k : ℝ) 
  (h1 : inverse_square_relation k 0.1111111111111111 6) 
  (h2 : k > 0) :
  ∃ y : ℝ, inverse_square_relation k 1 y ∧ y = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_y_value_when_x_is_one_l2122_212259


namespace NUMINAMATH_CALUDE_triangle_separation_l2122_212207

/-- A triangle in a 2D plane -/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- Check if two triangles have no common interior or boundary points -/
def no_common_points (t1 t2 : Triangle) : Prop := sorry

/-- Check if a line separates two triangles -/
def separates (line : ℝ × ℝ → Prop) (t1 t2 : Triangle) : Prop := sorry

/-- Check if a line is a side of a triangle -/
def is_side (line : ℝ × ℝ → Prop) (t : Triangle) : Prop := sorry

/-- Main theorem: For any two triangles with no common points, 
    there exists a side of one triangle that separates them -/
theorem triangle_separation (t1 t2 : Triangle) 
  (h : no_common_points t1 t2) : 
  ∃ (line : ℝ × ℝ → Prop), 
    (is_side line t1 ∨ is_side line t2) ∧ 
    separates line t1 t2 := by sorry

end NUMINAMATH_CALUDE_triangle_separation_l2122_212207


namespace NUMINAMATH_CALUDE_pigeons_flew_in_l2122_212296

theorem pigeons_flew_in (initial_count final_count : ℕ) 
  (h_initial : initial_count = 15)
  (h_final : final_count = 21) :
  final_count - initial_count = 6 := by
  sorry

end NUMINAMATH_CALUDE_pigeons_flew_in_l2122_212296


namespace NUMINAMATH_CALUDE_mod_twelve_power_six_l2122_212271

theorem mod_twelve_power_six (n : ℕ) : 12^6 ≡ n [ZMOD 9] → 0 ≤ n → n < 9 → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_mod_twelve_power_six_l2122_212271


namespace NUMINAMATH_CALUDE_complement_M_union_N_eq_nonneg_reals_l2122_212245

-- Define the set of real numbers
variable (r : Set ℝ)

-- Define set M
def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (1 - 2/x)}

-- Define set N
def N : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 1)}

-- Statement to prove
theorem complement_M_union_N_eq_nonneg_reals :
  (Set.univ \ M) ∪ N = Set.Ici (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_complement_M_union_N_eq_nonneg_reals_l2122_212245


namespace NUMINAMATH_CALUDE_expression_equality_l2122_212247

theorem expression_equality : (8 * 10^10) / (2 * 10^5 * 4) = 100000 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2122_212247


namespace NUMINAMATH_CALUDE_hundred_brick_tower_heights_l2122_212282

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  length : Nat
  width : Nat
  height : Nat

/-- Calculates the number of different tower heights achievable -/
def towerHeights (brickCount : Nat) (dimensions : BrickDimensions) : Nat :=
  sorry

/-- The main theorem stating the number of different tower heights -/
theorem hundred_brick_tower_heights :
  let brickDims : BrickDimensions := { length := 3, width := 11, height := 18 }
  towerHeights 100 brickDims = 1404 := by
  sorry

end NUMINAMATH_CALUDE_hundred_brick_tower_heights_l2122_212282
