import Mathlib

namespace NUMINAMATH_CALUDE_triangle_area_l3692_369243

theorem triangle_area (a b c : ℝ) (ha : a = 10) (hb : b = 24) (hc : c = 26) :
  (1 / 2) * a * b = 120 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3692_369243


namespace NUMINAMATH_CALUDE_triangle_inequality_l3692_369203

theorem triangle_inequality (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_perimeter : a + b + c = 1) : 
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3692_369203


namespace NUMINAMATH_CALUDE_problem_solution_l3692_369202

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 7 * x^2 + 21 * x * y = x^3 + 3 * x^2 * y^2) : x = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3692_369202


namespace NUMINAMATH_CALUDE_gala_luncheon_croissant_cost_l3692_369235

/-- Calculates the cost of croissants for a gala luncheon --/
theorem gala_luncheon_croissant_cost
  (people : ℕ)
  (sandwiches_per_person : ℕ)
  (croissants_per_set : ℕ)
  (cost_per_set : ℚ)
  (h1 : people = 24)
  (h2 : sandwiches_per_person = 2)
  (h3 : croissants_per_set = 12)
  (h4 : cost_per_set = 8) :
  (people * sandwiches_per_person / croissants_per_set : ℚ) * cost_per_set = 32 := by
  sorry

#check gala_luncheon_croissant_cost

end NUMINAMATH_CALUDE_gala_luncheon_croissant_cost_l3692_369235


namespace NUMINAMATH_CALUDE_number_equation_solution_l3692_369241

theorem number_equation_solution : ∃ x : ℝ, 5 * x + 4 = 19 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3692_369241


namespace NUMINAMATH_CALUDE_min_T_minus_S_and_max_T_l3692_369231

/-- Given non-negative real numbers a, b, and c, S and T are defined as follows:
    S = a + 2b + 3c
    T = a + b^2 + c^3 -/
def S (a b c : ℝ) : ℝ := a + 2*b + 3*c
def T (a b c : ℝ) : ℝ := a + b^2 + c^3

theorem min_T_minus_S_and_max_T (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (∀ a' b' c' : ℝ, 0 ≤ a' → 0 ≤ b' → 0 ≤ c' → -3 ≤ T a' b' c' - S a' b' c') ∧
  (S a b c = 4 → T a b c ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_min_T_minus_S_and_max_T_l3692_369231


namespace NUMINAMATH_CALUDE_refrigerator_cost_l3692_369279

/-- Proves that the cost of the refrigerator is 25000 given the problem conditions -/
theorem refrigerator_cost
  (mobile_cost : ℕ)
  (refrigerator_loss_percent : ℚ)
  (mobile_profit_percent : ℚ)
  (total_profit : ℕ)
  (h1 : mobile_cost = 8000)
  (h2 : refrigerator_loss_percent = 4 / 100)
  (h3 : mobile_profit_percent = 10 / 100)
  (h4 : total_profit = 200) :
  ∃ (refrigerator_cost : ℕ),
    refrigerator_cost = 25000 ∧
    (refrigerator_cost : ℚ) * (1 - refrigerator_loss_percent) +
    (mobile_cost : ℚ) * (1 + mobile_profit_percent) =
    (refrigerator_cost + mobile_cost + total_profit : ℚ) :=
sorry

end NUMINAMATH_CALUDE_refrigerator_cost_l3692_369279


namespace NUMINAMATH_CALUDE_porch_width_calculation_l3692_369296

/-- Given a house and porch with specific dimensions, calculate the width of the porch. -/
theorem porch_width_calculation (house_length house_width porch_length total_shingle_area : ℝ)
  (h1 : house_length = 20.5)
  (h2 : house_width = 10)
  (h3 : porch_length = 6)
  (h4 : total_shingle_area = 232) :
  let house_area := house_length * house_width
  let porch_area := total_shingle_area - house_area
  porch_area / porch_length = 4.5 := by sorry

end NUMINAMATH_CALUDE_porch_width_calculation_l3692_369296


namespace NUMINAMATH_CALUDE_divisors_of_squared_number_l3692_369244

theorem divisors_of_squared_number (n : ℕ) (h : n > 1) :
  (Finset.card (Nat.divisors n) = 4) → (Finset.card (Nat.divisors (n^2)) = 9) := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_squared_number_l3692_369244


namespace NUMINAMATH_CALUDE_symmetric_sine_cosine_l3692_369287

/-- A function f is symmetric about a line x = c if f(c + h) = f(c - h) for all h -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ h, f (c + h) = f (c - h)

/-- The main theorem -/
theorem symmetric_sine_cosine (a : ℝ) :
  SymmetricAbout (fun x ↦ Real.sin (2 * x) + a * Real.cos (2 * x)) (π / 8) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_sine_cosine_l3692_369287


namespace NUMINAMATH_CALUDE_selection_methods_equal_l3692_369253

def male_students : ℕ := 20
def female_students : ℕ := 30
def total_students : ℕ := male_students + female_students
def select_count : ℕ := 4

def selection_method_1 : ℕ := Nat.choose total_students select_count - 
                               Nat.choose male_students select_count - 
                               Nat.choose female_students select_count

def selection_method_2 : ℕ := Nat.choose male_students 1 * Nat.choose female_students 3 +
                               Nat.choose male_students 2 * Nat.choose female_students 2 +
                               Nat.choose male_students 3 * Nat.choose female_students 1

theorem selection_methods_equal : selection_method_1 = selection_method_2 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_equal_l3692_369253


namespace NUMINAMATH_CALUDE_isosceles_triangle_unique_range_l3692_369273

theorem isosceles_triangle_unique_range (a : ℝ) :
  (∃ (x y : ℝ), x^2 - 6*x + a = 0 ∧ y^2 - 6*y + a = 0 ∧ 
   x ≠ y ∧ 
   (x < y → 2*x ≤ y) ∧
   (y < x → 2*y ≤ x)) ↔
  (0 < a ∧ a ≤ 9) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_unique_range_l3692_369273


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3692_369270

theorem system_of_equations_solution 
  (x y z a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (eq1 : y * z / (y + z) = a)
  (eq2 : x * z / (x + z) = b)
  (eq3 : x * y / (x + y) = c) :
  x = 2 * a * b * c / (a * c + a * b - b * c) ∧
  y = 2 * a * b * c / (a * b + b * c - a * c) ∧
  z = 2 * a * b * c / (a * c + b * c - a * b) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3692_369270


namespace NUMINAMATH_CALUDE_x_squared_mod_25_l3692_369204

theorem x_squared_mod_25 (x : ℤ) (h1 : 5 * x ≡ 15 [ZMOD 25]) (h2 : 4 * x ≡ 12 [ZMOD 25]) :
  x^2 ≡ 9 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_25_l3692_369204


namespace NUMINAMATH_CALUDE_implication_not_equivalence_l3692_369218

theorem implication_not_equivalence :
  ∃ (a : ℝ), (∀ (x : ℝ), (abs (5 * x - 1) > a) → (x^2 - (3/2) * x + 1/2 > 0)) ∧
             (∃ (y : ℝ), (y^2 - (3/2) * y + 1/2 > 0) ∧ (abs (5 * y - 1) ≤ a)) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_implication_not_equivalence_l3692_369218


namespace NUMINAMATH_CALUDE_minimizes_f_l3692_369219

/-- The function f(x) defined in the problem -/
def f (a b x : ℝ) : ℝ := 3 * (x - a)^2 + 3 * (x - b)^2

/-- The statement that (a+b)/2 minimizes f(x) -/
theorem minimizes_f (a b : ℝ) :
  ∀ x : ℝ, f a b ((a + b) / 2) ≤ f a b x :=
sorry

end NUMINAMATH_CALUDE_minimizes_f_l3692_369219


namespace NUMINAMATH_CALUDE_tate_total_years_l3692_369290

/-- Calculates the total years spent by Tate in education and experiences --/
def totalYears (typicalHighSchoolYears : ℕ) : ℕ :=
  let highSchoolYears := typicalHighSchoolYears - 1
  let travelYears := 2
  let bachelorsYears := 2 * highSchoolYears
  let workExperienceYears := 1
  let phdYears := 3 * (highSchoolYears + bachelorsYears)
  highSchoolYears + travelYears + bachelorsYears + workExperienceYears + phdYears

/-- Theorem stating that Tate's total years spent is 39 --/
theorem tate_total_years : totalYears 4 = 39 := by
  sorry

end NUMINAMATH_CALUDE_tate_total_years_l3692_369290


namespace NUMINAMATH_CALUDE_triangle_ratio_l3692_369260

/-- An ellipse with foci on the x-axis -/
structure Ellipse where
  p : ℝ
  q : ℝ
  equation : ∀ x y : ℝ, x^2 / p^2 + y^2 / q^2 = 1

/-- An equilateral triangle -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_equilateral : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 
                   (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
                   (A.1 - B.1)^2 + (A.2 - B.2)^2 = 
                   (A.1 - C.1)^2 + (A.2 - C.2)^2

/-- The configuration described in the problem -/
structure Configuration where
  E : Ellipse
  T : EquilateralTriangle
  B_on_ellipse : T.B = (0, E.q)
  AC_parallel_x : T.A.2 = T.C.2
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  F₁_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
             F₁ = (t * T.B.1 + (1 - t) * T.C.1, t * T.B.2 + (1 - t) * T.C.2)
  F₂_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
             F₂ = (t * T.A.1 + (1 - t) * T.B.1, t * T.A.2 + (1 - t) * T.B.2)
  focal_distance : (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 4

theorem triangle_ratio (c : Configuration) : 
  let AB := ((c.T.A.1 - c.T.B.1)^2 + (c.T.A.2 - c.T.B.2)^2).sqrt
  let F₁F₂ := ((c.F₁.1 - c.F₂.1)^2 + (c.F₁.2 - c.F₂.2)^2).sqrt
  AB / F₁F₂ = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_l3692_369260


namespace NUMINAMATH_CALUDE_tennis_ball_count_l3692_369233

theorem tennis_ball_count : 
  ∀ (lily frodo brian sam : ℕ),
  lily = 12 →
  frodo = lily + 15 →
  brian = 3 * frodo →
  sam = frodo + lily - 5 →
  sam = 34 := by
sorry

end NUMINAMATH_CALUDE_tennis_ball_count_l3692_369233


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l3692_369266

theorem sum_of_squares_lower_bound (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (sum_eq_6 : a + b + c = 6) : 
  a^2 + b^2 + c^2 ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l3692_369266


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l3692_369254

theorem largest_digit_divisible_by_six : 
  ∀ N : ℕ, N ≤ 9 → (5678 * 10 + N) % 6 = 0 → N ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l3692_369254


namespace NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_equation_l3692_369215

theorem linear_coefficient_of_quadratic_equation :
  let equation := fun x : ℝ => x^2 - 2022*x - 2023
  ∃ a b c : ℝ, (∀ x, equation x = a*x^2 + b*x + c) ∧ b = -2022 :=
sorry

end NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_equation_l3692_369215


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3692_369274

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x + a| ≥ 3) ↔ (a ≤ -5 ∨ a ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3692_369274


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3692_369211

theorem quadratic_factorization : ∀ x : ℝ, 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3692_369211


namespace NUMINAMATH_CALUDE_circle_line_intersection_l3692_369238

/-- Given a circle and a line, if their intersection forms a chord of length 4, then the parameter 'a' in the circle equation equals -4 -/
theorem circle_line_intersection (x y : ℝ) (a : ℝ) : 
  (x^2 + y^2 + 2*x - 2*y + a = 0) →  -- Circle equation
  (x + y + 2 = 0) →                  -- Line equation
  (∃ p q : ℝ × ℝ, p ≠ q ∧            -- Existence of two distinct intersection points
    (p.1^2 + p.2^2 + 2*p.1 - 2*p.2 + a = 0) ∧
    (p.1 + p.2 + 2 = 0) ∧
    (q.1^2 + q.2^2 + 2*q.1 - 2*q.2 + a = 0) ∧
    (q.1 + q.2 + 2 = 0) ∧
    ((p.1 - q.1)^2 + (p.2 - q.2)^2 = 16)) → -- Chord length is 4
  a = -4 := by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3692_369238


namespace NUMINAMATH_CALUDE_potato_harvest_problem_l3692_369285

theorem potato_harvest_problem :
  ∃! (x y : ℕ+), 
    x * y * 5 = 45715 ∧ 
    x ≤ 100 ∧  -- reasonable upper bound for number of students
    y ≤ 1000   -- reasonable upper bound for daily output per student
  := by sorry

end NUMINAMATH_CALUDE_potato_harvest_problem_l3692_369285


namespace NUMINAMATH_CALUDE_sqrt_two_minus_a_l3692_369228

theorem sqrt_two_minus_a (a : ℝ) : a = -2 → Real.sqrt (2 - a) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_a_l3692_369228


namespace NUMINAMATH_CALUDE_jeremy_remaining_money_l3692_369214

/-- Given an initial amount and the costs of various items, calculate the remaining amount --/
def remaining_amount (initial : ℕ) (jersey_cost : ℕ) (jersey_count : ℕ) (basketball_cost : ℕ) (shorts_cost : ℕ) : ℕ :=
  initial - (jersey_cost * jersey_count + basketball_cost + shorts_cost)

/-- Prove that Jeremy has $14 left after his purchases --/
theorem jeremy_remaining_money :
  remaining_amount 50 2 5 18 8 = 14 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_remaining_money_l3692_369214


namespace NUMINAMATH_CALUDE_lg_6_equals_a_plus_b_l3692_369281

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem lg_6_equals_a_plus_b (a b : ℝ) (h1 : lg 2 = a) (h2 : lg 3 = b) : lg 6 = a + b := by
  sorry

end NUMINAMATH_CALUDE_lg_6_equals_a_plus_b_l3692_369281


namespace NUMINAMATH_CALUDE_at_least_one_not_greater_than_neg_four_l3692_369291

theorem at_least_one_not_greater_than_neg_four
  (a b c : ℝ)
  (ha : a < 0)
  (hb : b < 0)
  (hc : c < 0) :
  (a + 4 / b ≤ -4) ∨ (b + 4 / c ≤ -4) ∨ (c + 4 / a ≤ -4) := by
sorry

end NUMINAMATH_CALUDE_at_least_one_not_greater_than_neg_four_l3692_369291


namespace NUMINAMATH_CALUDE_rectangle_circle_intersection_area_l3692_369264

/-- The area of intersection between a rectangle and a circle with shared center -/
theorem rectangle_circle_intersection_area :
  ∀ (rectangle_length rectangle_width circle_radius : ℝ),
  rectangle_length = 10 →
  rectangle_width = 2 * Real.sqrt 3 →
  circle_radius = 3 →
  ∃ (intersection_area : ℝ),
  intersection_area = (9 * Real.pi) / 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_circle_intersection_area_l3692_369264


namespace NUMINAMATH_CALUDE_inequality_proof_l3692_369220

theorem inequality_proof (a b x₁ x₂ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) :
  (a * x₁ + b * x₂) * (a * x₂ + b * x₁) ≥ x₁ * x₂ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3692_369220


namespace NUMINAMATH_CALUDE_ali_nada_difference_l3692_369255

def total_amount : ℕ := 67
def john_amount : ℕ := 48

theorem ali_nada_difference (ali_amount nada_amount : ℕ) 
  (h1 : ali_amount + nada_amount + john_amount = total_amount)
  (h2 : ali_amount < nada_amount)
  (h3 : john_amount = 4 * nada_amount) :
  nada_amount - ali_amount = 5 := by
sorry

end NUMINAMATH_CALUDE_ali_nada_difference_l3692_369255


namespace NUMINAMATH_CALUDE_archie_marbles_l3692_369227

theorem archie_marbles (initial : ℕ) : 
  (initial : ℝ) * (1 - 0.6) * 0.5 = 20 → initial = 100 := by
  sorry

end NUMINAMATH_CALUDE_archie_marbles_l3692_369227


namespace NUMINAMATH_CALUDE_parallelogram_height_l3692_369294

/-- Given a parallelogram with area 576 cm² and base 32 cm, its height is 18 cm. -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 576 ∧ base = 32 ∧ area = base * height → height = 18 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_height_l3692_369294


namespace NUMINAMATH_CALUDE_cookie_count_pastry_shop_cookies_l3692_369213

/-- Given a ratio of doughnuts, cookies, and muffins, and the number of doughnuts and muffins,
    calculate the number of cookies. -/
theorem cookie_count (doughnut_ratio cookie_ratio muffin_ratio : ℕ) 
                     (doughnut_count muffin_count : ℕ) : ℕ :=
  let total_ratio := doughnut_ratio + cookie_ratio + muffin_ratio
  let part_value := doughnut_count / doughnut_ratio
  cookie_ratio * part_value

/-- Prove that given the ratio of doughnuts, cookies, and muffins is 5 : 3 : 1,
    and there are 50 doughnuts and 10 muffins, the number of cookies is 30. -/
theorem pastry_shop_cookies : cookie_count 5 3 1 50 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_pastry_shop_cookies_l3692_369213


namespace NUMINAMATH_CALUDE_remaining_flight_time_l3692_369205

def flight_duration : ℕ := 10 * 60  -- in minutes
def tv_episode_duration : ℕ := 25  -- in minutes
def num_tv_episodes : ℕ := 3
def sleep_duration : ℕ := 270  -- 4.5 hours in minutes
def movie_duration : ℕ := 105  -- 1 hour 45 minutes in minutes
def num_movies : ℕ := 2

theorem remaining_flight_time :
  flight_duration - (num_tv_episodes * tv_episode_duration + sleep_duration + num_movies * movie_duration) = 45 := by
  sorry

end NUMINAMATH_CALUDE_remaining_flight_time_l3692_369205


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3692_369206

theorem cubic_equation_roots (x : ℝ) : 
  ∃ (r₁ r₂ r₃ : ℝ), 
    (r₁ < 0 ∧ r₂ < 0 ∧ r₃ > 0) ∧
    (∀ y : ℝ, y^3 + 3*y^2 - 4*y - 12 = 0 ↔ y = r₁ ∨ y = r₂ ∨ y = r₃) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3692_369206


namespace NUMINAMATH_CALUDE_tan_graph_property_l3692_369288

theorem tan_graph_property (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + π))) → 
  a * Real.tan (b * (π / 4)) = 3 → 
  a * b = 3 := by
sorry

end NUMINAMATH_CALUDE_tan_graph_property_l3692_369288


namespace NUMINAMATH_CALUDE_unique_albums_count_l3692_369216

/-- Represents the album collections of Andrew, John, and Samantha -/
structure AlbumCollections where
  andrew_total : ℕ
  john_total : ℕ
  samantha_total : ℕ
  andrew_john_shared : ℕ
  andrew_samantha_shared : ℕ
  john_samantha_shared : ℕ

/-- Calculates the number of unique albums given the album collections -/
def uniqueAlbums (c : AlbumCollections) : ℕ :=
  (c.andrew_total - c.andrew_john_shared - c.andrew_samantha_shared) +
  (c.john_total - c.andrew_john_shared - c.john_samantha_shared) +
  (c.samantha_total - c.andrew_samantha_shared - c.john_samantha_shared)

/-- Theorem stating that the number of unique albums is 26 for the given collection -/
theorem unique_albums_count :
  let c : AlbumCollections := {
    andrew_total := 23,
    john_total := 20,
    samantha_total := 15,
    andrew_john_shared := 12,
    andrew_samantha_shared := 3,
    john_samantha_shared := 5
  }
  uniqueAlbums c = 26 := by
  sorry

end NUMINAMATH_CALUDE_unique_albums_count_l3692_369216


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l3692_369271

theorem reciprocal_inequality (a b : ℝ) : a < b → b < 0 → (1 : ℝ) / a > (1 : ℝ) / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l3692_369271


namespace NUMINAMATH_CALUDE_system_solution_l3692_369298

theorem system_solution (x y z : ℝ) : 
  x + y = 5 ∧ y + z = -1 ∧ x + z = -2 → x = 2 ∧ y = 3 ∧ z = -4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3692_369298


namespace NUMINAMATH_CALUDE_ratio_of_65_to_13_l3692_369217

theorem ratio_of_65_to_13 (certain_number : ℚ) (h : certain_number = 65) : 
  certain_number / 13 = 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_65_to_13_l3692_369217


namespace NUMINAMATH_CALUDE_ab_100_necessary_not_sufficient_for_log_sum_2_l3692_369250

theorem ab_100_necessary_not_sufficient_for_log_sum_2 :
  (∀ a b : ℝ, (Real.log a + Real.log b = 2) → (a * b = 100)) ∧
  (∃ a b : ℝ, a * b = 100 ∧ Real.log a + Real.log b ≠ 2) := by
  sorry

end NUMINAMATH_CALUDE_ab_100_necessary_not_sufficient_for_log_sum_2_l3692_369250


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l3692_369261

theorem smallest_four_digit_divisible_by_35 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 → n ≥ 1015 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l3692_369261


namespace NUMINAMATH_CALUDE_negation_equivalence_l3692_369207

theorem negation_equivalence :
  ¬(∀ x : ℝ, x^3 - x^2 + 2 < 0) ↔ ∃ x : ℝ, x^3 - x^2 + 2 ≥ 0 := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3692_369207


namespace NUMINAMATH_CALUDE_rose_group_size_l3692_369252

theorem rose_group_size (group_size : ℕ) : 
  (9 > 0) →
  (group_size > 0) →
  (Nat.lcm 9 group_size = 171) →
  (171 % group_size = 0) →
  (9 % group_size ≠ 0) →
  group_size = 19 := by
  sorry

end NUMINAMATH_CALUDE_rose_group_size_l3692_369252


namespace NUMINAMATH_CALUDE_tv_screen_diagonal_l3692_369267

theorem tv_screen_diagonal (s : ℝ) (h : s^2 = 256 + 34) :
  Real.sqrt (2 * s^2) = Real.sqrt 580 := by
  sorry

end NUMINAMATH_CALUDE_tv_screen_diagonal_l3692_369267


namespace NUMINAMATH_CALUDE_fraction_simplification_l3692_369223

theorem fraction_simplification 
  (d e f : ℝ) 
  (h : d + e + f ≠ 0) : 
  (d^2 + e^2 - f^2 + 2*d*e) / (d^2 + f^2 - e^2 + 3*d*f) = (d + e - f) / (d + f - e) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3692_369223


namespace NUMINAMATH_CALUDE_iphone_average_cost_l3692_369200

/-- Proves that the average cost of an iPhone is $1000 given the sales data --/
theorem iphone_average_cost (iphone_count : Nat) (ipad_count : Nat) (appletv_count : Nat)
  (ipad_cost : ℝ) (appletv_cost : ℝ) (total_average : ℝ)
  (h1 : iphone_count = 100)
  (h2 : ipad_count = 20)
  (h3 : appletv_count = 80)
  (h4 : ipad_cost = 900)
  (h5 : appletv_cost = 200)
  (h6 : total_average = 670) :
  (iphone_count * 1000 + ipad_count * ipad_cost + appletv_count * appletv_cost) /
    (iphone_count + ipad_count + appletv_count : ℝ) = total_average :=
by sorry

end NUMINAMATH_CALUDE_iphone_average_cost_l3692_369200


namespace NUMINAMATH_CALUDE_f_equals_three_l3692_369209

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 1
  else if x < 2 then x^2
  else 2*x

theorem f_equals_three (x : ℝ) : f x = 3 ↔ x = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_f_equals_three_l3692_369209


namespace NUMINAMATH_CALUDE_theater_revenue_l3692_369251

/-- Calculates the total revenue for a theater performance series -/
theorem theater_revenue (seats : ℕ) (capacity : ℚ) (ticket_price : ℕ) (days : ℕ) :
  seats = 400 →
  capacity = 4/5 →
  ticket_price = 30 →
  days = 3 →
  (seats : ℚ) * capacity * (ticket_price : ℚ) * (days : ℚ) = 28800 := by
sorry

end NUMINAMATH_CALUDE_theater_revenue_l3692_369251


namespace NUMINAMATH_CALUDE_number_of_schedules_l3692_369299

/-- Represents the number of periods in a day -/
def total_periods : ℕ := 6

/-- Represents the number of morning periods -/
def morning_periods : ℕ := 3

/-- Represents the number of afternoon periods -/
def afternoon_periods : ℕ := 3

/-- Represents the total number of classes -/
def total_classes : ℕ := 6

/-- Represents the constraint that Mathematics must be in the morning -/
def math_in_morning : Prop := true

/-- Represents the constraint that Art must be in the afternoon -/
def art_in_afternoon : Prop := true

/-- The main theorem stating the number of possible schedules -/
theorem number_of_schedules :
  math_in_morning →
  art_in_afternoon →
  (total_periods = morning_periods + afternoon_periods) →
  (∃ (n : ℕ), n = 216 ∧ n = number_of_possible_schedules) :=
sorry

end NUMINAMATH_CALUDE_number_of_schedules_l3692_369299


namespace NUMINAMATH_CALUDE_hyperbola_perpendicular_product_l3692_369258

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

/-- The asymptotes of the hyperbola -/
def asymptote₁ (x y : ℝ) : Prop := x + Real.sqrt 3 * y = 0
def asymptote₂ (x y : ℝ) : Prop := x - Real.sqrt 3 * y = 0

/-- A point on the hyperbola -/
def P : ℝ × ℝ → Prop := λ p => hyperbola p.1 p.2

/-- Feet of perpendiculars from P to asymptotes -/
def A : (ℝ × ℝ) → (ℝ × ℝ) → Prop := λ p a => 
  asymptote₁ a.1 a.2 ∧ (p.1 - a.1) * (Real.sqrt 3) + (p.2 - a.2) = 0

def B : (ℝ × ℝ) → (ℝ × ℝ) → Prop := λ p b => 
  asymptote₂ b.1 b.2 ∧ (p.1 - b.1) * (Real.sqrt 3) - (p.2 - b.2) = 0

/-- The theorem to be proved -/
theorem hyperbola_perpendicular_product (p a b : ℝ × ℝ) :
  P p → A p a → B p b → 
  (p.1 - a.1) * (p.1 - b.1) + (p.2 - a.2) * (p.2 - b.2) = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_perpendicular_product_l3692_369258


namespace NUMINAMATH_CALUDE_max_backpacks_filled_fifteen_backpacks_possible_max_backpacks_is_fifteen_l3692_369226

def pencils : ℕ := 150
def notebooks : ℕ := 255
def pens : ℕ := 315

theorem max_backpacks_filled (n : ℕ) : 
  (pencils % n = 0 ∧ notebooks % n = 0 ∧ pens % n = 0) →
  n ≤ 15 :=
by
  sorry

theorem fifteen_backpacks_possible : 
  pencils % 15 = 0 ∧ notebooks % 15 = 0 ∧ pens % 15 = 0 :=
by
  sorry

theorem max_backpacks_is_fifteen : 
  ∀ n : ℕ, (pencils % n = 0 ∧ notebooks % n = 0 ∧ pens % n = 0) → n ≤ 15 :=
by
  sorry

end NUMINAMATH_CALUDE_max_backpacks_filled_fifteen_backpacks_possible_max_backpacks_is_fifteen_l3692_369226


namespace NUMINAMATH_CALUDE_sum_relations_l3692_369263

theorem sum_relations (a b c d : ℝ) 
  (hab : a + b = 4)
  (hcd : c + d = 3)
  (had : a + d = 2) :
  b + c = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_relations_l3692_369263


namespace NUMINAMATH_CALUDE_parallelogram_BJ_length_l3692_369225

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram ABCD with additional points H, J, and K -/
structure Parallelogram :=
  (A B C D H J K : Point)

/-- Checks if three points are collinear -/
def collinear (P Q R : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (P Q : Point) : ℝ := sorry

/-- Checks if two line segments are parallel -/
def parallel (P Q R S : Point) : Prop := sorry

theorem parallelogram_BJ_length
  (ABCD : Parallelogram)
  (h1 : collinear ABCD.A ABCD.D ABCD.H)
  (h2 : collinear ABCD.B ABCD.H ABCD.J)
  (h3 : collinear ABCD.B ABCD.H ABCD.K)
  (h4 : collinear ABCD.A ABCD.C ABCD.J)
  (h5 : collinear ABCD.D ABCD.C ABCD.K)
  (h6 : distance ABCD.J ABCD.H = 20)
  (h7 : distance ABCD.K ABCD.H = 30)
  (h8 : distance ABCD.A ABCD.D = 2 * distance ABCD.B ABCD.C)
  (h9 : parallel ABCD.A ABCD.B ABCD.D ABCD.C)
  (h10 : parallel ABCD.A ABCD.D ABCD.B ABCD.C) :
  distance ABCD.B ABCD.J = 5 := by sorry

end NUMINAMATH_CALUDE_parallelogram_BJ_length_l3692_369225


namespace NUMINAMATH_CALUDE_function_second_derivative_at_e_l3692_369224

open Real

theorem function_second_derivative_at_e (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_def : ∀ x, f x = 2 * x * (deriv^[2] f e) + log x) : 
  deriv^[2] f e = -1 / e := by
  sorry

end NUMINAMATH_CALUDE_function_second_derivative_at_e_l3692_369224


namespace NUMINAMATH_CALUDE_seth_boxes_theorem_l3692_369208

/-- The number of boxes Seth bought at the market -/
def market_boxes : ℕ := 3

/-- The number of boxes Seth bought at the farm -/
def farm_boxes : ℕ := 2 * market_boxes

/-- The total number of boxes Seth initially had -/
def initial_boxes : ℕ := market_boxes + farm_boxes

/-- The number of boxes Seth gave to his mother -/
def mother_boxes : ℕ := 1

/-- The number of boxes Seth had after giving to his mother -/
def after_mother_boxes : ℕ := initial_boxes - mother_boxes

/-- The number of boxes Seth donated to charity -/
def charity_boxes : ℕ := after_mother_boxes / 4

/-- The number of boxes Seth had after donating to charity -/
def after_charity_boxes : ℕ := after_mother_boxes - charity_boxes

/-- The number of boxes Seth had left at the end -/
def final_boxes : ℕ := 4

/-- The number of boxes Seth gave to his friends -/
def friend_boxes : ℕ := after_charity_boxes - final_boxes

/-- The total number of boxes Seth bought -/
def total_boxes : ℕ := initial_boxes

theorem seth_boxes_theorem : total_boxes = 14 := by
  sorry

end NUMINAMATH_CALUDE_seth_boxes_theorem_l3692_369208


namespace NUMINAMATH_CALUDE_f_monotonicity_and_zeros_l3692_369282

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 2) * x - Real.log x

theorem f_monotonicity_and_zeros (a : ℝ) :
  (∀ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ → f a x₁ > f a x₂) ∨
  (∃ c, 0 < c ∧ 
    (∀ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ ∧ x₂ < c → f a x₁ > f a x₂) ∧
    (∀ x₁ x₂, c < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) ∧
  (∃ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ 0 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_zeros_l3692_369282


namespace NUMINAMATH_CALUDE_gcd_4830_3289_l3692_369256

theorem gcd_4830_3289 : Nat.gcd 4830 3289 = 23 := by
  sorry

end NUMINAMATH_CALUDE_gcd_4830_3289_l3692_369256


namespace NUMINAMATH_CALUDE_total_age_problem_l3692_369276

theorem total_age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  b = 10 →
  a + b + c = 27 :=
by sorry

end NUMINAMATH_CALUDE_total_age_problem_l3692_369276


namespace NUMINAMATH_CALUDE_journey_distance_l3692_369245

/-- Proves that a journey with given conditions has a total distance of 126 km -/
theorem journey_distance (total_time : ℝ) (speed1 speed2 speed3 : ℝ) :
  total_time = 12 ∧
  speed1 = 21 ∧
  speed2 = 14 ∧
  speed3 = 6 →
  (1 / speed1 + 1 / speed2 + 1 / speed3) * (total_time / 3) = 126 := by
  sorry


end NUMINAMATH_CALUDE_journey_distance_l3692_369245


namespace NUMINAMATH_CALUDE_nils_geese_count_l3692_369269

/-- Represents the number of geese on Nils' farm -/
def n : ℕ := sorry

/-- Represents the number of days the feed lasts initially -/
def k : ℕ := sorry

/-- The amount of feed consumed by one goose per day -/
def x : ℝ := sorry

/-- The total amount of feed available -/
def A : ℝ := sorry

/-- The feed lasts k days for n geese -/
axiom initial_feed : A = k * x * n

/-- The feed lasts (k + 20) days for (n - 75) geese -/
axiom sell_75_geese : A = (k + 20) * x * (n - 75)

/-- The feed lasts (k - 15) days for (n + 100) geese -/
axiom buy_100_geese : A = (k - 15) * x * (n + 100)

theorem nils_geese_count : n = 300 := by sorry

end NUMINAMATH_CALUDE_nils_geese_count_l3692_369269


namespace NUMINAMATH_CALUDE_jen_candy_profit_l3692_369280

/-- Calculates the profit from selling candy bars --/
def candy_profit (buy_price sell_price : ℕ) (bought sold : ℕ) : ℕ :=
  (sell_price - buy_price) * sold

/-- Proves that Jen's profit from selling candy bars is 960 cents --/
theorem jen_candy_profit : candy_profit 80 100 50 48 = 960 := by
  sorry

end NUMINAMATH_CALUDE_jen_candy_profit_l3692_369280


namespace NUMINAMATH_CALUDE_smallest_integer_greater_than_neg_seven_thirds_l3692_369232

theorem smallest_integer_greater_than_neg_seven_thirds :
  Int.ceil (-7/3 : ℚ) = -2 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_greater_than_neg_seven_thirds_l3692_369232


namespace NUMINAMATH_CALUDE_diagonal_passes_through_840_cubes_l3692_369284

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_through (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: An internal diagonal of a 200 × 360 × 450 rectangular solid passes through 840 cubes -/
theorem diagonal_passes_through_840_cubes :
  cubes_passed_through 200 360 450 = 840 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_passes_through_840_cubes_l3692_369284


namespace NUMINAMATH_CALUDE_sin_240_degrees_l3692_369265

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l3692_369265


namespace NUMINAMATH_CALUDE_lcm_problem_l3692_369247

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3692_369247


namespace NUMINAMATH_CALUDE_final_price_is_91_percent_l3692_369249

/-- Represents the price increase factor -/
def price_increase : ℝ := 1.4

/-- Represents the discount factor -/
def discount : ℝ := 0.65

/-- Theorem stating that the final price after increase and discount is 91% of the original price -/
theorem final_price_is_91_percent (original_price : ℝ) :
  discount * (price_increase * original_price) = 0.91 * original_price := by
  sorry

end NUMINAMATH_CALUDE_final_price_is_91_percent_l3692_369249


namespace NUMINAMATH_CALUDE_greatest_integer_prime_absolute_value_l3692_369201

theorem greatest_integer_prime_absolute_value : 
  ∃ (x : ℤ), (∀ (y : ℤ), y > x → ¬(Nat.Prime (Int.natAbs (8 * y^2 - 56 * y + 21)))) ∧ 
  (Nat.Prime (Int.natAbs (8 * x^2 - 56 * x + 21))) ∧ 
  x = 4 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_prime_absolute_value_l3692_369201


namespace NUMINAMATH_CALUDE_root_product_theorem_l3692_369295

theorem root_product_theorem (x₁ x₂ x₃ x₄ x₅ : ℂ) : 
  (x₁^5 - 3*x₁^3 + x₁ + 6 = 0) →
  (x₂^5 - 3*x₂^3 + x₂ + 6 = 0) →
  (x₃^5 - 3*x₃^3 + x₃ + 6 = 0) →
  (x₄^5 - 3*x₄^3 + x₄ + 6 = 0) →
  (x₅^5 - 3*x₅^3 + x₅ + 6 = 0) →
  ((x₁^2 - 2) * (x₂^2 - 2) * (x₃^2 - 2) * (x₄^2 - 2) * (x₅^2 - 2) = 10) := by
  sorry

end NUMINAMATH_CALUDE_root_product_theorem_l3692_369295


namespace NUMINAMATH_CALUDE_cell_population_growth_l3692_369246

/-- The number of cells in a population after n hours, given the specified conditions -/
def cell_count (n : ℕ) : ℕ :=
  2^(n-1) + 4

/-- Theorem stating that the cell_count function correctly models the cell population growth -/
theorem cell_population_growth (n : ℕ) :
  let initial_cells := 5
  let cells_lost_per_hour := 2
  let division_factor := 2
  cell_count n = (initial_cells - cells_lost_per_hour) * division_factor^n + cells_lost_per_hour :=
by sorry

end NUMINAMATH_CALUDE_cell_population_growth_l3692_369246


namespace NUMINAMATH_CALUDE_sin_double_sum_eq_four_sin_product_l3692_369236

/-- Given that α + β + γ = π, prove that sin 2α + sin 2β + sin 2γ = 4 sin α sin β sin γ -/
theorem sin_double_sum_eq_four_sin_product (α β γ : Real) (h : α + β + γ = Real.pi) :
  Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ) = 4 * Real.sin α * Real.sin β * Real.sin γ := by
  sorry

end NUMINAMATH_CALUDE_sin_double_sum_eq_four_sin_product_l3692_369236


namespace NUMINAMATH_CALUDE_other_bases_with_square_property_existence_of_other_bases_l3692_369248

theorem other_bases_with_square_property (B : ℕ) (V : ℕ) : Prop :=
  2 < B ∧ 1 < V ∧ V < B ∧ V * V % B = V % B

theorem existence_of_other_bases :
  ∃ B V, B ≠ 50 ∧ other_bases_with_square_property B V := by
  sorry

end NUMINAMATH_CALUDE_other_bases_with_square_property_existence_of_other_bases_l3692_369248


namespace NUMINAMATH_CALUDE_total_profit_is_8640_l3692_369259

/-- Represents the investment and profit distribution of a business partnership --/
structure BusinessPartnership where
  total_investment : ℕ
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  a_profit_share : ℕ

/-- Calculates the total profit based on the given business partnership --/
def calculate_total_profit (bp : BusinessPartnership) : ℕ :=
  let investment_ratio := bp.a_investment + bp.b_investment + bp.c_investment
  let profit_per_ratio := bp.a_profit_share * investment_ratio / bp.a_investment
  profit_per_ratio

/-- Theorem stating the total profit for the given business scenario --/
theorem total_profit_is_8640 (bp : BusinessPartnership) 
  (h1 : bp.total_investment = 90000)
  (h2 : bp.a_investment = bp.b_investment + 6000)
  (h3 : bp.c_investment = bp.b_investment + 3000)
  (h4 : bp.a_investment + bp.b_investment + bp.c_investment = bp.total_investment)
  (h5 : bp.a_profit_share = 3168) :
  calculate_total_profit bp = 8640 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_8640_l3692_369259


namespace NUMINAMATH_CALUDE_train_average_speed_l3692_369272

/-- Proves that the average speed of a train including stoppages is 36 kmph,
    given its speed excluding stoppages and the duration of stoppages. -/
theorem train_average_speed
  (speed_without_stops : ℝ)
  (stop_duration : ℝ)
  (h1 : speed_without_stops = 60)
  (h2 : stop_duration = 24)
  : (speed_without_stops * (60 - stop_duration) / 60) = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_average_speed_l3692_369272


namespace NUMINAMATH_CALUDE_cards_distribution_l3692_369239

theorem cards_distribution (total_cards : ℕ) (num_people : ℕ) 
  (h1 : total_cards = 52) (h2 : num_people = 9) :
  let base_cards := total_cards / num_people
  let remainder := total_cards % num_people
  let people_with_extra := remainder
  let people_with_base := num_people - people_with_extra
  people_with_base = 2 ∧ base_cards + 1 < 7 := by sorry

end NUMINAMATH_CALUDE_cards_distribution_l3692_369239


namespace NUMINAMATH_CALUDE_book_prices_l3692_369257

theorem book_prices (book1_discounted : ℚ) (book2_discounted : ℚ)
  (h1 : book1_discounted = 8)
  (h2 : book2_discounted = 9)
  (h3 : book1_discounted = (1 / 8 : ℚ) * book1_discounted / (1 / 8 : ℚ))
  (h4 : book2_discounted = (1 / 9 : ℚ) * book2_discounted / (1 / 9 : ℚ)) :
  book1_discounted / (1 / 8 : ℚ) + book2_discounted / (1 / 9 : ℚ) = 145 := by
sorry

end NUMINAMATH_CALUDE_book_prices_l3692_369257


namespace NUMINAMATH_CALUDE_unpainted_side_length_approx_l3692_369262

/-- A rectangular parking space with three painted sides -/
structure ParkingSpace where
  length : ℝ
  width : ℝ
  painted_sides_sum : length + 2 * width = 37
  area : length * width = 125

/-- The length of the unpainted side of the parking space -/
def unpainted_side_length (p : ParkingSpace) : ℝ := p.length

/-- The unpainted side length is approximately 8.90 feet -/
theorem unpainted_side_length_approx (p : ParkingSpace) :
  ∃ ε > 0, |unpainted_side_length p - 8.90| < ε :=
sorry

end NUMINAMATH_CALUDE_unpainted_side_length_approx_l3692_369262


namespace NUMINAMATH_CALUDE_last_score_is_71_l3692_369292

def scores : List Nat := [71, 74, 79, 85, 88, 92]

def is_valid_last_score (last_score : Nat) : Prop :=
  last_score ∈ scores ∧
  ∀ n : Nat, 1 ≤ n ∧ n ≤ 6 → 
    (scores.sum - last_score) % n = 0

theorem last_score_is_71 : 
  ∃! last_score, is_valid_last_score last_score ∧ last_score = 71 := by sorry

end NUMINAMATH_CALUDE_last_score_is_71_l3692_369292


namespace NUMINAMATH_CALUDE_f_properties_l3692_369289

noncomputable def f (a b c x : ℝ) : ℝ := -x^3 + a*x^2 + b*x + c

theorem f_properties (a b c : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ 0 → f a b c x₁ > f a b c x₂) →
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → f a b c x₁ < f a b c x₂) →
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ f a b c x₁ = 0 ∧ f a b c x₂ = 0 ∧ f a b c x₃ = 0) →
  f a b c 1 = 0 →
  b = 0 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3692_369289


namespace NUMINAMATH_CALUDE_layla_phone_probability_l3692_369277

def first_segment_choices : ℕ := 3
def last_segment_digits : ℕ := 4

theorem layla_phone_probability :
  (1 : ℚ) / (first_segment_choices * Nat.factorial last_segment_digits) = 1 / 72 :=
by sorry

end NUMINAMATH_CALUDE_layla_phone_probability_l3692_369277


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3692_369242

theorem inequality_and_equality_condition (p q : ℝ) (hp : 0 < p) (hq : p < q)
  (α β γ δ ε : ℝ) (hα : p ≤ α ∧ α ≤ q) (hβ : p ≤ β ∧ β ≤ q)
  (hγ : p ≤ γ ∧ γ ≤ q) (hδ : p ≤ δ ∧ δ ≤ q) (hε : p ≤ ε ∧ ε ≤ q) :
  (α + β + γ + δ + ε) * (1/α + 1/β + 1/γ + 1/δ + 1/ε) ≤ 25 + 6 * (Real.sqrt (p/q) - Real.sqrt (q/p))^2 ∧
  ((α + β + γ + δ + ε) * (1/α + 1/β + 1/γ + 1/δ + 1/ε) = 25 + 6 * (Real.sqrt (p/q) - Real.sqrt (q/p))^2 ↔
   ((α = p ∧ β = p ∧ γ = q ∧ δ = q ∧ ε = q) ∨
    (α = p ∧ β = q ∧ γ = p ∧ δ = q ∧ ε = q) ∨
    (α = p ∧ β = q ∧ γ = q ∧ δ = p ∧ ε = q) ∨
    (α = p ∧ β = q ∧ γ = q ∧ δ = q ∧ ε = p) ∨
    (α = q ∧ β = p ∧ γ = p ∧ δ = q ∧ ε = q) ∨
    (α = q ∧ β = p ∧ γ = q ∧ δ = p ∧ ε = q) ∨
    (α = q ∧ β = p ∧ γ = q ∧ δ = q ∧ ε = p) ∨
    (α = q ∧ β = q ∧ γ = p ∧ δ = p ∧ ε = q) ∨
    (α = q ∧ β = q ∧ γ = p ∧ δ = q ∧ ε = p) ∨
    (α = q ∧ β = q ∧ γ = q ∧ δ = p ∧ ε = p))) :=
sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3692_369242


namespace NUMINAMATH_CALUDE_equation_holds_for_all_y_l3692_369297

theorem equation_holds_for_all_y (x : ℝ) : 
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_holds_for_all_y_l3692_369297


namespace NUMINAMATH_CALUDE_apartment_rent_calculation_required_rent_is_correct_l3692_369283

/-- Calculate the required monthly rent for an apartment investment --/
theorem apartment_rent_calculation (investment : ℝ) (maintenance_rate : ℝ) 
  (annual_taxes : ℝ) (desired_return_rate : ℝ) : ℝ :=
  let annual_return := investment * desired_return_rate
  let total_annual_requirement := annual_return + annual_taxes
  let monthly_net_requirement := total_annual_requirement / 12
  let monthly_rent := monthly_net_requirement / (1 - maintenance_rate)
  monthly_rent

/-- The required monthly rent is approximately $153.70 --/
theorem required_rent_is_correct : 
  ∃ ε > 0, |apartment_rent_calculation 20000 0.1 460 0.06 - 153.70| < ε :=
sorry

end NUMINAMATH_CALUDE_apartment_rent_calculation_required_rent_is_correct_l3692_369283


namespace NUMINAMATH_CALUDE_square_area_from_rectangle_l3692_369229

theorem square_area_from_rectangle (r l b : ℝ) : 
  l = r / 4 →  -- length of rectangle is 1/4 of circle radius
  l * b = 35 → -- area of rectangle is 35
  b = 5 →      -- breadth of rectangle is 5
  r^2 = 784 := by sorry

end NUMINAMATH_CALUDE_square_area_from_rectangle_l3692_369229


namespace NUMINAMATH_CALUDE_calculation_proof_l3692_369275

theorem calculation_proof :
  (((3 * Real.sqrt 48) - (2 * Real.sqrt 27)) / Real.sqrt 3 = 6) ∧
  ((Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) - Real.sqrt ((-3)^2) + (1 / (2 - Real.sqrt 5)) = -3 - Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l3692_369275


namespace NUMINAMATH_CALUDE_circle_area_ratio_l3692_369234

/-- Given a triangle with sides 13, 14, and 15, the ratio of the area of its 
    circumscribed circle to the area of its inscribed circle is (65/32)^2 -/
theorem circle_area_ratio (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) :
  let p := (a + b + c) / 2
  let s := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let r := s / p
  let R := (a * b * c) / (4 * s)
  (R / r) ^ 2 = (65 / 32) ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_circle_area_ratio_l3692_369234


namespace NUMINAMATH_CALUDE_average_speed_theorem_l3692_369286

theorem average_speed_theorem (total_distance : ℝ) (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) 
  (h1 : total_distance = 80)
  (h2 : distance1 = 30)
  (h3 : speed1 = 30)
  (h4 : distance2 = 50)
  (h5 : speed2 = 50)
  (h6 : total_distance = distance1 + distance2) :
  (total_distance) / ((distance1 / speed1) + (distance2 / speed2)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_theorem_l3692_369286


namespace NUMINAMATH_CALUDE_simplify_expression_l3692_369278

theorem simplify_expression (x : ℝ) : (3 * x)^3 + (2 * x) * (x^4) = 27 * x^3 + 2 * x^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3692_369278


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3692_369221

theorem least_subtraction_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (k : Nat), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : Nat), m < k → (n - m) % d ≠ 0 := by
  sorry

theorem problem_solution : 
  let n := 568219
  let d := 89
  ∃ (k : Nat), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : Nat), m < k → (n - m) % d ≠ 0 ∧ k = 45 := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3692_369221


namespace NUMINAMATH_CALUDE_investment_return_calculation_l3692_369237

theorem investment_return_calculation (total_investment small_investment large_investment : ℝ)
  (combined_return_rate small_return_rate : ℝ) :
  total_investment = small_investment + large_investment →
  small_investment = 500 →
  large_investment = 1500 →
  combined_return_rate = 0.085 →
  small_return_rate = 0.07 →
  (small_return_rate * small_investment + 
   (combined_return_rate * total_investment - small_return_rate * small_investment) / large_investment)
  = 0.09 := by
sorry

end NUMINAMATH_CALUDE_investment_return_calculation_l3692_369237


namespace NUMINAMATH_CALUDE_halfway_between_fractions_average_of_fractions_l3692_369293

theorem halfway_between_fractions : 
  (2 : ℚ) / 7 + (4 : ℚ) / 9 = (46 : ℚ) / 63 :=
by sorry

theorem average_of_fractions : 
  ((2 : ℚ) / 7 + (4 : ℚ) / 9) / 2 = (23 : ℚ) / 63 :=
by sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_average_of_fractions_l3692_369293


namespace NUMINAMATH_CALUDE_smallest_base_for_100_l3692_369240

theorem smallest_base_for_100 : 
  ∃ b : ℕ, (b ≥ 5 ∧ b^2 ≤ 100 ∧ 100 < b^3) ∧ 
  (∀ c : ℕ, c < b → (c^2 > 100 ∨ 100 ≥ c^3)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_for_100_l3692_369240


namespace NUMINAMATH_CALUDE_average_weight_of_class_l3692_369230

theorem average_weight_of_class (num_male num_female : ℕ) 
                                (avg_weight_male avg_weight_female : ℚ) :
  num_male = 20 →
  num_female = 20 →
  avg_weight_male = 42 →
  avg_weight_female = 38 →
  (num_male * avg_weight_male + num_female * avg_weight_female) / (num_male + num_female) = 40 := by
sorry

end NUMINAMATH_CALUDE_average_weight_of_class_l3692_369230


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_height_ratio_l3692_369222

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- The height corresponding to the hypotenuse -/
  m : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- m is positive -/
  m_pos : 0 < m
  /-- r is positive -/
  r_pos : 0 < r

/-- The ratio of the inscribed circle radius to the height is between 0.4 and 0.5 -/
theorem inscribed_circle_radius_height_ratio 
  (t : RightTriangleWithInscribedCircle) : 0.4 < t.r / t.m ∧ t.r / t.m < 0.5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_height_ratio_l3692_369222


namespace NUMINAMATH_CALUDE_expression_value_l3692_369268

theorem expression_value (x y : ℝ) (h : (x + 2)^2 + |y - 1/2| = 0) :
  (x - 2*y) * (x + 2*y) - (x - 2*y)^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3692_369268


namespace NUMINAMATH_CALUDE_profit_and_marginal_profit_max_not_equal_l3692_369210

def marginal_function (f : ℕ → ℝ) : ℕ → ℝ := λ x => f (x + 1) - f x

def revenue (a : ℝ) : ℕ → ℝ := λ x => 3000 * x + a * x^2

def cost (k : ℝ) : ℕ → ℝ := λ x => k * x + 4000

def profit (a k : ℝ) : ℕ → ℝ := λ x => revenue a x - cost k x

def marginal_profit (a k : ℝ) : ℕ → ℝ := marginal_function (profit a k)

theorem profit_and_marginal_profit_max_not_equal :
  ∃ (a k : ℝ),
    (∀ x : ℕ, 0 < x ∧ x ≤ 100 → profit a k x ≤ 74120) ∧
    (∃ x : ℕ, 0 < x ∧ x ≤ 100 ∧ profit a k x = 74120) ∧
    (∀ x : ℕ, 0 < x ∧ x ≤ 100 → marginal_profit a k x ≤ 2440) ∧
    (∃ x : ℕ, 0 < x ∧ x ≤ 100 ∧ marginal_profit a k x = 2440) ∧
    (cost k 10 = 9000) ∧
    (profit a k 10 = 19000) ∧
    74120 ≠ 2440 :=
by
  sorry

end NUMINAMATH_CALUDE_profit_and_marginal_profit_max_not_equal_l3692_369210


namespace NUMINAMATH_CALUDE_inequality_max_value_inequality_range_l3692_369212

theorem inequality_max_value (x y : ℝ) (hx : x ∈ Set.Icc 1 2) (hy : y ∈ Set.Icc 1 3) :
  (2 * x^2 + y^2) / (x * y) ≤ 2 * Real.sqrt 2 :=
by sorry

theorem inequality_range (a : ℝ) :
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 3 → 2 * x^2 - a * x * y + y^2 ≥ 0) →
  a ≤ 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_max_value_inequality_range_l3692_369212
