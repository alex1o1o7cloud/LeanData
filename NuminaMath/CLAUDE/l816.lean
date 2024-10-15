import Mathlib

namespace NUMINAMATH_CALUDE_travel_methods_count_l816_81647

/-- The number of transportation options from Shijiazhuang to Qingdao -/
def shijiazhuang_to_qingdao : Nat := 3

/-- The number of transportation options from Qingdao to Guangzhou -/
def qingdao_to_guangzhou : Nat := 4

/-- The total number of travel methods for the entire journey -/
def total_travel_methods : Nat := shijiazhuang_to_qingdao * qingdao_to_guangzhou

theorem travel_methods_count : total_travel_methods = 12 := by
  sorry

end NUMINAMATH_CALUDE_travel_methods_count_l816_81647


namespace NUMINAMATH_CALUDE_age_ratio_solution_l816_81639

/-- Represents the age ratio problem of Mandy and her siblings -/
def age_ratio_problem (mandy_age brother_age sister_age : ℚ) : Prop :=
  mandy_age = 3 ∧
  sister_age = brother_age - 5 ∧
  mandy_age - sister_age = 4 ∧
  brother_age / mandy_age = 4 / 3

/-- Theorem stating that there exists a unique solution to the age ratio problem -/
theorem age_ratio_solution :
  ∃! (mandy_age brother_age sister_age : ℚ),
    age_ratio_problem mandy_age brother_age sister_age :=
by
  sorry

#check age_ratio_solution

end NUMINAMATH_CALUDE_age_ratio_solution_l816_81639


namespace NUMINAMATH_CALUDE_sqrt_36_div_6_l816_81648

theorem sqrt_36_div_6 : Real.sqrt 36 / 6 = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_36_div_6_l816_81648


namespace NUMINAMATH_CALUDE_complex_multiplication_l816_81663

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (2 + 3 * i) = -3 + 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l816_81663


namespace NUMINAMATH_CALUDE_seeds_in_fourth_pot_l816_81697

/-- Given 10 total seeds, 4 pots, and 3 seeds per pot for the first 3 pots,
    prove that the number of seeds in the fourth pot is 1. -/
theorem seeds_in_fourth_pot
  (total_seeds : ℕ)
  (num_pots : ℕ)
  (seeds_per_pot : ℕ)
  (h1 : total_seeds = 10)
  (h2 : num_pots = 4)
  (h3 : seeds_per_pot = 3)
  : total_seeds - (seeds_per_pot * (num_pots - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_seeds_in_fourth_pot_l816_81697


namespace NUMINAMATH_CALUDE_milk_ratio_is_two_fifths_l816_81692

/-- The number of milk boxes Lolita drinks on weekdays -/
def weekday_boxes : ℕ := 3

/-- The number of milk boxes Lolita drinks on Sundays -/
def sunday_boxes : ℕ := 3 * weekday_boxes

/-- The total number of milk boxes Lolita drinks per week -/
def total_boxes : ℕ := 30

/-- The number of milk boxes Lolita drinks on Saturdays -/
def saturday_boxes : ℕ := total_boxes - (5 * weekday_boxes + sunday_boxes)

/-- The ratio of milk boxes on Saturdays to weekdays -/
def milk_ratio : ℚ := saturday_boxes / (5 * weekday_boxes)

theorem milk_ratio_is_two_fifths : milk_ratio = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_milk_ratio_is_two_fifths_l816_81692


namespace NUMINAMATH_CALUDE_max_third_term_l816_81642

/-- An arithmetic sequence of four positive integers with sum 50 -/
structure ArithSequence :=
  (a : ℕ+) -- First term
  (d : ℕ+) -- Common difference
  (sum_eq_50 : a + (a + d) + (a + 2*d) + (a + 3*d) = 50)

/-- The third term of an arithmetic sequence -/
def third_term (seq : ArithSequence) : ℕ := seq.a + 2*seq.d

/-- Theorem: The maximum possible value of the third term is 16 -/
theorem max_third_term :
  ∀ seq : ArithSequence, third_term seq ≤ 16 ∧ ∃ seq : ArithSequence, third_term seq = 16 :=
sorry

end NUMINAMATH_CALUDE_max_third_term_l816_81642


namespace NUMINAMATH_CALUDE_dark_tiles_three_fourths_l816_81627

/-- Represents a tiled floor with a repeating pattern -/
structure TiledFloor :=
  (pattern_size : Nat)
  (corner_dark_tiles : Nat)
  (corner_size : Nat)

/-- The fraction of dark tiles in the entire floor -/
def dark_tile_fraction (floor : TiledFloor) : Rat :=
  floor.corner_dark_tiles / (floor.corner_size * floor.corner_size)

/-- Theorem stating that for a floor with a 4x4 repeating pattern and 3 dark tiles
    in a 2x2 corner section, 3/4 of the entire floor is made of darker tiles -/
theorem dark_tiles_three_fourths (floor : TiledFloor)
  (h1 : floor.pattern_size = 4)
  (h2 : floor.corner_size = 2)
  (h3 : floor.corner_dark_tiles = 3) :
  dark_tile_fraction floor = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_dark_tiles_three_fourths_l816_81627


namespace NUMINAMATH_CALUDE_cost_of_one_each_l816_81616

theorem cost_of_one_each (x y z : ℝ) 
  (eq1 : 3 * x + 7 * y + z = 325)
  (eq2 : 4 * x + 10 * y + z = 410) :
  x + y + z = 155 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_one_each_l816_81616


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l816_81687

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h_q_pos : q > 0)
  (h_condition : a 5 * a 7 = 4 * (a 4)^2)
  (h_a2 : a 2 = 1) :
  a 1 = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l816_81687


namespace NUMINAMATH_CALUDE_cos_equality_problem_l816_81668

theorem cos_equality_problem (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → (Real.cos (n * π / 180) = Real.cos (832 * π / 180) ↔ n = 112) := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_problem_l816_81668


namespace NUMINAMATH_CALUDE_fraction_unchanged_when_multiplied_by_two_l816_81673

theorem fraction_unchanged_when_multiplied_by_two (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x / (x + y) = (2 * x) / (2 * (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_unchanged_when_multiplied_by_two_l816_81673


namespace NUMINAMATH_CALUDE_joe_remaining_money_l816_81649

theorem joe_remaining_money (pocket_money : ℚ) (chocolate_fraction : ℚ) (fruit_fraction : ℚ) :
  pocket_money = 450 ∧
  chocolate_fraction = 1/9 ∧
  fruit_fraction = 2/5 →
  pocket_money - (chocolate_fraction * pocket_money + fruit_fraction * pocket_money) = 220 :=
by sorry

end NUMINAMATH_CALUDE_joe_remaining_money_l816_81649


namespace NUMINAMATH_CALUDE_consecutive_cube_product_divisible_l816_81609

theorem consecutive_cube_product_divisible (a : ℤ) : 
  504 ∣ ((a^3 - 1) * a^3 * (a^3 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_cube_product_divisible_l816_81609


namespace NUMINAMATH_CALUDE_special_haircut_price_l816_81666

/-- Represents the cost of different types of haircuts and the hairstylist's earnings --/
structure HaircutPrices where
  normal : ℝ
  special : ℝ
  trendy : ℝ
  daily_normal : ℕ
  daily_special : ℕ
  daily_trendy : ℕ
  weekly_earnings : ℝ
  days_per_week : ℕ

/-- Theorem stating that the special haircut price is $6 given the conditions --/
theorem special_haircut_price (h : HaircutPrices) 
    (h_normal : h.normal = 5)
    (h_trendy : h.trendy = 8)
    (h_daily_normal : h.daily_normal = 5)
    (h_daily_special : h.daily_special = 3)
    (h_daily_trendy : h.daily_trendy = 2)
    (h_weekly_earnings : h.weekly_earnings = 413)
    (h_days_per_week : h.days_per_week = 7) :
  h.special = 6 := by
  sorry

#check special_haircut_price

end NUMINAMATH_CALUDE_special_haircut_price_l816_81666


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l816_81641

theorem arithmetic_sequence_proof :
  ∀ (a : ℕ → ℤ),
    (∀ i j : ℕ, a (i + 1) - a i = a (j + 1) - a j) →  -- arithmetic sequence condition
    (a 0 = 3^2) →  -- first term is 3²
    (a 2 = 3^4) →  -- third term is 3⁴
    (a 1 = 33 ∧ a 3 = 105) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l816_81641


namespace NUMINAMATH_CALUDE_c_share_is_75_l816_81651

-- Define the total payment
def total_payment : ℚ := 600

-- Define the time taken by each worker individually
def a_time : ℚ := 6
def b_time : ℚ := 8

-- Define the time taken by all three workers together
def abc_time : ℚ := 3

-- Define the shares of A and B
def a_share : ℚ := 300
def b_share : ℚ := 225

-- Define C's share as a function of the given parameters
def c_share (total : ℚ) (a_t b_t abc_t : ℚ) (a_s b_s : ℚ) : ℚ :=
  total - (a_s + b_s)

-- Theorem statement
theorem c_share_is_75 :
  c_share total_payment a_time b_time abc_time a_share b_share = 75 := by
  sorry


end NUMINAMATH_CALUDE_c_share_is_75_l816_81651


namespace NUMINAMATH_CALUDE_sine_equality_solution_l816_81629

theorem sine_equality_solution (m : ℤ) : 
  -180 ≤ m ∧ m ≤ 180 ∧ Real.sin (m * π / 180) = Real.sin (780 * π / 180) → 
  m = 60 ∨ m = 120 := by
  sorry

end NUMINAMATH_CALUDE_sine_equality_solution_l816_81629


namespace NUMINAMATH_CALUDE_triangle_property_and_function_value_l816_81660

theorem triangle_property_and_function_value (a b c A : ℝ) :
  0 < A ∧ A < π →
  b^2 + c^2 = a^2 + Real.sqrt 3 * b * c →
  let m : ℝ × ℝ := (Real.sin A, Real.cos A)
  let n : ℝ × ℝ := (Real.cos A, Real.sqrt 3 * Real.cos A)
  let f : ℝ → ℝ := fun x => m.1 * n.1 + m.2 * n.2 - Real.sqrt 3 / 2
  f A = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_property_and_function_value_l816_81660


namespace NUMINAMATH_CALUDE_number_problem_l816_81622

theorem number_problem (x : ℝ) : (0.7 * x - 40 = 30) → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l816_81622


namespace NUMINAMATH_CALUDE_series_solution_l816_81637

/-- The sum of the infinite series 1 + 3x + 6x^2 + ... -/
noncomputable def S (x : ℝ) : ℝ := 1 / (1 - x)^3

/-- Theorem: If S(x) = 4, then x = 1 - 1/∛4 -/
theorem series_solution (x : ℝ) (h : S x = 4) : x = 1 - 1 / Real.rpow 4 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_series_solution_l816_81637


namespace NUMINAMATH_CALUDE_functional_equation_solution_l816_81611

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f y + f (x + y) = x * y) →
  (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = -x - 1) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l816_81611


namespace NUMINAMATH_CALUDE_exists_composite_power_sum_l816_81632

theorem exists_composite_power_sum (x y : ℕ) (hx : 2 ≤ x ∧ x ≤ 100) (hy : 2 ≤ y ∧ y ≤ 100) :
  ∃ n : ℕ, ∃ k : ℕ, k > 1 ∧ k ∣ (x^(2^n) + y^(2^n)) :=
by sorry

end NUMINAMATH_CALUDE_exists_composite_power_sum_l816_81632


namespace NUMINAMATH_CALUDE_max_non_functional_segments_is_13_l816_81690

/-- Represents a seven-segment display --/
structure SevenSegmentDisplay :=
  (segments : Fin 7 → Bool)

/-- Represents a four-digit clock display --/
structure ClockDisplay :=
  (digits : Fin 4 → SevenSegmentDisplay)

/-- The set of valid digits for each position --/
def validDigits : Fin 4 → Set ℕ
  | 0 => {0, 1, 2}
  | 1 => {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  | 2 => {0, 1, 2, 3, 4, 5}
  | 3 => {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- A function that determines if a time can be unambiguously read --/
def isUnambiguous (display : ClockDisplay) : Prop := sorry

/-- The maximum number of non-functional segments --/
def maxNonFunctionalSegments : ℕ := 13

/-- The main theorem --/
theorem max_non_functional_segments_is_13 :
  ∀ (display : ClockDisplay),
    (∀ (i : Fin 4), ∃ (d : ℕ), d ∈ validDigits i ∧ isUnambiguous display) →
    (∃ (n : ℕ), n = maxNonFunctionalSegments ∧
      ∀ (m : ℕ), m > n →
        ¬(∀ (i : Fin 4), ∃ (d : ℕ), d ∈ validDigits i ∧ isUnambiguous display)) :=
by sorry

end NUMINAMATH_CALUDE_max_non_functional_segments_is_13_l816_81690


namespace NUMINAMATH_CALUDE_tartar_arrangements_l816_81610

/-- The number of unique arrangements of letters in a word -/
def uniqueArrangements (totalLetters : ℕ) (duplicateSets : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (duplicateSets.map Nat.factorial).prod

/-- The word TARTAR has 6 letters with T, A, and R each appearing twice -/
theorem tartar_arrangements :
  uniqueArrangements 6 [2, 2, 2] = 90 := by
  sorry

end NUMINAMATH_CALUDE_tartar_arrangements_l816_81610


namespace NUMINAMATH_CALUDE_parabola_translation_l816_81653

-- Define the original parabola function
def original_parabola (x : ℝ) : ℝ := -x^2

-- Define the translated parabola function
def translated_parabola (x : ℝ) : ℝ := -(x + 2)^2 + 3

-- Theorem statement
theorem parabola_translation :
  ∀ x : ℝ, translated_parabola x = original_parabola (x + 2) + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l816_81653


namespace NUMINAMATH_CALUDE_third_grade_swim_caps_l816_81604

theorem third_grade_swim_caps (b r : ℕ) : 
  b = 4 * r + 2 →
  b = r + 24 →
  b + r = 37 :=
by sorry

end NUMINAMATH_CALUDE_third_grade_swim_caps_l816_81604


namespace NUMINAMATH_CALUDE_cubic_equation_roots_relation_l816_81601

theorem cubic_equation_roots_relation (a b c : ℝ) (s₁ s₂ s₃ : ℂ) :
  (s₁^3 + a*s₁^2 + b*s₁ + c = 0) →
  (s₂^3 + a*s₂^2 + b*s₂ + c = 0) →
  (s₃^3 + a*s₃^2 + b*s₃ + c = 0) →
  (∃ p q r : ℝ, (s₁^2)^3 + p*(s₁^2)^2 + q*(s₁^2) + r = 0 ∧
               (s₂^2)^3 + p*(s₂^2)^2 + q*(s₂^2) + r = 0 ∧
               (s₃^2)^3 + p*(s₃^2)^2 + q*(s₃^2) + r = 0) →
  (∃ p q r : ℝ, p = a^2 - 2*b ∧ q = b^2 + 2*a*c ∧ r = c^2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_relation_l816_81601


namespace NUMINAMATH_CALUDE_total_broken_bulbs_to_replace_l816_81640

/-- Represents the number of broken light bulbs that need to be replaced -/
def broken_bulbs_to_replace (kitchen_bulbs foyer_broken_bulbs living_room_bulbs : ℕ) : ℕ :=
  let kitchen_broken := (3 * kitchen_bulbs) / 5
  let foyer_broken := foyer_broken_bulbs
  let living_room_broken := living_room_bulbs / 2
  kitchen_broken + foyer_broken + living_room_broken

/-- Theorem stating the total number of broken light bulbs to be replaced -/
theorem total_broken_bulbs_to_replace :
  broken_bulbs_to_replace 35 10 24 = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_total_broken_bulbs_to_replace_l816_81640


namespace NUMINAMATH_CALUDE_linear_independence_preservation_l816_81628

variable {n : ℕ}
variable (v : Fin (n - 1) → (Fin n → ℝ))

/-- P_{i,k} sets the i-th component of a vector to zero -/
def P (i k : ℕ) (x : Fin k → ℝ) : Fin k → ℝ :=
  λ j => if j = i then 0 else x j

theorem linear_independence_preservation (hn : n ≥ 2) 
  (hv : LinearIndependent ℝ v) :
  ∃ k : Fin n, LinearIndependent ℝ (λ i => P k n (v i)) := by
  sorry

end NUMINAMATH_CALUDE_linear_independence_preservation_l816_81628


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l816_81620

theorem absolute_value_inequality (x : ℝ) :
  |((3 * x + 2) / (x + 1))| > 3 ↔ x < -1 ∨ (-5/6 < x ∧ x < -1) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l816_81620


namespace NUMINAMATH_CALUDE_train_speed_difference_l816_81696

/-- Given two trains traveling towards each other, this theorem proves that
    the difference in their speeds is 30 km/hr under specific conditions. -/
theorem train_speed_difference 
  (distance : ℝ) 
  (meeting_time : ℝ) 
  (express_speed : ℝ) 
  (h1 : distance = 390) 
  (h2 : meeting_time = 3) 
  (h3 : express_speed = 80) : 
  express_speed - (distance / meeting_time - express_speed) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_difference_l816_81696


namespace NUMINAMATH_CALUDE_sector_area_l816_81693

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (area : ℝ) : 
  perimeter = 16 → central_angle = 2 → area = 16 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l816_81693


namespace NUMINAMATH_CALUDE_book_pages_calculation_l816_81624

theorem book_pages_calculation (pages_read : ℕ) (pages_unread : ℕ) (additional_pages : ℕ) :
  pages_read + pages_unread > 0 →
  pages_read = pages_unread / 3 →
  additional_pages = 48 →
  (pages_read + additional_pages : ℚ) / (pages_read + pages_unread + additional_pages) = 2/5 →
  pages_read + pages_unread = 320 := by
sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l816_81624


namespace NUMINAMATH_CALUDE_paco_cookies_eaten_l816_81671

/-- Represents the number of cookies Paco ate -/
structure CookiesEaten where
  sweet : ℕ
  salty : ℕ

/-- Proves that if Paco ate 20 sweet cookies and 14 more salty cookies than sweet cookies,
    then he ate 34 salty cookies. -/
theorem paco_cookies_eaten (cookies : CookiesEaten) 
  (h1 : cookies.sweet = 20) 
  (h2 : cookies.salty = cookies.sweet + 14) : 
  cookies.salty = 34 := by
  sorry

#check paco_cookies_eaten

end NUMINAMATH_CALUDE_paco_cookies_eaten_l816_81671


namespace NUMINAMATH_CALUDE_equation_holds_iff_specific_values_l816_81650

/-- The equation holds for all real x if and only if a, b, p, and q have specific values -/
theorem equation_holds_iff_specific_values :
  ∀ (a b p q : ℝ),
    (∀ x : ℝ, (2*x - 1)^20 - (a*x + b)^20 = (x^2 + p*x + q)^10) ↔
    (a = (2^20 - 1)^(1/20) ∧
     b = -(2^20 - 1)^(1/20) / 2 ∧
     p = -1 ∧
     q = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_equation_holds_iff_specific_values_l816_81650


namespace NUMINAMATH_CALUDE_pizza_special_pricing_l816_81617

/-- Represents the cost calculation for pizzas with special pricing --/
def pizza_cost (standard_price : ℕ) (triple_cheese_count : ℕ) (meat_lovers_count : ℕ) : ℕ :=
  let triple_cheese_cost := (triple_cheese_count / 2) * standard_price
  let meat_lovers_cost := ((meat_lovers_count + 2) / 3 * 2) * standard_price
  triple_cheese_cost + meat_lovers_cost

/-- Theorem stating the total cost of pizzas under special pricing --/
theorem pizza_special_pricing :
  pizza_cost 5 10 9 = 55 := by
  sorry


end NUMINAMATH_CALUDE_pizza_special_pricing_l816_81617


namespace NUMINAMATH_CALUDE_right_triangle_area_l816_81643

/-- The area of a right triangle given the sum of its legs and the altitude from the right angle. -/
theorem right_triangle_area (l h : ℝ) (hl : l > 0) (hh : h > 0) :
  ∃ S : ℝ, S = (1/2) * h * (Real.sqrt (l^2 + h^2) - h) ∧ 
  S > 0 ∧ 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = l ∧ 
  S = (1/2) * x * h ∧ S = (1/2) * y * h :=
by sorry


end NUMINAMATH_CALUDE_right_triangle_area_l816_81643


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l816_81674

/-- Given an incident ray y = 2x + 1 reflected by the line y = x, 
    the equation of the reflected ray is x - 2y - 1 = 0 -/
theorem reflected_ray_equation (x y : ℝ) : 
  (y = 2*x + 1) →  -- incident ray
  (y = x) →        -- reflecting line
  (x - 2*y - 1 = 0) -- reflected ray
  := by sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l816_81674


namespace NUMINAMATH_CALUDE_max_abs_z_given_distance_from_2i_l816_81614

theorem max_abs_z_given_distance_from_2i (z : ℂ) : 
  Complex.abs (z - 2 * Complex.I) = 1 → Complex.abs z ≤ 3 ∧ ∃ w : ℂ, Complex.abs (w - 2 * Complex.I) = 1 ∧ Complex.abs w = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_z_given_distance_from_2i_l816_81614


namespace NUMINAMATH_CALUDE_initial_car_cost_l816_81694

/-- The initial cost of John's car, given his Uber profit and the car's trade-in value -/
theorem initial_car_cost (profit : ℕ) (trade_in_value : ℕ) 
  (h1 : profit = 18000)
  (h2 : trade_in_value = 6000) :
  profit + trade_in_value = 24000 := by
  sorry

end NUMINAMATH_CALUDE_initial_car_cost_l816_81694


namespace NUMINAMATH_CALUDE_correct_transformation_l816_81683

def original_expression : List Int := [-17, 3, -5, -8]

def transformed_expression : List Int := [-17, 3, 5, -8]

theorem correct_transformation :
  (original_expression.map (fun x => if x < 0 then -x else x)).foldl (· - ·) 0 =
  transformed_expression.foldl (· + ·) 0 :=
sorry

end NUMINAMATH_CALUDE_correct_transformation_l816_81683


namespace NUMINAMATH_CALUDE_ascending_order_l816_81608

theorem ascending_order (x y : ℝ) (hx : x > 1) (hy : -1 < y ∧ y < 0) :
  y < -y ∧ -y < -x*y ∧ -x*y < x := by sorry

end NUMINAMATH_CALUDE_ascending_order_l816_81608


namespace NUMINAMATH_CALUDE_roots_opposite_signs_l816_81681

theorem roots_opposite_signs (a b c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 2 * b * x + c = 0 ∧ a * y^2 + 2 * b * y + c = 0) →
  (∀ z : ℝ, a^2 * z^2 + 2 * b^2 * z + c^2 ≠ 0) →
  a * c < 0 := by
sorry


end NUMINAMATH_CALUDE_roots_opposite_signs_l816_81681


namespace NUMINAMATH_CALUDE_amanda_candy_bars_l816_81606

/-- Amanda's candy bar problem -/
theorem amanda_candy_bars :
  let initial_bars : ℕ := 7
  let first_day_given : ℕ := 3
  let second_day_given : ℕ := 4 * first_day_given
  let kept_for_self : ℕ := 22
  let bought_next_day : ℕ := kept_for_self + second_day_given - (initial_bars - first_day_given)
  bought_next_day = 30 := by sorry

end NUMINAMATH_CALUDE_amanda_candy_bars_l816_81606


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_l816_81662

/-- Given four points on a plane, the distance between any two points
    is less than or equal to the sum of the distances along a path
    through the other two points. -/
theorem quadrilateral_inequality (A B C D : EuclideanSpace ℝ (Fin 2)) :
  dist A D ≤ dist A B + dist B C + dist C D := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_l816_81662


namespace NUMINAMATH_CALUDE_total_monthly_time_is_200_l816_81654

/-- Represents the days of the week -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Returns true if the given day is a weekday -/
def is_weekday (d : Day) : Bool :=
  match d with
  | Day.Saturday | Day.Sunday => false
  | _ => true

/-- Returns the amount of TV time for a given day -/
def tv_time (d : Day) : Nat :=
  match d with
  | Day.Monday | Day.Wednesday | Day.Friday => 4
  | Day.Tuesday | Day.Thursday => 3
  | Day.Saturday | Day.Sunday => 5

/-- Returns the amount of piano practice time for a given day -/
def piano_time (d : Day) : Nat :=
  if is_weekday d then 2 else 3

/-- Calculates the total weekly TV time -/
def total_weekly_tv_time : Nat :=
  (tv_time Day.Monday) + (tv_time Day.Tuesday) + (tv_time Day.Wednesday) +
  (tv_time Day.Thursday) + (tv_time Day.Friday) + (tv_time Day.Saturday) +
  (tv_time Day.Sunday)

/-- Calculates the average daily TV time -/
def avg_daily_tv_time : Nat :=
  total_weekly_tv_time / 7

/-- Calculates the total weekly video game time -/
def total_weekly_video_game_time : Nat :=
  (avg_daily_tv_time / 2) * 3

/-- Calculates the total weekly piano time -/
def total_weekly_piano_time : Nat :=
  (piano_time Day.Monday) + (piano_time Day.Tuesday) + (piano_time Day.Wednesday) +
  (piano_time Day.Thursday) + (piano_time Day.Friday) + (piano_time Day.Saturday) +
  (piano_time Day.Sunday)

/-- Calculates the total weekly time for all activities -/
def total_weekly_time : Nat :=
  total_weekly_tv_time + total_weekly_video_game_time + total_weekly_piano_time

/-- The main theorem stating that the total monthly time is 200 hours -/
theorem total_monthly_time_is_200 :
  total_weekly_time * 4 = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_monthly_time_is_200_l816_81654


namespace NUMINAMATH_CALUDE_office_absenteeism_l816_81670

theorem office_absenteeism (p : ℕ) (x : ℚ) (h : 0 < p) :
  (1 / ((1 - x) * p) - 1 / p = 1 / (3 * p)) → x = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_office_absenteeism_l816_81670


namespace NUMINAMATH_CALUDE_move_left_result_l816_81680

/-- Moving a point 2 units to the left in a Cartesian coordinate system. -/
def moveLeft (x y : ℝ) : ℝ × ℝ := (x - 2, y)

/-- The theorem stating that moving (-2, 3) 2 units to the left results in (-4, 3). -/
theorem move_left_result : moveLeft (-2) 3 = (-4, 3) := by
  sorry

end NUMINAMATH_CALUDE_move_left_result_l816_81680


namespace NUMINAMATH_CALUDE_h_of_h_of_two_equals_91265_l816_81676

/-- Given a function h(x) = 3x^3 + 2x^2 - x + 1, prove that h(h(2)) = 91265 -/
theorem h_of_h_of_two_equals_91265 : 
  let h : ℝ → ℝ := fun x ↦ 3 * x^3 + 2 * x^2 - x + 1
  h (h 2) = 91265 := by
  sorry

end NUMINAMATH_CALUDE_h_of_h_of_two_equals_91265_l816_81676


namespace NUMINAMATH_CALUDE_quadratic_polynomial_functional_equation_l816_81618

theorem quadratic_polynomial_functional_equation 
  (P : ℝ → ℝ) 
  (h_quadratic : ∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c) 
  (f : ℝ → ℝ) 
  (h_add : ∀ x y, f (x + y) = f x + f y) 
  (h_poly : ∀ x, f (P x) = f x) : 
  ∀ x, f x = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_functional_equation_l816_81618


namespace NUMINAMATH_CALUDE_sum_congruence_modulo_nine_l816_81652

theorem sum_congruence_modulo_nine : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_modulo_nine_l816_81652


namespace NUMINAMATH_CALUDE_race_completion_time_l816_81699

theorem race_completion_time (total_runners : ℕ) (avg_time_all : ℝ) (fastest_time : ℝ) : 
  total_runners = 4 →
  avg_time_all = 30 →
  fastest_time = 15 →
  (((avg_time_all * total_runners) - fastest_time) / (total_runners - 1) : ℝ) = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_race_completion_time_l816_81699


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l816_81657

theorem pure_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) : ℂ) = (a^2 + 2*a - 3 : ℝ) + Complex.I * (a - 1) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l816_81657


namespace NUMINAMATH_CALUDE_constant_e_value_l816_81655

theorem constant_e_value (x y e : ℝ) 
  (h1 : x / (2 * y) = 3 / e) 
  (h2 : (7 * x + 4 * y) / (x - 2 * y) = 25) : 
  e = 2 := by sorry

end NUMINAMATH_CALUDE_constant_e_value_l816_81655


namespace NUMINAMATH_CALUDE_race_time_patrick_l816_81684

theorem race_time_patrick (patrick_time manu_time amy_time : ℕ) : 
  manu_time = patrick_time + 12 →
  amy_time * 2 = manu_time →
  amy_time = 36 →
  patrick_time = 60 := by
sorry

end NUMINAMATH_CALUDE_race_time_patrick_l816_81684


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l816_81656

theorem sqrt_equation_solution (y : ℚ) :
  (Real.sqrt (4 * y + 3) / Real.sqrt (8 * y + 10) = Real.sqrt 3 / 2) →
  y = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l816_81656


namespace NUMINAMATH_CALUDE_negative_expression_l816_81644

theorem negative_expression : 
  let expr1 := -(-1)
  let expr2 := (-1)^2
  let expr3 := |-1|
  let expr4 := -|-1|
  (expr1 ≥ 0 ∧ expr2 ≥ 0 ∧ expr3 ≥ 0 ∧ expr4 < 0) := by sorry

end NUMINAMATH_CALUDE_negative_expression_l816_81644


namespace NUMINAMATH_CALUDE_family_income_proof_l816_81672

/-- Proves that the initial average monthly income of a family is 840 given the conditions --/
theorem family_income_proof (initial_members : ℕ) (deceased_income new_average : ℚ) :
  initial_members = 4 →
  deceased_income = 1410 →
  new_average = 650 →
  (initial_members : ℚ) * (initial_members * new_average + deceased_income) / initial_members = 840 :=
by sorry

end NUMINAMATH_CALUDE_family_income_proof_l816_81672


namespace NUMINAMATH_CALUDE_permutation_equation_solution_l816_81691

theorem permutation_equation_solution (x : ℕ) : 
  (3 * (Nat.factorial 8 / Nat.factorial (8 - x)) = 4 * (Nat.factorial 9 / Nat.factorial (10 - x))) ∧ 
  (1 ≤ x) ∧ (x ≤ 8) → 
  x = 6 := by sorry

end NUMINAMATH_CALUDE_permutation_equation_solution_l816_81691


namespace NUMINAMATH_CALUDE_smallest_number_with_55_divisors_l816_81621

/-- The number of divisors of n = p₁^k₁ * p₂^k₂ * ... * pₘ^kₘ is (k₁+1)(k₂+1)...(kₘ+1) -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n has exactly 55 divisors -/
def has_55_divisors (n : ℕ) : Prop := num_divisors n = 55

theorem smallest_number_with_55_divisors :
  ∃ (n : ℕ), has_55_divisors n ∧ ∀ (m : ℕ), has_55_divisors m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_55_divisors_l816_81621


namespace NUMINAMATH_CALUDE_candy_problem_smallest_n_l816_81603

theorem candy_problem (x y z n : ℕ+) : 
  (18 * x = 21 * y) ∧ (21 * y = 10 * z) ∧ (10 * z = 30 * n) → n ≥ 21 := by
  sorry

theorem smallest_n : ∃ (x y z : ℕ+), 18 * x = 21 * y ∧ 21 * y = 10 * z ∧ 10 * z = 30 * 21 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_smallest_n_l816_81603


namespace NUMINAMATH_CALUDE_triangle_formation_with_6_and_8_l816_81633

/-- A function that checks if three lengths can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating which length among given options can form a triangle with sides 6 and 8 --/
theorem triangle_formation_with_6_and_8 :
  can_form_triangle 6 8 13 ∧
  ¬(can_form_triangle 6 8 1) ∧
  ¬(can_form_triangle 6 8 2) ∧
  ¬(can_form_triangle 6 8 14) := by
  sorry


end NUMINAMATH_CALUDE_triangle_formation_with_6_and_8_l816_81633


namespace NUMINAMATH_CALUDE_det_A_l816_81685

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 0, -2; 8, 5, -4; 3, 3, 6]

theorem det_A : Matrix.det A = 108 := by sorry

end NUMINAMATH_CALUDE_det_A_l816_81685


namespace NUMINAMATH_CALUDE_sum_of_squares_quadratic_solutions_l816_81695

theorem sum_of_squares_quadratic_solutions : ∀ x₁ x₂ : ℝ, 
  x₁^2 - 16*x₁ + 15 = 0 → 
  x₂^2 - 16*x₂ + 15 = 0 → 
  x₁ ≠ x₂ → 
  x₁^2 + x₂^2 = 226 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_quadratic_solutions_l816_81695


namespace NUMINAMATH_CALUDE_max_a_value_l816_81636

theorem max_a_value (x a : ℤ) : 
  (x^2 + a*x = -28) → 
  (a > 0) → 
  ∃ (max_a : ℤ), max_a = 29 ∧ 
    ∀ (b : ℤ), (∃ (y : ℤ), y^2 + b*y = -28) → b ≤ max_a :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l816_81636


namespace NUMINAMATH_CALUDE_square_of_97_l816_81669

theorem square_of_97 : 97 * 97 = 9409 := by
  sorry

end NUMINAMATH_CALUDE_square_of_97_l816_81669


namespace NUMINAMATH_CALUDE_greening_problem_l816_81615

/-- The greening problem -/
theorem greening_problem 
  (total_area : ℝ) 
  (team_a_speed : ℝ) 
  (team_b_speed : ℝ) 
  (team_a_cost : ℝ) 
  (team_b_cost : ℝ) 
  (max_cost : ℝ) 
  (h1 : total_area = 1800) 
  (h2 : team_a_speed = 2 * team_b_speed) 
  (h3 : 400 / team_a_speed + 4 = 400 / team_b_speed) 
  (h4 : team_a_cost = 0.4) 
  (h5 : team_b_cost = 0.25) 
  (h6 : max_cost = 8) :
  ∃ (team_a_area team_b_area min_days : ℝ),
    team_a_area = 100 ∧ 
    team_b_area = 50 ∧ 
    min_days = 10 ∧
    (∀ y : ℝ, y ≥ min_days → 
      team_a_cost * y + team_b_cost * ((total_area - team_a_area * y) / team_b_area) ≤ max_cost) := by
  sorry

end NUMINAMATH_CALUDE_greening_problem_l816_81615


namespace NUMINAMATH_CALUDE_problem_1_l816_81605

theorem problem_1 : (1/3 - 3/4 + 5/6) / (1/12) = 5 := by sorry

end NUMINAMATH_CALUDE_problem_1_l816_81605


namespace NUMINAMATH_CALUDE_power_division_l816_81645

theorem power_division (a : ℝ) : a^8 / a^2 = a^6 :=
by sorry

end NUMINAMATH_CALUDE_power_division_l816_81645


namespace NUMINAMATH_CALUDE_max_sum_theorem_l816_81630

theorem max_sum_theorem (x y z v w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_v : v > 0) (pos_w : w > 0)
  (sum_sq : x^2 + y^2 + z^2 + v^2 + w^2 = 2025) : 
  ∃ (N x_N y_N z_N v_N w_N : ℝ),
    (∀ a b c d e : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → 
      a^2 + b^2 + c^2 + d^2 + e^2 = 2025 → 
      a*c + 3*b*c + 5*c*d + 2*c*e ≤ N) ∧
    x_N > 0 ∧ y_N > 0 ∧ z_N > 0 ∧ v_N > 0 ∧ w_N > 0 ∧
    x_N^2 + y_N^2 + z_N^2 + v_N^2 + w_N^2 = 2025 ∧
    x_N*z_N + 3*y_N*z_N + 5*z_N*v_N + 2*z_N*w_N = N ∧
    N + x_N + y_N + z_N + v_N + w_N = 55 + 3037.5 * Real.sqrt 13 + 5 * Real.sqrt 202.5 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_theorem_l816_81630


namespace NUMINAMATH_CALUDE_solution_f_gt_2_min_value_f_l816_81678

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

-- Theorem for the solution of f(x) > 2
theorem solution_f_gt_2 (x : ℝ) : f x > 2 ↔ x < -7 ∨ x > 5/3 := by sorry

-- Theorem for the minimum value of f
theorem min_value_f : ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ f x = -9/2 := by sorry

end NUMINAMATH_CALUDE_solution_f_gt_2_min_value_f_l816_81678


namespace NUMINAMATH_CALUDE_min_area_triangle_AOB_l816_81679

/-- Given a line l: mx + ny - 1 = 0 intersecting the x-axis at A and y-axis at B,
    and forming a chord of length 2 with the circle x² + y² = 4,
    the minimum area of triangle AOB is 3. -/
theorem min_area_triangle_AOB (m n : ℝ) :
  ∃ (A B : ℝ × ℝ),
    (m * A.1 + n * A.2 - 1 = 0) ∧
    (m * B.1 + n * B.2 - 1 = 0) ∧
    (A.2 = 0) ∧
    (B.1 = 0) ∧
    (∃ (C D : ℝ × ℝ),
      (m * C.1 + n * C.2 - 1 = 0) ∧
      (m * D.1 + n * D.2 - 1 = 0) ∧
      (C.1^2 + C.2^2 = 4) ∧
      (D.1^2 + D.2^2 = 4) ∧
      ((C.1 - D.1)^2 + (C.2 - D.2)^2 = 4)) →
  ∃ (area_min : ℝ),
    (∀ (A' B' : ℝ × ℝ),
      (m * A'.1 + n * A'.2 - 1 = 0) →
      (A'.2 = 0) →
      (m * B'.1 + n * B'.2 - 1 = 0) →
      (B'.1 = 0) →
      area_min ≤ (1/2) * A'.1 * B'.2) ∧
    area_min = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_area_triangle_AOB_l816_81679


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_is_seven_min_value_exists_l816_81634

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h : 1 / (x + 1) + 8 / y = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1 / (a + 1) + 8 / b = 2 → 2 * x + y ≤ 2 * a + b :=
by sorry

theorem min_value_is_seven (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h : 1 / (x + 1) + 8 / y = 2) : 
  2 * x + y ≥ 7 :=
by sorry

theorem min_value_exists (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h : 1 / (x + 1) + 8 / y = 2) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1 / (a + 1) + 8 / b = 2 ∧ 2 * a + b = 7 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_is_seven_min_value_exists_l816_81634


namespace NUMINAMATH_CALUDE_razorback_shop_profit_l816_81658

theorem razorback_shop_profit : 
  let tshirt_profit : ℕ := 67
  let jersey_profit : ℕ := 165
  let hat_profit : ℕ := 32
  let jacket_profit : ℕ := 245
  let tshirts_sold : ℕ := 74
  let jerseys_sold : ℕ := 156
  let hats_sold : ℕ := 215
  let jackets_sold : ℕ := 45
  (tshirt_profit * tshirts_sold + 
   jersey_profit * jerseys_sold + 
   hat_profit * hats_sold + 
   jacket_profit * jackets_sold) = 48603 :=
by sorry

end NUMINAMATH_CALUDE_razorback_shop_profit_l816_81658


namespace NUMINAMATH_CALUDE_exists_n_congruence_l816_81665

theorem exists_n_congruence (l : ℕ+) : ∃ n : ℕ, (n^n + 47) % (2^l.val) = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_congruence_l816_81665


namespace NUMINAMATH_CALUDE_trapezoidal_channel_bottom_width_l816_81613

theorem trapezoidal_channel_bottom_width
  (top_width : ℝ)
  (area : ℝ)
  (depth : ℝ)
  (h_top_width : top_width = 12)
  (h_area : area = 700)
  (h_depth : depth = 70) :
  ∃ bottom_width : ℝ,
    bottom_width = 8 ∧
    area = (1 / 2) * (top_width + bottom_width) * depth :=
by sorry

end NUMINAMATH_CALUDE_trapezoidal_channel_bottom_width_l816_81613


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l816_81689

/-- Given the following conditions:
    - 3 blankets at Rs. 100 each
    - 6 blankets at Rs. 150 each
    - 2 blankets at an unknown rate
    - The average price of all blankets is Rs. 150
    Prove that the unknown rate must be Rs. 225 per blanket -/
theorem unknown_blanket_rate (price1 : ℕ) (price2 : ℕ) (unknown_price : ℕ) 
    (h1 : price1 = 100)
    (h2 : price2 = 150)
    (h3 : (3 * price1 + 6 * price2 + 2 * unknown_price) / 11 = 150) :
    unknown_price = 225 := by
  sorry

end NUMINAMATH_CALUDE_unknown_blanket_rate_l816_81689


namespace NUMINAMATH_CALUDE_encyclopedia_sorting_l816_81682

/-- Represents the number of volumes in the encyclopedia --/
def n : ℕ := 30

/-- Represents an operation of swapping two adjacent volumes --/
def swap : ℕ → ℕ → List ℕ → List ℕ := sorry

/-- Checks if a list of volumes is in the correct order --/
def is_sorted : List ℕ → Prop := sorry

/-- The maximum number of disorders in any arrangement of n volumes --/
def max_disorders (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The minimum number of operations required to sort n volumes --/
def min_operations (n : ℕ) : ℕ := max_disorders n

theorem encyclopedia_sorting (arrangement : List ℕ) 
  (h : arrangement.length = n) :
  ∃ (sequence : List (ℕ × ℕ)), 
    sequence.length ≤ min_operations n ∧ 
    is_sorted (sequence.foldl (λ acc (i, j) => swap i j acc) arrangement) := by
  sorry

#eval min_operations n  -- Should evaluate to 435

end NUMINAMATH_CALUDE_encyclopedia_sorting_l816_81682


namespace NUMINAMATH_CALUDE_flea_problem_l816_81677

/-- Represents the number of ways a flea can reach a point given the distance and number of jumps -/
def flea_jumps (distance : ℤ) (jumps : ℕ) : ℕ := sorry

/-- Represents whether it's possible for a flea to reach a point given the distance and number of jumps -/
def flea_can_reach (distance : ℤ) (jumps : ℕ) : Prop := sorry

theorem flea_problem :
  (flea_jumps 5 7 = 7) ∧
  (flea_jumps 5 9 = 36) ∧
  ¬(flea_can_reach 2013 2028) := by sorry

end NUMINAMATH_CALUDE_flea_problem_l816_81677


namespace NUMINAMATH_CALUDE_tangent_implies_t_equals_4e_l816_81623

-- Define the curves C₁ and C₂
def C₁ (t : ℝ) (x y : ℝ) : Prop := y^2 = t*x ∧ y > 0 ∧ t > 0

def C₂ (x y : ℝ) : Prop := y = Real.exp (x + 1) - 1

-- Define the tangent line condition
def tangent_condition (t : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), C₁ t (4/t) 2 ∧ C₁ t x₀ y₀ ∧ C₂ x₀ y₀ ∧
  (∀ (x y : ℝ), y - 2 = (t/4)*(x - 4/t) → (C₁ t x y ∨ C₂ x y))

-- State the theorem
theorem tangent_implies_t_equals_4e :
  ∀ t : ℝ, tangent_condition t → t = 4 * Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_implies_t_equals_4e_l816_81623


namespace NUMINAMATH_CALUDE_solution_to_modular_equation_l816_81638

theorem solution_to_modular_equation :
  ∃ x : ℤ, (7 * x + 2) % 15 = 11 % 15 ∧ x % 15 = 12 % 15 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_modular_equation_l816_81638


namespace NUMINAMATH_CALUDE_grandson_age_l816_81667

theorem grandson_age (grandson_age grandfather_age : ℕ) : 
  grandfather_age = 6 * grandson_age →
  (grandson_age + 4) + (grandfather_age + 4) = 78 →
  grandson_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_grandson_age_l816_81667


namespace NUMINAMATH_CALUDE_nth_equation_pattern_l816_81664

theorem nth_equation_pattern (n : ℕ) :
  (-n : ℚ) * (n / (n + 1)) = -n + (n / (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_pattern_l816_81664


namespace NUMINAMATH_CALUDE_dividend_divisor_quotient_ratio_l816_81602

theorem dividend_divisor_quotient_ratio 
  (dividend : ℚ) (divisor : ℚ) (quotient : ℚ) 
  (h : dividend / divisor = 9 / 2) : 
  divisor / quotient = 2 / 9 := by
sorry

end NUMINAMATH_CALUDE_dividend_divisor_quotient_ratio_l816_81602


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l816_81619

/-- Given vectors a and b in ℝ², if a is perpendicular to (a + 2b), then the second component of b is -3/4 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h : a = (-1, 2)) (h' : b.1 = 1) 
    (h'' : a.1 * (a.1 + 2 * b.1) + a.2 * (a.2 + 2 * b.2) = 0) : 
    b.2 = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l816_81619


namespace NUMINAMATH_CALUDE_tomatoes_rotted_l816_81626

def initial_shipment : ℕ := 1000
def saturday_sales : ℕ := 300
def monday_shipment : ℕ := 2 * initial_shipment
def tuesday_ready : ℕ := 2500

theorem tomatoes_rotted (rotted : ℕ) : 
  rotted = initial_shipment - saturday_sales + monday_shipment - tuesday_ready := by sorry

end NUMINAMATH_CALUDE_tomatoes_rotted_l816_81626


namespace NUMINAMATH_CALUDE_profit_decrease_calculation_l816_81631

theorem profit_decrease_calculation (march_profit : ℝ) (april_may_decrease : ℝ) :
  march_profit > 0 →
  (march_profit * 1.3 * (1 - april_may_decrease / 100) * 1.5 = march_profit * 1.5600000000000001) →
  april_may_decrease = 20 := by
sorry

end NUMINAMATH_CALUDE_profit_decrease_calculation_l816_81631


namespace NUMINAMATH_CALUDE_f_negative_2017_l816_81686

noncomputable def f (x : ℝ) : ℝ :=
  ((x + 1)^2 + Real.log (Real.sqrt (1 + 9*x^2) - 3*x) * Real.cos x) / (x^2 + 1)

theorem f_negative_2017 (h : f 2017 = 2016) : f (-2017) = -2014 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_2017_l816_81686


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l816_81688

/-- A function that returns the digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a number ends with 37 -/
def ends_with_37 (n : ℕ) : Prop := sorry

theorem smallest_number_with_conditions : 
  ∀ n : ℕ, 
    n ≥ 99937 → 
    (ends_with_37 n ∧ digit_sum n = 37 ∧ n % 37 = 0) → 
    n = 99937 := by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l816_81688


namespace NUMINAMATH_CALUDE_second_village_sales_l816_81698

/-- Given the number of cookie packs sold in the first village and the total number of packs sold,
    calculate the number of packs sold in the second village. -/
def cookiesSoldInSecondVillage (firstVillage : ℕ) (total : ℕ) : ℕ :=
  total - firstVillage

/-- Theorem stating that the number of cookie packs sold in the second village
    is equal to the total number of packs sold minus the number sold in the first village. -/
theorem second_village_sales (firstVillage : ℕ) (total : ℕ) 
    (h : firstVillage ≤ total) :
  cookiesSoldInSecondVillage firstVillage total = total - firstVillage := by
  sorry

#eval cookiesSoldInSecondVillage 23 51  -- Expected output: 28

end NUMINAMATH_CALUDE_second_village_sales_l816_81698


namespace NUMINAMATH_CALUDE_rectangular_field_length_l816_81625

theorem rectangular_field_length
  (area : ℝ)
  (length_increase : ℝ)
  (area_increase : ℝ)
  (h1 : area = 144)
  (h2 : length_increase = 6)
  (h3 : area_increase = 54)
  (h4 : ∀ l w, l * w = area → (l + length_increase) * w = area + area_increase) :
  ∃ l w, l * w = area ∧ l = 16 :=
sorry

end NUMINAMATH_CALUDE_rectangular_field_length_l816_81625


namespace NUMINAMATH_CALUDE_trailing_zeros_of_product_sum_of_digits_l816_81661

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Product of sum of digits from 1 to n -/
def product_of_sum_of_digits (n : ℕ) : ℕ := sorry

/-- Number of trailing zeros in a natural number -/
def trailing_zeros (n : ℕ) : ℕ := sorry

/-- Theorem: The number of trailing zeros in the product of sum of digits from 1 to 100 is 19 -/
theorem trailing_zeros_of_product_sum_of_digits : 
  trailing_zeros (product_of_sum_of_digits 100) = 19 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_product_sum_of_digits_l816_81661


namespace NUMINAMATH_CALUDE_magic_wheel_product_l816_81600

def is_even (n : ℕ) : Prop := n % 2 = 0

def has_two_odd_between (a d : ℕ) : Prop :=
  ∃ b c : ℕ, a < b ∧ b < c ∧ c < d ∧
  ¬(is_even b) ∧ ¬(is_even c) ∧
  (d - a) % 16 = 3

theorem magic_wheel_product :
  ∀ a d : ℕ,
  1 ≤ a ∧ a ≤ 16 ∧
  1 ≤ d ∧ d ≤ 16 ∧
  is_even a ∧
  is_even d ∧
  has_two_odd_between a d →
  a * d = 120 :=
sorry

end NUMINAMATH_CALUDE_magic_wheel_product_l816_81600


namespace NUMINAMATH_CALUDE_percentage_8_years_plus_is_24_percent_l816_81675

/-- Represents the number of employees for each year range --/
structure EmployeeDistribution :=
  (less_than_2 : ℕ)
  (from_2_to_4 : ℕ)
  (from_4_to_6 : ℕ)
  (from_6_to_8 : ℕ)
  (from_8_to_10 : ℕ)
  (from_10_to_12 : ℕ)
  (from_12_to_14 : ℕ)

/-- Calculates the total number of employees --/
def total_employees (d : EmployeeDistribution) : ℕ :=
  d.less_than_2 + d.from_2_to_4 + d.from_4_to_6 + d.from_6_to_8 +
  d.from_8_to_10 + d.from_10_to_12 + d.from_12_to_14

/-- Calculates the number of employees with 8 or more years of employment --/
def employees_8_years_plus (d : EmployeeDistribution) : ℕ :=
  d.from_8_to_10 + d.from_10_to_12 + d.from_12_to_14

/-- Calculates the percentage of employees with 8 or more years of employment --/
def percentage_8_years_plus (d : EmployeeDistribution) : ℚ :=
  (employees_8_years_plus d : ℚ) / (total_employees d : ℚ) * 100

/-- Theorem stating that the percentage of employees with 8 or more years of employment is 24% --/
theorem percentage_8_years_plus_is_24_percent (d : EmployeeDistribution)
  (h1 : d.less_than_2 = 4)
  (h2 : d.from_2_to_4 = 6)
  (h3 : d.from_4_to_6 = 5)
  (h4 : d.from_6_to_8 = 4)
  (h5 : d.from_8_to_10 = 3)
  (h6 : d.from_10_to_12 = 2)
  (h7 : d.from_12_to_14 = 1) :
  percentage_8_years_plus d = 24 := by
  sorry

end NUMINAMATH_CALUDE_percentage_8_years_plus_is_24_percent_l816_81675


namespace NUMINAMATH_CALUDE_sequence_periodicity_l816_81607

theorem sequence_periodicity (u : ℕ → ℝ) 
  (h : ∀ n : ℕ, u (n + 2) = |u (n + 1)| - u n) : 
  ∃ p : ℕ+, ∀ n : ℕ, u n = u (n + p) := by sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l816_81607


namespace NUMINAMATH_CALUDE_number_of_paths_equals_combinations_l816_81612

-- Define the grid size
def gridSize : Nat := 6

-- Define the total number of moves
def totalMoves : Nat := gridSize * 2

-- Define the number of rightward (or downward) moves
def directionMoves : Nat := gridSize

-- Theorem statement
theorem number_of_paths_equals_combinations :
  (Nat.choose totalMoves directionMoves) = 924 := by
  sorry

end NUMINAMATH_CALUDE_number_of_paths_equals_combinations_l816_81612


namespace NUMINAMATH_CALUDE_power_equation_solution_l816_81646

theorem power_equation_solution :
  ∃ x : ℤ, (3 : ℝ)^7 * (3 : ℝ)^x = 81 ∧ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l816_81646


namespace NUMINAMATH_CALUDE_four_digit_perfect_cubes_divisible_by_16_l816_81659

theorem four_digit_perfect_cubes_divisible_by_16 :
  (Finset.filter (fun n : ℕ => 
    1000 ≤ 8 * n^3 ∧ 8 * n^3 ≤ 9999) (Finset.range 1000)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_perfect_cubes_divisible_by_16_l816_81659


namespace NUMINAMATH_CALUDE_farmer_apples_l816_81635

theorem farmer_apples (initial_apples given_apples : ℕ) 
  (h1 : initial_apples = 127)
  (h2 : given_apples = 88) :
  initial_apples - given_apples = 39 := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l816_81635
