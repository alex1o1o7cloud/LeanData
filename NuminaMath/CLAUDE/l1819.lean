import Mathlib

namespace NUMINAMATH_CALUDE_money_left_after_gift_l1819_181972

def gift_package_cost : ℚ := 445
def erika_savings : ℚ := 155
def sam_savings : ℚ := 175
def cake_flowers_skincare_cost : ℚ := 25 + 35 + 45

def rick_savings : ℚ := gift_package_cost / 2
def amy_savings : ℚ := 2 * cake_flowers_skincare_cost

def total_savings : ℚ := erika_savings + rick_savings + sam_savings + amy_savings

theorem money_left_after_gift (h : total_savings - gift_package_cost = 317.5) :
  total_savings - gift_package_cost = 317.5 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_gift_l1819_181972


namespace NUMINAMATH_CALUDE_james_bike_ride_l1819_181937

theorem james_bike_ride (first_hour : ℝ) : 
  first_hour > 0 →
  let second_hour := 1.2 * first_hour
  let third_hour := 1.25 * second_hour
  first_hour + second_hour + third_hour = 55.5 →
  second_hour = 18 := by
sorry

end NUMINAMATH_CALUDE_james_bike_ride_l1819_181937


namespace NUMINAMATH_CALUDE_share_difference_l1819_181905

/-- Represents the distribution of money among three people -/
structure MoneyDistribution where
  total : ℕ
  ratio_faruk : ℕ
  ratio_vasim : ℕ
  ratio_ranjith : ℕ

/-- Calculates the share of a person given their ratio and the total amount -/
def calculate_share (dist : MoneyDistribution) (ratio : ℕ) : ℕ :=
  dist.total * ratio / (dist.ratio_faruk + dist.ratio_vasim + dist.ratio_ranjith)

theorem share_difference (dist : MoneyDistribution) 
  (h1 : dist.ratio_faruk = 3)
  (h2 : dist.ratio_vasim = 5)
  (h3 : dist.ratio_ranjith = 9)
  (h4 : calculate_share dist dist.ratio_vasim = 1500) :
  calculate_share dist dist.ratio_ranjith - calculate_share dist dist.ratio_faruk = 1800 :=
by sorry

end NUMINAMATH_CALUDE_share_difference_l1819_181905


namespace NUMINAMATH_CALUDE_chloe_carrot_count_l1819_181961

/-- Given Chloe's carrot picking scenario, prove the final number of carrots. -/
theorem chloe_carrot_count (initial_carrots : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ) 
  (h1 : initial_carrots = 48)
  (h2 : thrown_out = 45)
  (h3 : picked_next_day = 42) :
  initial_carrots - thrown_out + picked_next_day = 45 :=
by sorry

end NUMINAMATH_CALUDE_chloe_carrot_count_l1819_181961


namespace NUMINAMATH_CALUDE_carter_drum_sticks_l1819_181957

/-- The number of drum stick sets Carter uses per show -/
def sticks_used_per_show : ℕ := 8

/-- The number of drum stick sets Carter tosses to the audience after each show -/
def sticks_tossed_per_show : ℕ := 10

/-- The number of nights Carter performs -/
def number_of_shows : ℕ := 45

/-- The total number of drum stick sets Carter goes through -/
def total_sticks : ℕ := (sticks_used_per_show + sticks_tossed_per_show) * number_of_shows

theorem carter_drum_sticks :
  total_sticks = 810 := by sorry

end NUMINAMATH_CALUDE_carter_drum_sticks_l1819_181957


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l1819_181944

theorem nested_fraction_evaluation :
  1 + 1 / (2 + 1 / (3 + 2)) = 16 / 11 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l1819_181944


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l1819_181942

theorem smallest_gcd_multiple (a b : ℕ+) (h : Nat.gcd a b = 10) :
  (∃ (x y : ℕ+), Nat.gcd x y = 10 ∧ Nat.gcd (8 * x) (14 * y) = 20) ∧
  ∀ (c d : ℕ+), Nat.gcd c d = 10 → Nat.gcd (8 * c) (14 * d) ≥ 20 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l1819_181942


namespace NUMINAMATH_CALUDE_contact_list_count_is_45_l1819_181991

/-- The number of people on Jerome's contact list at the end of the month -/
def contact_list_count : ℕ :=
  let classmates : ℕ := 20
  let out_of_school_friends : ℕ := classmates / 2
  let immediate_family : ℕ := 3
  let extended_family_added : ℕ := 5
  let acquaintances_added : ℕ := 7
  let coworkers_added : ℕ := 10
  let extended_family_removed : ℕ := 3
  let acquaintances_removed : ℕ := 4
  let coworkers_removed : ℕ := (coworkers_added * 3) / 10

  let total_added : ℕ := classmates + out_of_school_friends + immediate_family + 
                         extended_family_added + acquaintances_added + coworkers_added
  let total_removed : ℕ := extended_family_removed + acquaintances_removed + coworkers_removed

  total_added - total_removed

theorem contact_list_count_is_45 : contact_list_count = 45 := by
  sorry

end NUMINAMATH_CALUDE_contact_list_count_is_45_l1819_181991


namespace NUMINAMATH_CALUDE_function_max_value_solution_l1819_181962

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + a^2 - 2 * a + 2

/-- The maximum value of f(x) on the interval [0, 2] -/
def max_value : ℝ := 3

/-- The theorem stating the solution -/
theorem function_max_value_solution (a : ℝ) :
  (∀ x ∈ Set.Icc 0 2, f a x ≤ max_value) ∧
  (∃ x ∈ Set.Icc 0 2, f a x = max_value) →
  a = 5 - Real.sqrt 10 ∨ a = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_function_max_value_solution_l1819_181962


namespace NUMINAMATH_CALUDE_equal_area_triangles_l1819_181904

noncomputable def triangle_area (a b : ℝ) (θ : ℝ) : ℝ := (1/2) * a * b * Real.sin θ

theorem equal_area_triangles (AB AC AD : ℝ) (θ : ℝ) (AE : ℝ) : 
  AB = 4 →
  AC = 5 →
  AD = 2.5 →
  θ = Real.pi / 3 →
  triangle_area AB AC θ = triangle_area AD AE θ →
  AE = 8 :=
by sorry

end NUMINAMATH_CALUDE_equal_area_triangles_l1819_181904


namespace NUMINAMATH_CALUDE_f_geq_g_for_all_real_l1819_181965

theorem f_geq_g_for_all_real : ∀ x : ℝ, x^2 * Real.exp x ≥ 2 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_f_geq_g_for_all_real_l1819_181965


namespace NUMINAMATH_CALUDE_honey_production_optimal_tax_revenue_optimal_l1819_181920

/-- The inverse demand function for honey -/
def inverse_demand (Q : ℝ) : ℝ := 310 - 3 * Q

/-- The production cost per jar of honey -/
def production_cost : ℝ := 10

/-- The profit function without tax -/
def profit (Q : ℝ) : ℝ := (inverse_demand Q) * Q - production_cost * Q

/-- The profit function with tax -/
def profit_with_tax (Q t : ℝ) : ℝ := (inverse_demand Q) * Q - production_cost * Q - t * Q

/-- The tax revenue function -/
def tax_revenue (Q t : ℝ) : ℝ := Q * t

theorem honey_production_optimal (Q : ℝ) :
  profit Q ≤ profit 50 := by sorry

theorem tax_revenue_optimal (t : ℝ) :
  tax_revenue ((310 - t) / 6) t ≤ tax_revenue ((310 - 150) / 6) 150 := by sorry

end NUMINAMATH_CALUDE_honey_production_optimal_tax_revenue_optimal_l1819_181920


namespace NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_100_l1819_181950

theorem greatest_common_multiple_9_15_under_100 : ∃ n : ℕ, n = 90 ∧ 
  (∀ m : ℕ, m < 100 → m % 9 = 0 → m % 15 = 0 → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_100_l1819_181950


namespace NUMINAMATH_CALUDE_halloween_cleanup_time_l1819_181919

theorem halloween_cleanup_time (
  egg_cleanup_time : ℕ)
  (tp_cleanup_time : ℕ)
  (graffiti_cleanup_time : ℕ)
  (pumpkin_cleanup_time : ℕ)
  (num_eggs : ℕ)
  (num_tp_rolls : ℕ)
  (sq_ft_graffiti : ℕ)
  (num_pumpkins : ℕ)
  (h1 : egg_cleanup_time = 15)
  (h2 : tp_cleanup_time = 30)
  (h3 : graffiti_cleanup_time = 45)
  (h4 : pumpkin_cleanup_time = 10)
  (h5 : num_eggs = 60)
  (h6 : num_tp_rolls = 7)
  (h7 : sq_ft_graffiti = 8)
  (h8 : num_pumpkins = 5) :
  (num_eggs * egg_cleanup_time) / 60 +
  num_tp_rolls * tp_cleanup_time +
  sq_ft_graffiti * graffiti_cleanup_time +
  num_pumpkins * pumpkin_cleanup_time = 635 := by
sorry

end NUMINAMATH_CALUDE_halloween_cleanup_time_l1819_181919


namespace NUMINAMATH_CALUDE_purely_imaginary_implies_a_eq_three_halves_l1819_181955

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined as (1+2i)(3+ai) where a is a real number. -/
def z (a : ℝ) : ℂ := (1 + 2*Complex.I) * (3 + a*Complex.I)

/-- If z is purely imaginary, then a = 3/2. -/
theorem purely_imaginary_implies_a_eq_three_halves :
  ∀ a : ℝ, IsPurelyImaginary (z a) → a = 3/2 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_implies_a_eq_three_halves_l1819_181955


namespace NUMINAMATH_CALUDE_train_speed_correct_l1819_181953

/-- The speed of the train in km/hr given the conditions -/
def train_speed : ℝ :=
  let train_length : ℝ := 110  -- meters
  let passing_time : ℝ := 4.399648028157747  -- seconds
  let man_speed : ℝ := 6  -- km/hr
  84  -- km/hr

/-- Theorem stating that the calculated train speed is correct -/
theorem train_speed_correct :
  let train_length : ℝ := 110  -- meters
  let passing_time : ℝ := 4.399648028157747  -- seconds
  let man_speed : ℝ := 6  -- km/hr
  train_speed = 84 := by sorry

end NUMINAMATH_CALUDE_train_speed_correct_l1819_181953


namespace NUMINAMATH_CALUDE_min_S6_arithmetic_sequence_l1819_181922

/-- Given an arithmetic sequence with common ratio q > 1, where S_n denotes the sum of first n terms,
    and S_4 = 2S_2 + 1, the minimum value of S_6 is 2√3 + 3. -/
theorem min_S6_arithmetic_sequence (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  q > 1 →
  (∀ n, a (n + 1) = a n + q) →
  (∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * q) / 2) →
  S 4 = 2 * S 2 + 1 →
  (∀ s : ℝ, s = S 6 → s ≥ 2 * Real.sqrt 3 + 3) ∧
  ∃ s : ℝ, s = S 6 ∧ s = 2 * Real.sqrt 3 + 3 :=
by sorry


end NUMINAMATH_CALUDE_min_S6_arithmetic_sequence_l1819_181922


namespace NUMINAMATH_CALUDE_safari_arrangement_l1819_181934

/-- Represents the number of animal pairs in the safari park -/
def num_pairs : ℕ := 6

/-- Calculates the number of ways to arrange animals with alternating genders -/
def arrange_animals : ℕ := sorry

/-- Theorem stating the number of ways to arrange the animals -/
theorem safari_arrangement :
  arrange_animals = 86400 := by sorry

end NUMINAMATH_CALUDE_safari_arrangement_l1819_181934


namespace NUMINAMATH_CALUDE_slope_of_l₃_l1819_181973

-- Define the lines and points
def l₁ (x y : ℝ) : Prop := 4 * x - 3 * y = 2
def l₂ (x y : ℝ) : Prop := y = 2
def A : ℝ × ℝ := (0, -3)

-- Define the existence of point B
def B_exists : Prop := ∃ x y : ℝ, l₁ x y ∧ l₂ x y

-- Define the existence of point C
def C_exists : Prop := ∃ x : ℝ, l₂ x (2 : ℝ)

-- Define the properties of line l₃
def l₃_properties (m : ℝ) : Prop :=
  m > 0 ∧ 
  (∃ b : ℝ, ∀ x : ℝ, m * x + b = -3 → x = 0) ∧
  (∃ x : ℝ, l₂ x (m * x + -3))

-- Define the area of triangle ABC
def triangle_area (m : ℝ) : Prop :=
  ∃ B C : ℝ × ℝ, 
    l₁ B.1 B.2 ∧ l₂ B.1 B.2 ∧
    l₂ C.1 C.2 ∧
    (C.2 - A.2) = m * (C.1 - A.1) ∧
    abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) = 10

-- Theorem statement
theorem slope_of_l₃ :
  B_exists → C_exists → ∃ m : ℝ, l₃_properties m ∧ triangle_area m → m = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_l₃_l1819_181973


namespace NUMINAMATH_CALUDE_complex_modulus_l1819_181933

theorem complex_modulus (z : ℂ) (h : (z - 2) * Complex.I = 1 + Complex.I) : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1819_181933


namespace NUMINAMATH_CALUDE_max_min_a_plus_b_l1819_181977

noncomputable def f (x : ℝ) : ℝ := x - Real.sqrt x

theorem max_min_a_plus_b (a b : ℝ) (h : f (a + 1) + f (b + 2) = 3) :
  (a + b ≤ 1 + Real.sqrt 7) ∧ (a + b ≥ (1 + Real.sqrt 13) / 2) :=
by sorry

end NUMINAMATH_CALUDE_max_min_a_plus_b_l1819_181977


namespace NUMINAMATH_CALUDE_sequence_length_l1819_181941

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

theorem sequence_length :
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence 2.5 5 n = 87.5 ∧
  ∀ m : ℕ, m > 0 ∧ m ≠ n → arithmetic_sequence 2.5 5 m ≠ 87.5 :=
by
  use 18
  sorry

end NUMINAMATH_CALUDE_sequence_length_l1819_181941


namespace NUMINAMATH_CALUDE_partial_week_salary_l1819_181924

/-- Calculates the salary for a partial work week --/
theorem partial_week_salary
  (usual_hours : ℝ)
  (worked_fraction : ℝ)
  (hourly_rate : ℝ)
  (h1 : usual_hours = 40)
  (h2 : worked_fraction = 4/5)
  (h3 : hourly_rate = 15) :
  worked_fraction * usual_hours * hourly_rate = 480 := by
  sorry

#check partial_week_salary

end NUMINAMATH_CALUDE_partial_week_salary_l1819_181924


namespace NUMINAMATH_CALUDE_solution_equality_l1819_181998

theorem solution_equality (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0)
  (h₁ : (x₁^2 - x₃*x₅)*(x₂^2 - x₃*x₅) ≤ 0)
  (h₂ : (x₂^2 - x₄*x₁)*(x₃^2 - x₄*x₁) ≤ 0)
  (h₃ : (x₃^2 - x₅*x₂)*(x₄^2 - x₅*x₂) ≤ 0)
  (h₄ : (x₄^2 - x₁*x₃)*(x₅^2 - x₁*x₃) ≤ 0)
  (h₅ : (x₅^2 - x₂*x₄)*(x₁^2 - x₂*x₄) ≤ 0) :
  x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅ :=
by sorry

end NUMINAMATH_CALUDE_solution_equality_l1819_181998


namespace NUMINAMATH_CALUDE_g_of_6_eq_0_l1819_181901

/-- The polynomial g(x) = 3x^4 - 18x^3 + 31x^2 - 29x - 72 -/
def g (x : ℝ) : ℝ := 3*x^4 - 18*x^3 + 31*x^2 - 29*x - 72

/-- Theorem: g(6) = 0 -/
theorem g_of_6_eq_0 : g 6 = 0 := by sorry

end NUMINAMATH_CALUDE_g_of_6_eq_0_l1819_181901


namespace NUMINAMATH_CALUDE_max_rooks_on_100x100_board_l1819_181940

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)

/-- Represents a nearsighted rook --/
structure NearsightedRook :=
  (range : ℕ)

/-- Calculates the maximum number of non-attacking nearsighted rooks on a chessboard --/
def max_non_attacking_rooks (board : Chessboard) (rook : NearsightedRook) : ℕ :=
  sorry

/-- Theorem stating the maximum number of non-attacking nearsighted rooks on a 100x100 board --/
theorem max_rooks_on_100x100_board :
  let board : Chessboard := ⟨100⟩
  let rook : NearsightedRook := ⟨60⟩
  max_non_attacking_rooks board rook = 178 :=
sorry

end NUMINAMATH_CALUDE_max_rooks_on_100x100_board_l1819_181940


namespace NUMINAMATH_CALUDE_trig_identities_l1819_181976

theorem trig_identities (α : Real) (h : Real.tan α = 2) :
  (Real.sin (α - 3 * Real.pi) + Real.cos (Real.pi + α)) /
  (Real.sin (-α) - Real.cos (Real.pi + α)) = 3 ∧
  Real.cos α ^ 2 - 2 * Real.sin α * Real.cos α = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l1819_181976


namespace NUMINAMATH_CALUDE_equation_solution_set_l1819_181917

theorem equation_solution_set : ∃ (S : Set ℝ),
  S = {x : ℝ | Real.sqrt (x + 3 - 4 * Real.sqrt (x - 1)) + Real.sqrt (x + 8 - 6 * Real.sqrt (x - 1)) = 1 ∧
                x ≥ 5 ∧ x ≤ 10} ∧
  ∀ x : ℝ, x ∈ S ↔ (Real.sqrt (x + 3 - 4 * Real.sqrt (x - 1)) + Real.sqrt (x + 8 - 6 * Real.sqrt (x - 1)) = 1 ∧
                    x ≥ 5 ∧ x ≤ 10) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_set_l1819_181917


namespace NUMINAMATH_CALUDE_article_price_proof_l1819_181995

/-- The normal price of an article before discounts -/
def normal_price : ℝ := 100

/-- The final price after discounts -/
def final_price : ℝ := 72

/-- The first discount rate -/
def discount1 : ℝ := 0.1

/-- The second discount rate -/
def discount2 : ℝ := 0.2

theorem article_price_proof :
  normal_price * (1 - discount1) * (1 - discount2) = final_price :=
by sorry

end NUMINAMATH_CALUDE_article_price_proof_l1819_181995


namespace NUMINAMATH_CALUDE_car_average_speed_l1819_181986

/-- Proves that the average speed of a car is 72 km/h given specific travel conditions -/
theorem car_average_speed (s : ℝ) (h : s > 0) : 
  let t1 := s / 2 / 60
  let t2 := s / 6 / 120
  let t3 := s / 3 / 80
  s / (t1 + t2 + t3) = 72 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l1819_181986


namespace NUMINAMATH_CALUDE_trigonometric_fraction_bounds_l1819_181971

theorem trigonometric_fraction_bounds (x : ℝ) : 
  -1/3 ≤ (6 * Real.cos x + Real.sin x - 5) / (2 * Real.cos x - 3 * Real.sin x - 5) ∧ 
  (6 * Real.cos x + Real.sin x - 5) / (2 * Real.cos x - 3 * Real.sin x - 5) ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_fraction_bounds_l1819_181971


namespace NUMINAMATH_CALUDE_regression_unit_change_l1819_181931

/-- Represents a linear regression equation of the form y = mx + b -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- The change in y for a unit change in x in a linear regression -/
def unitChange (reg : LinearRegression) : ℝ := reg.slope

theorem regression_unit_change 
  (reg : LinearRegression) 
  (h : reg = { slope := -1.5, intercept := 2 }) : 
  unitChange reg = -1.5 := by sorry

end NUMINAMATH_CALUDE_regression_unit_change_l1819_181931


namespace NUMINAMATH_CALUDE_dissimilarTerms_eq_distributionWays_l1819_181915

/-- The number of dissimilar terms in the expansion of (a + b + c + d)^10 -/
def dissimilarTerms : ℕ := Nat.choose 13 3

/-- The number of ways to distribute 10 indistinguishable objects into 4 distinguishable boxes -/
def distributionWays : ℕ := Nat.choose 13 3

theorem dissimilarTerms_eq_distributionWays : dissimilarTerms = distributionWays := by
  sorry

end NUMINAMATH_CALUDE_dissimilarTerms_eq_distributionWays_l1819_181915


namespace NUMINAMATH_CALUDE_meow_to_paw_ratio_l1819_181970

/-- The number of cats in Cat Cafe Cool -/
def cool_cats : ℕ := 5

/-- The number of cats in Cat Cafe Paw -/
def paw_cats : ℕ := 2 * cool_cats

/-- The total number of cats in Cat Cafe Meow and Cat Cafe Paw combined -/
def total_cats : ℕ := 40

/-- The number of cats in Cat Cafe Meow -/
def meow_cats : ℕ := total_cats - paw_cats

/-- The theorem stating that Cat Cafe Meow has 3 times as many cats as Cat Cafe Paw -/
theorem meow_to_paw_ratio : meow_cats = 3 * paw_cats := by
  sorry

end NUMINAMATH_CALUDE_meow_to_paw_ratio_l1819_181970


namespace NUMINAMATH_CALUDE_direction_vector_implies_a_eq_plus_minus_two_l1819_181963

/-- Two lines with equations ax + 2y + 3 = 0 and 2x + ay - 1 = 0 have the same direction vector -/
def same_direction_vector (a : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ k * a = 2 ∧ k * 2 = a

theorem direction_vector_implies_a_eq_plus_minus_two (a : ℝ) :
  same_direction_vector a → a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_direction_vector_implies_a_eq_plus_minus_two_l1819_181963


namespace NUMINAMATH_CALUDE_car_sale_profit_percentage_l1819_181966

/-- Calculates the profit percentage on a car sale given specific conditions --/
theorem car_sale_profit_percentage (P : ℝ) : 
  let discount_rate : ℝ := 0.1
  let discounted_price : ℝ := P * (1 - discount_rate)
  let first_year_expense_rate : ℝ := 0.05
  let second_year_expense_rate : ℝ := 0.04
  let third_year_expense_rate : ℝ := 0.03
  let selling_price_increase_rate : ℝ := 0.8
  
  let first_year_value : ℝ := discounted_price * (1 + first_year_expense_rate)
  let second_year_value : ℝ := first_year_value * (1 + second_year_expense_rate)
  let third_year_value : ℝ := second_year_value * (1 + third_year_expense_rate)
  
  let selling_price : ℝ := discounted_price * (1 + selling_price_increase_rate)
  let profit : ℝ := selling_price - P
  let profit_percentage : ℝ := (profit / P) * 100
  
  profit_percentage = 62 := by sorry

end NUMINAMATH_CALUDE_car_sale_profit_percentage_l1819_181966


namespace NUMINAMATH_CALUDE_intersection_P_Q_l1819_181914

def P : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def Q : Set ℝ := {1, 2, 3, 4}

theorem intersection_P_Q : P ∩ Q = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l1819_181914


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1819_181935

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  a 1 = 2 ∧
  a 2 + a 3 = 13

/-- The sum of the 4th, 5th, and 6th terms equals 42 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 4 + a 5 + a 6 = 42 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1819_181935


namespace NUMINAMATH_CALUDE_constant_value_proof_l1819_181979

theorem constant_value_proof :
  ∀ (t : ℝ) (constant : ℝ),
    let x := 1 - 2 * t
    let y := constant * t - 2
    (t = 0.75 → x = y) →
    constant = 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_proof_l1819_181979


namespace NUMINAMATH_CALUDE_system_solution_in_first_quadrant_l1819_181985

theorem system_solution_in_first_quadrant (c : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x - y = 2 ∧ c * x + y = 3) ↔ -1 < c ∧ c < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_in_first_quadrant_l1819_181985


namespace NUMINAMATH_CALUDE_liu_data_correct_l1819_181994

/-- Represents the agricultural data for the Li and Liu families -/
structure FamilyData where
  li_land : ℕ
  li_yield : ℕ
  liu_land_diff : ℕ
  total_production : ℕ

/-- Calculates the Liu family's total production and yield difference -/
def calculate_liu_data (data : FamilyData) : ℕ × ℕ :=
  let liu_land := data.li_land - data.liu_land_diff
  let liu_production := data.total_production
  let liu_yield := liu_production / liu_land
  let li_yield := data.li_yield
  let yield_diff := liu_yield - li_yield
  (liu_production, yield_diff)

/-- Theorem stating the correctness of the calculation -/
theorem liu_data_correct (data : FamilyData) 
  (h1 : data.li_land = 100)
  (h2 : data.li_yield = 600)
  (h3 : data.liu_land_diff = 20)
  (h4 : data.total_production = data.li_land * data.li_yield) :
  calculate_liu_data data = (6000, 15) := by
  sorry

#eval calculate_liu_data ⟨100, 600, 20, 60000⟩

end NUMINAMATH_CALUDE_liu_data_correct_l1819_181994


namespace NUMINAMATH_CALUDE_whistle_solution_l1819_181958

/-- The number of whistles Sean, Charles, and Alex have. -/
def whistle_problem (W_Sean W_Charles W_Alex : ℕ) : Prop :=
  W_Sean = 2483 ∧ 
  W_Charles = W_Sean - 463 ∧
  W_Alex = W_Charles - 131

theorem whistle_solution :
  ∀ W_Sean W_Charles W_Alex : ℕ,
  whistle_problem W_Sean W_Charles W_Alex →
  W_Charles = 2020 ∧ 
  W_Alex = 1889 ∧
  W_Sean + W_Charles + W_Alex = 6392 :=
by
  sorry

#check whistle_solution

end NUMINAMATH_CALUDE_whistle_solution_l1819_181958


namespace NUMINAMATH_CALUDE_min_S_19_l1819_181900

/-- An arithmetic sequence with its sum properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2
  arithmetic : ∀ n m, a (n + m) - a n = m * (a 2 - a 1)

/-- The minimum value of S_19 given the conditions -/
theorem min_S_19 (seq : ArithmeticSequence) 
  (h1 : seq.S 8 ≤ 6) (h2 : seq.S 11 ≥ 27) : 
  seq.S 19 ≥ 133 := by
  sorry

#check min_S_19

end NUMINAMATH_CALUDE_min_S_19_l1819_181900


namespace NUMINAMATH_CALUDE_relationship_abc_l1819_181928

theorem relationship_abc : ∀ (a b c : ℕ),
  a = 2^12 → b = 3^8 → c = 7^4 → b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l1819_181928


namespace NUMINAMATH_CALUDE_equation_solution_l1819_181932

theorem equation_solution : ∃ x : ℝ, (10 - 2 * x = 14) ∧ (x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1819_181932


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_four_and_six_l1819_181943

/-- Represents a repeating decimal with a single digit repetend -/
def RepeatingDecimal (n : ℕ) : ℚ := n / 9

theorem sum_of_repeating_decimals_four_and_six :
  RepeatingDecimal 4 + RepeatingDecimal 6 = 10 / 9 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_four_and_six_l1819_181943


namespace NUMINAMATH_CALUDE_tree_height_calculation_l1819_181938

/-- Given the height and shadow length of a person and the shadow length of a tree,
    calculate the height of the tree using the principle of similar triangles. -/
theorem tree_height_calculation (person_height person_shadow tree_shadow : ℝ)
  (h_person_height : person_height = 1.6)
  (h_person_shadow : person_shadow = 0.8)
  (h_tree_shadow : tree_shadow = 4.8)
  (h_positive : person_height > 0 ∧ person_shadow > 0 ∧ tree_shadow > 0) :
  (person_height / person_shadow) * tree_shadow = 9.6 :=
by
  sorry

#check tree_height_calculation

end NUMINAMATH_CALUDE_tree_height_calculation_l1819_181938


namespace NUMINAMATH_CALUDE_random_simulation_approximates_actual_probability_l1819_181956

/-- Random simulation method for estimating probabilities -/
def RandomSimulationMethod : Type := Unit

/-- Estimated probability from random simulation -/
def estimated_probability (method : RandomSimulationMethod) : ℝ := sorry

/-- Actual probability of the event -/
def actual_probability : ℝ := sorry

/-- Definition of approximation -/
def is_approximation (x y : ℝ) : Prop := sorry

theorem random_simulation_approximates_actual_probability 
  (method : RandomSimulationMethod) : 
  is_approximation (estimated_probability method) actual_probability := by
  sorry

end NUMINAMATH_CALUDE_random_simulation_approximates_actual_probability_l1819_181956


namespace NUMINAMATH_CALUDE_compare_expressions_l1819_181959

theorem compare_expressions (x : ℝ) (h : x ≥ 0) :
  (x > 2 → 5*x^2 - 1 > 3*x^2 + 3*x + 1) ∧
  (x = 2 → 5*x^2 - 1 = 3*x^2 + 3*x + 1) ∧
  (0 ≤ x ∧ x < 2 → 5*x^2 - 1 < 3*x^2 + 3*x + 1) :=
by sorry

end NUMINAMATH_CALUDE_compare_expressions_l1819_181959


namespace NUMINAMATH_CALUDE_first_terrific_tuesday_l1819_181987

/-- Represents a date with a day and a month -/
structure Date where
  day : ℕ
  month : ℕ

/-- Represents a day of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the number of days in a given month (assuming non-leap year) -/
def daysInMonth (month : ℕ) : ℕ :=
  if month == 2 then 28 else if month ∈ [4, 6, 9, 11] then 30 else 31

/-- Returns the weekday of a given date, assuming February 1 is a Tuesday -/
def weekdayOfDate (d : Date) : Weekday :=
  sorry

/-- Returns true if the given date is a Tuesday -/
def isTuesday (d : Date) : Prop :=
  weekdayOfDate d = Weekday.Tuesday

/-- Returns true if the given date is a Terrific Tuesday (5th Tuesday of the month) -/
def isTerrificTuesday (d : Date) : Prop :=
  isTuesday d ∧ d.day > 28

/-- The main theorem: The first Terrific Tuesday after February 1 is March 29 -/
theorem first_terrific_tuesday : 
  ∃ (d : Date), d.month = 3 ∧ d.day = 29 ∧ isTerrificTuesday d ∧
  ∀ (d' : Date), (d'.month < 3 ∨ (d'.month = 3 ∧ d'.day < 29)) → ¬isTerrificTuesday d' :=
  sorry

end NUMINAMATH_CALUDE_first_terrific_tuesday_l1819_181987


namespace NUMINAMATH_CALUDE_division_problem_l1819_181921

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2 / 5) : 
  c / a = 5 / 6 := by sorry

end NUMINAMATH_CALUDE_division_problem_l1819_181921


namespace NUMINAMATH_CALUDE_ab_value_l1819_181990

theorem ab_value (a b : ℝ) (h1 : |a| = 3) (h2 : |b - 2| = 9) (h3 : a + b > 0) :
  ab = 33 ∨ ab = -33 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1819_181990


namespace NUMINAMATH_CALUDE_system_solution_l1819_181974

theorem system_solution (a b c x y z : ℝ) : 
  (x + a * y + a^2 * z = a^3) →
  (x + b * y + b^2 * z = b^3) →
  (x + c * y + c^2 * z = c^3) →
  (x = a * b * c ∧ y = -(a * b + b * c + c * a) ∧ z = a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1819_181974


namespace NUMINAMATH_CALUDE_expansion_properties_l1819_181982

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of the kth term in the expansion of (x + 1/(2√x))^n -/
def coefficient (n k : ℕ) : ℚ := (1 / 2^k : ℚ) * binomial n k

/-- The expansion of (x + 1/(2√x))^n has its first three coefficients in arithmetic sequence -/
def first_three_in_arithmetic_sequence (n : ℕ) : Prop :=
  coefficient n 0 + coefficient n 2 = 2 * coefficient n 1

/-- The kth term has the maximum coefficient in the expansion -/
def max_coefficient (n k : ℕ) : Prop :=
  ∀ i, i ≠ k → coefficient n k ≥ coefficient n i

theorem expansion_properties :
  ∃ n : ℕ,
    first_three_in_arithmetic_sequence n ∧
    max_coefficient n 2 ∧
    max_coefficient n 3 ∧
    ∀ k, k ≠ 2 ∧ k ≠ 3 → ¬(max_coefficient n k) :=
  sorry

end NUMINAMATH_CALUDE_expansion_properties_l1819_181982


namespace NUMINAMATH_CALUDE_points_on_line_or_circle_l1819_181913

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in a 2D plane -/
structure Circle2D where
  center : Point2D
  radius : ℝ

/-- Function to check if three points are collinear -/
def areCollinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.y - p1.y) * (p3.x - p2.x) = (p3.y - p2.y) * (p2.x - p1.x)

/-- Function to generate points based on the described process -/
def generatePoints (p1 p2 p3 : Point2D) : Set Point2D :=
  sorry

/-- The main theorem -/
theorem points_on_line_or_circle (p1 p2 p3 : Point2D) :
  ∃ (l : Line2D) (c : Circle2D), 
    (areCollinear p1 p2 p3 ∧ generatePoints p1 p2 p3 ⊆ {p | p.x * l.a + p.y * l.b + l.c = 0}) ∨
    (¬areCollinear p1 p2 p3 ∧ generatePoints p1 p2 p3 ⊆ {p | (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2}) :=
  sorry

end NUMINAMATH_CALUDE_points_on_line_or_circle_l1819_181913


namespace NUMINAMATH_CALUDE_negative_a_squared_times_a_cubed_l1819_181996

theorem negative_a_squared_times_a_cubed (a : ℝ) : (-a)^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_squared_times_a_cubed_l1819_181996


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1819_181930

/-- The equation of a hyperbola with given parameters -/
theorem hyperbola_equation (a c : ℝ) (h1 : a > 0) (h2 : c > a) :
  let e := c / a
  let b := Real.sqrt (c^2 - a^2)
  ∀ x y : ℝ, 2 * a = 8 → e = 5/4 →
    (x^2 / a^2 - y^2 / b^2 = 1) ↔ (x^2 / 16 - y^2 / 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1819_181930


namespace NUMINAMATH_CALUDE_shoe_rebate_problem_l1819_181911

/-- Calculates the total rebate and quantity discount for a set of shoe purchases --/
def calculate_rebate_and_discount (prices : List ℝ) (rebate_percentages : List ℝ) 
  (discount_threshold_1 : ℝ) (discount_threshold_2 : ℝ) 
  (discount_rate_1 : ℝ) (discount_rate_2 : ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correct rebate and discount for the given problem --/
theorem shoe_rebate_problem :
  let prices := [28, 35, 40, 45, 50]
  let rebate_percentages := [10, 12, 15, 18, 20]
  let discount_threshold_1 := 200
  let discount_threshold_2 := 250
  let discount_rate_1 := 5
  let discount_rate_2 := 7
  let (total_rebate, quantity_discount) := 
    calculate_rebate_and_discount prices rebate_percentages 
      discount_threshold_1 discount_threshold_2 
      discount_rate_1 discount_rate_2
  total_rebate = 31.1 ∧ quantity_discount = 0 := by
  sorry

end NUMINAMATH_CALUDE_shoe_rebate_problem_l1819_181911


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l1819_181980

/-- For an infinite geometric series with first term a and common ratio r,
    if the sum of the series is 64 times the sum of the series with the first four terms removed,
    then r = 1/2 -/
theorem geometric_series_ratio (a r : ℝ) (h : r ≠ 1) :
  (a / (1 - r)) = 64 * (a * r^4 / (1 - r)) → r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l1819_181980


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l1819_181910

/-- The equation of the tangent line to the parabola y = x² that is parallel to y = 2x -/
theorem tangent_line_to_parabola (x y : ℝ) : 
  (∀ t, y = t^2 → (2 * t = 2 → x = t ∧ y = t^2)) →
  (2 * x - y - 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l1819_181910


namespace NUMINAMATH_CALUDE_right_triangle_sets_l1819_181906

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_sets :
  (is_right_triangle 1 2 (Real.sqrt 5)) ∧
  (is_right_triangle (Real.sqrt 2) (Real.sqrt 2) 2) ∧
  (is_right_triangle 13 12 5) ∧
  ¬(is_right_triangle 1 3 (Real.sqrt 7)) := by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l1819_181906


namespace NUMINAMATH_CALUDE_no_primitive_root_for_multiple_odd_primes_l1819_181969

theorem no_primitive_root_for_multiple_odd_primes (n : ℕ) 
  (h1 : ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p ≠ q ∧ Odd p ∧ Odd q ∧ n % p = 0 ∧ n % q = 0) : 
  ¬ ∃ a : ℕ, IsPrimitiveRoot a n :=
sorry

end NUMINAMATH_CALUDE_no_primitive_root_for_multiple_odd_primes_l1819_181969


namespace NUMINAMATH_CALUDE_tony_haircut_distance_l1819_181999

theorem tony_haircut_distance (total_distance halfway_distance groceries_distance doctor_distance : ℕ)
  (h1 : total_distance = 2 * halfway_distance)
  (h2 : halfway_distance = 15)
  (h3 : groceries_distance = 10)
  (h4 : doctor_distance = 5) :
  total_distance - (groceries_distance + doctor_distance) = 15 := by
  sorry

end NUMINAMATH_CALUDE_tony_haircut_distance_l1819_181999


namespace NUMINAMATH_CALUDE_m_range_l1819_181984

theorem m_range (m : ℝ) : 
  (∃ x₀ : ℝ, x₀^2 + m ≤ 0) ∧ 
  (∀ x : ℝ, x^2 + m*x + 1 > 0) ∧ 
  ¬((¬∃ x₀ : ℝ, x₀^2 + m ≤ 0) ∨ (∀ x : ℝ, x^2 + m*x + 1 > 0)) → 
  m ≤ -2 := by
sorry

end NUMINAMATH_CALUDE_m_range_l1819_181984


namespace NUMINAMATH_CALUDE_two_person_subcommittees_from_eight_l1819_181912

theorem two_person_subcommittees_from_eight (n : ℕ) (k : ℕ) : 
  n = 8 → k = 2 → Nat.choose n k = 28 := by
  sorry

end NUMINAMATH_CALUDE_two_person_subcommittees_from_eight_l1819_181912


namespace NUMINAMATH_CALUDE_inequality_proof_l1819_181993

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a + b + c + 3) / 4 ≥ 1 / (a + b) + 1 / (b + c) + 1 / (c + a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1819_181993


namespace NUMINAMATH_CALUDE_complex_vector_sum_l1819_181988

theorem complex_vector_sum (z₁ z₂ z₃ : ℂ) (x y : ℝ) 
  (h₁ : z₁ = -1 + I)
  (h₂ : z₂ = 1 + I)
  (h₃ : z₃ = 1 + 4*I)
  (h₄ : z₃ = x • z₁ + y • z₂) : 
  x + y = 4 := by sorry

end NUMINAMATH_CALUDE_complex_vector_sum_l1819_181988


namespace NUMINAMATH_CALUDE_green_silk_calculation_l1819_181964

/-- The number of yards of silk dyed for an order -/
def total_yards : ℕ := 111421

/-- The number of yards of silk dyed pink -/
def pink_yards : ℕ := 49500

/-- The number of yards of silk dyed green -/
def green_yards : ℕ := total_yards - pink_yards

theorem green_silk_calculation : green_yards = 61921 := by
  sorry

end NUMINAMATH_CALUDE_green_silk_calculation_l1819_181964


namespace NUMINAMATH_CALUDE_trigonometric_expression_value_l1819_181960

theorem trigonometric_expression_value :
  (Real.sin (330 * π / 180) * Real.tan (-13 * π / 3)) /
  (Real.cos (-19 * π / 6) * Real.cos (690 * π / 180)) = -2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_value_l1819_181960


namespace NUMINAMATH_CALUDE_min_distance_circle_to_line_l1819_181948

theorem min_distance_circle_to_line : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | 3*x + 4*y + 15 = 0}
  ∃ d : ℝ, d = 2 ∧ 
    ∀ p ∈ circle, ∀ q ∈ line, 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d ∧
      ∃ p' ∈ circle, ∃ q' ∈ line, 
        Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) = d :=
by
  sorry


end NUMINAMATH_CALUDE_min_distance_circle_to_line_l1819_181948


namespace NUMINAMATH_CALUDE_root_sum_theorem_l1819_181992

theorem root_sum_theorem (a b c d r : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (r - a) * (r - b) * (r - c) * (r - d) = 4 →
  4 * r = a + b + c + d := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l1819_181992


namespace NUMINAMATH_CALUDE_certain_number_problem_l1819_181936

theorem certain_number_problem (y : ℝ) : 
  (0.25 * 900 = 0.15 * y - 15) → y = 1600 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1819_181936


namespace NUMINAMATH_CALUDE_unique_prime_triplet_l1819_181978

theorem unique_prime_triplet :
  ∀ p q r : ℕ,
    Prime p → Prime q → Prime r →
    p + q = r →
    ∃ n : ℕ, (r - p) * (q - p) - 27 * p = n^2 →
    p = 2 ∧ q = 29 ∧ r = 31 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_triplet_l1819_181978


namespace NUMINAMATH_CALUDE_hedgehog_strawberries_l1819_181967

theorem hedgehog_strawberries : 
  ∀ (num_hedgehogs num_baskets strawberries_per_basket : ℕ) 
    (remaining_fraction : ℚ),
  num_hedgehogs = 2 →
  num_baskets = 3 →
  strawberries_per_basket = 900 →
  remaining_fraction = 2 / 9 →
  ∃ (strawberries_eaten_per_hedgehog : ℕ),
    strawberries_eaten_per_hedgehog = 1050 ∧
    (num_baskets * strawberries_per_basket) * (1 - remaining_fraction) = 
      num_hedgehogs * strawberries_eaten_per_hedgehog :=
by sorry

end NUMINAMATH_CALUDE_hedgehog_strawberries_l1819_181967


namespace NUMINAMATH_CALUDE_cosine_rational_values_l1819_181983

theorem cosine_rational_values (α : ℚ) (h : ∃ (r : ℚ), r = Real.cos (α * Real.pi)) :
  Real.cos (α * Real.pi) = 0 ∨
  Real.cos (α * Real.pi) = 1 ∨
  Real.cos (α * Real.pi) = -1 ∨
  Real.cos (α * Real.pi) = (1/2) ∨
  Real.cos (α * Real.pi) = -(1/2) :=
by sorry

end NUMINAMATH_CALUDE_cosine_rational_values_l1819_181983


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_l1819_181968

theorem smallest_integer_with_remainder (n : ℕ) : 
  (n > 1) →
  (n % 3 = 2) →
  (n % 5 = 2) →
  (n % 7 = 2) →
  (∀ m : ℕ, m > 1 → m % 3 = 2 → m % 5 = 2 → m % 7 = 2 → n ≤ m) →
  (n = 107 ∧ 90 < n ∧ n < 119) := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_l1819_181968


namespace NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_condition_l1819_181918

/-- The function f(x) = √x - a ln(x+1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt x - a * Real.log (x + 1)

/-- The derivative of f with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 / (2 * Real.sqrt x) - a / (x + 1)

theorem tangent_line_at_one (a : ℝ) :
  a = -1 → (fun x => x + Real.log 2) = fun x => f (-1) 1 + f_deriv (-1) 1 * (x - 1) := by sorry

theorem monotonicity_condition (a : ℝ) :
  a ≤ 1 → ∀ x y, 0 ≤ x ∧ x < y → f a x < f a y := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_condition_l1819_181918


namespace NUMINAMATH_CALUDE_fibonacci_properties_l1819_181989

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_properties :
  (∃ f : ℕ → ℕ → ℕ, ∀ k : ℕ, ∃ p q : ℕ, p > k ∧ q > k ∧ ∃ m : ℕ, (fibonacci p * fibonacci q - 1 = m^2)) ∧
  (∃ g : ℕ → ℕ → ℕ × ℕ, ∀ k : ℕ, ∃ m n : ℕ, m > k ∧ n > k ∧ 
    (fibonacci m ∣ fibonacci n^2 + 1) ∧ (fibonacci n ∣ fibonacci m^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_properties_l1819_181989


namespace NUMINAMATH_CALUDE_wheel_probability_l1819_181903

theorem wheel_probability (p_D p_E p_F : ℚ) : 
  p_D = 2/5 → p_E = 1/3 → p_D + p_E + p_F = 1 → p_F = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_wheel_probability_l1819_181903


namespace NUMINAMATH_CALUDE_always_quadratic_l1819_181954

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation (k^2+1)x^2-2x+1=0 is always a quadratic equation -/
theorem always_quadratic (k : ℝ) : 
  is_quadratic_equation (λ x => (k^2 + 1) * x^2 - 2 * x + 1) :=
sorry

end NUMINAMATH_CALUDE_always_quadratic_l1819_181954


namespace NUMINAMATH_CALUDE_relationship_between_x_squared_ax_bx_l1819_181946

theorem relationship_between_x_squared_ax_bx
  (x a b : ℝ)
  (h1 : x < a)
  (h2 : a < 0)
  (h3 : b > 0) :
  x^2 > a*x ∧ a*x > b*x :=
by sorry

end NUMINAMATH_CALUDE_relationship_between_x_squared_ax_bx_l1819_181946


namespace NUMINAMATH_CALUDE_quilt_sewing_percentage_l1819_181909

theorem quilt_sewing_percentage (total_squares : ℕ) (squares_left : ℕ) : 
  total_squares = 32 → squares_left = 24 → 
  (total_squares - squares_left : ℚ) / total_squares * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_quilt_sewing_percentage_l1819_181909


namespace NUMINAMATH_CALUDE_min_value_f_and_sum_squares_l1819_181947

def f (x : ℝ) : ℝ := |x - 4| + |x - 3|

theorem min_value_f_and_sum_squares :
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ (∃ (y : ℝ), f y = m) ∧ m = 1) ∧
  (∀ (a b c : ℝ), a + 2*b + 3*c = 1 → a^2 + b^2 + c^2 ≥ 1/14) ∧
  (∃ (a b c : ℝ), a + 2*b + 3*c = 1 ∧ a^2 + b^2 + c^2 = 1/14) := by
  sorry

end NUMINAMATH_CALUDE_min_value_f_and_sum_squares_l1819_181947


namespace NUMINAMATH_CALUDE_log_equation_implies_sum_l1819_181929

theorem log_equation_implies_sum (x y : ℝ) 
  (h1 : x > 1) (h2 : y > 1) 
  (h3 : (Real.log x / Real.log 4)^3 + (Real.log y / Real.log 5)^3 + 6 = 
        6 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) : 
  x^Real.sqrt 3 + y^Real.sqrt 3 = 189 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_implies_sum_l1819_181929


namespace NUMINAMATH_CALUDE_pirate_treasure_sum_l1819_181916

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The value of sapphires in base 7 -/
def sapphires : List Nat := [2, 3, 5, 6]

/-- The value of silverware in base 7 -/
def silverware : List Nat := [0, 5, 6, 1]

/-- The value of spices in base 7 -/
def spices : List Nat := [0, 5, 2]

/-- The theorem stating the sum of the treasures in base 10 -/
theorem pirate_treasure_sum :
  base7ToBase10 sapphires + base7ToBase10 silverware + base7ToBase10 spices = 3131 := by
  sorry


end NUMINAMATH_CALUDE_pirate_treasure_sum_l1819_181916


namespace NUMINAMATH_CALUDE_inequality_group_solution_set_l1819_181902

theorem inequality_group_solution_set :
  ∀ x : ℝ, (x > -3 ∧ x < 5) ↔ (-3 < x ∧ x < 5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_group_solution_set_l1819_181902


namespace NUMINAMATH_CALUDE_booklet_cost_l1819_181951

theorem booklet_cost (b : ℝ) : 
  (10 * b < 15) → (12 * b > 17) → b = 1.42 := by
  sorry

end NUMINAMATH_CALUDE_booklet_cost_l1819_181951


namespace NUMINAMATH_CALUDE_yearly_music_expenditure_l1819_181952

def hours_per_month : ℕ := 20
def minutes_per_song : ℕ := 3
def price_per_song : ℚ := 1/2
def months_per_year : ℕ := 12

def yearly_music_cost : ℚ :=
  (hours_per_month * 60 / minutes_per_song) * price_per_song * months_per_year

theorem yearly_music_expenditure :
  yearly_music_cost = 2400 := by
  sorry

end NUMINAMATH_CALUDE_yearly_music_expenditure_l1819_181952


namespace NUMINAMATH_CALUDE_share_ratio_problem_l1819_181975

theorem share_ratio_problem (total : ℕ) (john_share : ℕ) :
  total = 4800 →
  john_share = 1600 →
  ∃ (jose_share binoy_share : ℕ),
    total = john_share + jose_share + binoy_share ∧
    2 * jose_share = 4 * john_share ∧
    3 * jose_share = 6 * john_share ∧
    binoy_share = 3 * john_share :=
by sorry

end NUMINAMATH_CALUDE_share_ratio_problem_l1819_181975


namespace NUMINAMATH_CALUDE_grasshopper_movement_l1819_181925

/-- Represents the possible jump distances of the grasshopper -/
inductive Jump
| large : Jump  -- 36 cm jump
| small : Jump  -- 14 cm jump

/-- Represents the direction of the jump -/
inductive Direction
| left : Direction
| right : Direction

/-- Represents a single jump of the grasshopper -/
structure GrasshopperJump :=
  (distance : Jump)
  (direction : Direction)

/-- The distance covered by a single jump -/
def jumpDistance (j : GrasshopperJump) : ℤ :=
  match j.distance, j.direction with
  | Jump.large, Direction.right => 36
  | Jump.large, Direction.left  => -36
  | Jump.small, Direction.right => 14
  | Jump.small, Direction.left  => -14

/-- The total distance covered by a sequence of jumps -/
def totalDistance (jumps : List GrasshopperJump) : ℤ :=
  jumps.foldl (fun acc j => acc + jumpDistance j) 0

/-- Predicate to check if a distance is reachable by the grasshopper -/
def isReachable (d : ℤ) : Prop :=
  ∃ (jumps : List GrasshopperJump), totalDistance jumps = d

theorem grasshopper_movement :
  (¬ isReachable 3) ∧ (isReachable 2) ∧ (isReachable 1234) := by sorry

end NUMINAMATH_CALUDE_grasshopper_movement_l1819_181925


namespace NUMINAMATH_CALUDE_monotone_cubic_implies_nonneg_a_l1819_181907

/-- A function f : ℝ → ℝ is monotonically increasing if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The cubic function with parameter a -/
def f (a : ℝ) : ℝ → ℝ := λ x => x^3 + a*x

theorem monotone_cubic_implies_nonneg_a :
  ∀ a : ℝ, MonotonicallyIncreasing (f a) → a ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_monotone_cubic_implies_nonneg_a_l1819_181907


namespace NUMINAMATH_CALUDE_last_digit_power_of_two_divisibility_l1819_181939

theorem last_digit_power_of_two_divisibility (k : ℕ) (N a A : ℕ) :
  k ≥ 3 →
  N = 2^k →
  a = N % 10 →
  A * 10 + a = N →
  6 ∣ a * A :=
by sorry

end NUMINAMATH_CALUDE_last_digit_power_of_two_divisibility_l1819_181939


namespace NUMINAMATH_CALUDE_eugene_payment_l1819_181923

def tshirt_cost : ℕ := 20
def pants_cost : ℕ := 80
def shoes_cost : ℕ := 150
def discount_rate : ℚ := 1/10

def tshirt_quantity : ℕ := 4
def pants_quantity : ℕ := 3
def shoes_quantity : ℕ := 2

def total_cost : ℕ := tshirt_cost * tshirt_quantity + pants_cost * pants_quantity + shoes_cost * shoes_quantity

def discounted_cost : ℚ := (1 - discount_rate) * total_cost

theorem eugene_payment : discounted_cost = 558 := by
  sorry

end NUMINAMATH_CALUDE_eugene_payment_l1819_181923


namespace NUMINAMATH_CALUDE_angle_after_folding_is_60_degrees_l1819_181949

/-- An isosceles right triangle -/
structure IsoscelesRightTriangle where
  /-- The two equal sides of the triangle -/
  leg : ℝ
  /-- The hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- The condition that it's a right triangle -/
  right_angle : hypotenuse^2 = 2 * leg^2

/-- The angle between the legs after folding an isosceles right triangle along its height to the hypotenuse -/
def angle_after_folding (t : IsoscelesRightTriangle) : ℝ := sorry

/-- Theorem stating that the angle between the legs after folding is 60° -/
theorem angle_after_folding_is_60_degrees (t : IsoscelesRightTriangle) :
  angle_after_folding t = 60 * (π / 180) := by sorry

end NUMINAMATH_CALUDE_angle_after_folding_is_60_degrees_l1819_181949


namespace NUMINAMATH_CALUDE_power_function_positive_l1819_181926

theorem power_function_positive (α : ℚ) (x : ℝ) (h : x > 0) : x ^ (α : ℝ) > 0 := by
  sorry

end NUMINAMATH_CALUDE_power_function_positive_l1819_181926


namespace NUMINAMATH_CALUDE_parabola_points_condition_l1819_181997

/-- The parabola equation -/
def parabola (x y k : ℝ) : Prop := y = -2 * (x - 1)^2 + k

theorem parabola_points_condition (m y₁ y₂ k : ℝ) :
  parabola (m - 1) y₁ k →
  parabola m y₂ k →
  y₁ > y₂ →
  m > 3/2 := by sorry

end NUMINAMATH_CALUDE_parabola_points_condition_l1819_181997


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l1819_181981

-- Define the number of DVDs and prices for each store
def store_a_dvds : ℕ := 8
def store_a_price : ℚ := 15
def store_b_dvds : ℕ := 12
def store_b_price : ℚ := 12
def online_dvds : ℕ := 5
def online_price : ℚ := 16.99

-- Define the discount percentage
def discount_percent : ℚ := 15

-- Define the total cost function
def total_cost (store_a_dvds store_b_dvds online_dvds : ℕ) 
               (store_a_price store_b_price online_price : ℚ) 
               (discount_percent : ℚ) : ℚ :=
  let physical_store_cost := store_a_dvds * store_a_price + store_b_dvds * store_b_price
  let online_store_cost := online_dvds * online_price
  let discount := physical_store_cost * (discount_percent / 100)
  (physical_store_cost - discount) + online_store_cost

-- Theorem statement
theorem total_cost_is_correct : 
  total_cost store_a_dvds store_b_dvds online_dvds 
             store_a_price store_b_price online_price 
             discount_percent = 309.35 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l1819_181981


namespace NUMINAMATH_CALUDE_a_3_equals_negative_8_l1819_181908

/-- The sum of the first n terms of a geometric sequence -/
def S (n : ℕ) (x : ℝ) : ℝ := (x^2 + 3*x)*2^n - x + 1

/-- The n-th term of the geometric sequence -/
def a (n : ℕ) (x : ℝ) : ℝ :=
  if n = 1 then S 1 x
  else S n x - S (n-1) x

/-- The common ratio of the geometric sequence -/
def q : ℝ := 2

/-- The value of x that satisfies the given condition -/
def x : ℝ := -1

theorem a_3_equals_negative_8 : a 3 x = -8 := by sorry

end NUMINAMATH_CALUDE_a_3_equals_negative_8_l1819_181908


namespace NUMINAMATH_CALUDE_prism_volume_l1819_181945

/-- A right triangular prism with given base area and lateral face areas has volume 12 -/
theorem prism_volume (base_area : ℝ) (lateral_area1 lateral_area2 lateral_area3 : ℝ) 
  (h_base : base_area = 4)
  (h_lateral1 : lateral_area1 = 9)
  (h_lateral2 : lateral_area2 = 10)
  (h_lateral3 : lateral_area3 = 17) :
  base_area * (lateral_area1 / base_area.sqrt) = 12 :=
by sorry

end NUMINAMATH_CALUDE_prism_volume_l1819_181945


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1819_181927

theorem inequality_solution_set (x : ℝ) :
  (x^2 + 1) / ((x - 3) * (x + 2)) ≥ 0 ↔ x ≤ -2 ∨ x ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1819_181927
