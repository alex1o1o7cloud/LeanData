import Mathlib

namespace ratio_a_to_c_l321_32164

theorem ratio_a_to_c (a b c d : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 5) :
  a / c = 75 / 16 := by
sorry

end ratio_a_to_c_l321_32164


namespace magic_card_profit_theorem_l321_32117

/-- Calculates the profit from selling a Magic card that increases in value -/
def magic_card_profit (initial_price : ℝ) (value_multiplier : ℝ) : ℝ :=
  initial_price * value_multiplier - initial_price

/-- Theorem: The profit from selling a Magic card that triples in value from $100 is $200 -/
theorem magic_card_profit_theorem :
  magic_card_profit 100 3 = 200 := by
  sorry

end magic_card_profit_theorem_l321_32117


namespace quadratic_equation_solution_l321_32144

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 2 * x - 1
  f (-1/3) = 0 ∧ f 1 = 0 :=
by sorry

end quadratic_equation_solution_l321_32144


namespace pedestrian_speed_problem_l321_32179

/-- The problem of two pedestrians traveling between points A and B -/
theorem pedestrian_speed_problem (x : ℝ) :
  x > 0 →  -- The speed must be positive
  (11 / (x + 1.5) + 8 / x ≥ 12 / (x + 2) + 2) →  -- The inequality from the problem
  x ≤ 4 := by
  sorry

end pedestrian_speed_problem_l321_32179


namespace toaster_tax_rate_l321_32122

/-- Calculates the mandatory state tax rate for a toaster purchase. -/
theorem toaster_tax_rate (msrp : ℝ) (total_paid : ℝ) (insurance_rate : ℝ) : 
  msrp = 30 →
  total_paid = 54 →
  insurance_rate = 0.2 →
  (total_paid - msrp * (1 + insurance_rate)) / (msrp * (1 + insurance_rate)) = 0.5 := by
  sorry

#check toaster_tax_rate

end toaster_tax_rate_l321_32122


namespace updated_mean_after_decrement_l321_32155

theorem updated_mean_after_decrement (n : ℕ) (original_mean : ℚ) (decrement : ℚ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 9 →
  (n : ℚ) * original_mean - n * decrement = n * 191 := by
  sorry

end updated_mean_after_decrement_l321_32155


namespace optimal_fence_dimensions_l321_32188

/-- Represents the dimensions of a rectangular plot -/
structure PlotDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the total fence length for a given plot -/
def totalFenceLength (d : PlotDimensions) : ℝ :=
  3 * d.length + 2 * d.width

/-- Theorem stating the optimal dimensions for minimal fence length -/
theorem optimal_fence_dimensions :
  ∃ (d : PlotDimensions),
    d.length * d.width = 294 ∧
    d.length = 14 ∧
    d.width = 21 ∧
    ∀ (d' : PlotDimensions),
      d'.length * d'.width = 294 →
      totalFenceLength d ≤ totalFenceLength d' := by
  sorry

end optimal_fence_dimensions_l321_32188


namespace simplified_fraction_equals_ten_l321_32185

theorem simplified_fraction_equals_ten (x y z : ℝ) 
  (hx : x = 3) (hy : y = 2) (hz : z = 4) : 
  (15 * x^2 * y^4 * z^2) / (9 * x * y^3 * z) = 10 := by
  sorry

end simplified_fraction_equals_ten_l321_32185


namespace pyramid_volume_is_four_thirds_l321_32180

-- Define the cube IJKLMNO
structure Cube where
  volume : ℝ

-- Define the pyramid IJMO
structure Pyramid where
  base : Cube

-- Define the volume of the pyramid
def pyramid_volume (p : Pyramid) : ℝ := sorry

-- Theorem statement
theorem pyramid_volume_is_four_thirds (c : Cube) (p : Pyramid) 
  (h1 : c.volume = 8) 
  (h2 : p.base = c) : 
  pyramid_volume p = 4/3 := by sorry

end pyramid_volume_is_four_thirds_l321_32180


namespace angle_E_measure_l321_32116

-- Define the heptagon and its angles
structure Heptagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ
  G : ℝ

-- Define the properties of the heptagon
def is_valid_heptagon (h : Heptagon) : Prop :=
  h.A > 0 ∧ h.B > 0 ∧ h.C > 0 ∧ h.D > 0 ∧ h.E > 0 ∧ h.F > 0 ∧ h.G > 0 ∧
  h.A + h.B + h.C + h.D + h.E + h.F + h.G = 900

-- Define the conditions given in the problem
def satisfies_conditions (h : Heptagon) : Prop :=
  h.A = h.B ∧ h.A = h.C ∧ h.A = h.D ∧  -- A, B, C, D are congruent
  h.E = h.F ∧                          -- E and F are congruent
  h.A = h.E - 50 ∧                     -- A is 50° less than E
  h.G = 180 - h.E                      -- G is supplementary to E

-- The theorem to prove
theorem angle_E_measure (h : Heptagon) 
  (hvalid : is_valid_heptagon h) 
  (hcond : satisfies_conditions h) : 
  h.E = 184 := by
  sorry  -- The proof would go here


end angle_E_measure_l321_32116


namespace events_not_independent_l321_32106

/- Define the sample space -/
def Ω : Type := Fin 10

/- Define the events A and B -/
def A : Set Ω := {ω : Ω | ω.val < 5}
def B : Set Ω := {ω : Ω | ω.val % 2 = 0}

/- Define the probability measure -/
def P : Set Ω → ℝ := sorry

/- State the theorem -/
theorem events_not_independent : ¬(P (A ∩ B) = P A * P B) := by sorry

end events_not_independent_l321_32106


namespace count_valid_pairs_l321_32178

-- Define ω as a complex number that is a nonreal root of z^4 = 1
def ω : ℂ := sorry

-- Define the property for the ordered pairs we're looking for
def validPair (a b : ℤ) : Prop :=
  Complex.abs (a • ω + b) = 1

-- State the theorem
theorem count_valid_pairs :
  ∃! (n : ℕ), ∃ (S : Finset (ℤ × ℤ)), 
    S.card = n ∧ 
    (∀ (p : ℤ × ℤ), p ∈ S ↔ validPair p.1 p.2) ∧
    n = 4 := by sorry

end count_valid_pairs_l321_32178


namespace trig_identity_l321_32128

theorem trig_identity :
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) =
  4 * Real.sin (10 * π / 180) / Real.sin (40 * π / 180) := by
  sorry

end trig_identity_l321_32128


namespace remainder_2345678901_mod_101_l321_32142

theorem remainder_2345678901_mod_101 : 2345678901 % 101 = 12 := by
  sorry

end remainder_2345678901_mod_101_l321_32142


namespace train_length_calculation_l321_32156

/-- Calculates the length of a train given its speed, tunnel length, and time to pass through the tunnel. -/
theorem train_length_calculation (train_speed : ℝ) (tunnel_length : ℝ) (passing_time : ℝ) :
  train_speed = 72 →
  tunnel_length = 1.7 →
  passing_time = 1.5 / 60 →
  (train_speed * passing_time) - tunnel_length = 0.1 := by
  sorry

end train_length_calculation_l321_32156


namespace allison_sewing_time_l321_32197

/-- The time it takes Al to sew dresses individually -/
def al_time : ℝ := 12

/-- The time Allison and Al work together -/
def joint_work_time : ℝ := 3

/-- The additional time Allison works alone after Al leaves -/
def allison_extra_time : ℝ := 3.75

/-- The time it takes Allison to sew dresses individually -/
def allison_time : ℝ := 9

theorem allison_sewing_time : 
  (joint_work_time / allison_time + joint_work_time / al_time + allison_extra_time / allison_time) = 1 := by
  sorry

#check allison_sewing_time

end allison_sewing_time_l321_32197


namespace tobys_remaining_amount_l321_32140

/-- Calculates the remaining amount for Toby after sharing with his brothers -/
theorem tobys_remaining_amount (initial_amount : ℕ) (num_brothers : ℕ) 
  (h1 : initial_amount = 343)
  (h2 : num_brothers = 2) : 
  initial_amount - num_brothers * (initial_amount / 7) = 245 := by
  sorry

#eval 343 - 2 * (343 / 7)  -- Expected output: 245

end tobys_remaining_amount_l321_32140


namespace sum_of_reciprocals_positive_l321_32135

theorem sum_of_reciprocals_positive (a b c d : ℝ) 
  (ha : |a| > 1) (hb : |b| > 1) (hc : |c| > 1) (hd : |d| > 1)
  (h_eq : a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0) :
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 := by
  sorry

end sum_of_reciprocals_positive_l321_32135


namespace equality_or_opposite_equality_l321_32182

theorem equality_or_opposite_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^2 + b^3/a = b^2 + a^3/b → a = b ∨ a = -b := by
  sorry

end equality_or_opposite_equality_l321_32182


namespace black_and_white_drawing_cost_l321_32127

/-- The cost of a black and white drawing -/
def black_and_white_cost : ℝ := 160

/-- The cost of a color drawing -/
def color_cost : ℝ := 240

/-- The size of the drawing -/
def drawing_size : ℕ × ℕ := (9, 13)

theorem black_and_white_drawing_cost :
  black_and_white_cost = 160 ∧
  color_cost = black_and_white_cost * 1.5 ∧
  color_cost = 240 := by
  sorry

end black_and_white_drawing_cost_l321_32127


namespace hat_knitting_time_l321_32119

/-- Represents the time (in hours) to knit various items --/
structure KnittingTimes where
  hat : ℝ
  scarf : ℝ
  mitten : ℝ
  sock : ℝ
  sweater : ℝ

/-- Calculates the total time to knit one set of clothes --/
def timeForOneSet (t : KnittingTimes) : ℝ :=
  t.hat + t.scarf + 2 * t.mitten + 2 * t.sock + t.sweater

/-- The main theorem stating that the time to knit a hat is 2 hours --/
theorem hat_knitting_time (t : KnittingTimes) 
  (h_scarf : t.scarf = 3)
  (h_mitten : t.mitten = 1)
  (h_sock : t.sock = 1.5)
  (h_sweater : t.sweater = 6)
  (h_total_time : 3 * timeForOneSet t = 48) : 
  t.hat = 2 := by
  sorry

end hat_knitting_time_l321_32119


namespace min_max_values_l321_32177

/-- Given positive real numbers x and y satisfying x² + y² = x + y,
    prove that the minimum value of 1/x + 1/y is 2 and the maximum value of x + y is 2 -/
theorem min_max_values (x y : ℝ) (h_pos : x > 0 ∧ y > 0) (h_eq : x^2 + y^2 = x + y) :
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = a + b → 1/x + 1/y ≤ 1/a + 1/b) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = a + b → x + y ≥ a + b) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = a + b ∧ 1/a + 1/b = 2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = a + b ∧ a + b = 2) :=
by sorry

end min_max_values_l321_32177


namespace maria_average_sales_l321_32183

/-- The average number of kilograms of apples sold per hour by Maria at the market -/
def average_apples_sold (first_hour_sales second_hour_sales : ℕ) (total_hours : ℕ) : ℚ :=
  (first_hour_sales + second_hour_sales : ℚ) / total_hours

/-- Theorem stating that Maria's average apple sales per hour is 6 kg/hour -/
theorem maria_average_sales :
  average_apples_sold 10 2 2 = 6 := by
  sorry

end maria_average_sales_l321_32183


namespace decompose_50900300_l321_32107

theorem decompose_50900300 :
  ∃ (ten_thousands ones : ℕ),
    50900300 = ten_thousands * 10000 + ones ∧
    ten_thousands = 5090 ∧
    ones = 300 := by
  sorry

end decompose_50900300_l321_32107


namespace smallest_r_is_two_l321_32168

theorem smallest_r_is_two :
  ∃ (r : ℝ), r > 0 ∧ r = 2 ∧
  (∀ (a : ℝ), a > 0 →
    ∃ (x : ℝ), (2 - a * r ≤ x) ∧ (x ≤ 2) ∧ (a * x^3 + x^2 - 4 = 0)) ∧
  (∀ (r' : ℝ), r' > 0 →
    (∀ (a : ℝ), a > 0 →
      ∃ (x : ℝ), (2 - a * r' ≤ x) ∧ (x ≤ 2) ∧ (a * x^3 + x^2 - 4 = 0)) →
    r' ≥ r) :=
by sorry

end smallest_r_is_two_l321_32168


namespace reciprocal_of_abs_neg_three_l321_32104

theorem reciprocal_of_abs_neg_three (x : ℝ) : x = |(-3)| → 1 / x = 1 / 3 := by
  sorry

end reciprocal_of_abs_neg_three_l321_32104


namespace binomial_11_choose_9_l321_32101

theorem binomial_11_choose_9 : Nat.choose 11 9 = 55 := by
  sorry

end binomial_11_choose_9_l321_32101


namespace exists_special_function_l321_32145

/-- The closed interval [0, 1] -/
def ClosedUnitInterval : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

/-- A continuous function from [0, 1] to [0, 1] -/
def ContinuousUnitFunction (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ ∀ x ∈ ClosedUnitInterval, f x ∈ ClosedUnitInterval

/-- A line in ℝ² -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The number of intersections between a function and a line -/
def NumberOfIntersections (f : ℝ → ℝ) (l : Line) : ℕ := sorry

/-- The existence of a function with the required properties -/
theorem exists_special_function :
  ∃ g : ℝ → ℝ,
    ContinuousUnitFunction g ∧
    (∀ l : Line, NumberOfIntersections g l < ω) ∧
    (∀ n : ℕ, ∃ l : Line, NumberOfIntersections g l > n) :=
sorry

end exists_special_function_l321_32145


namespace geometric_progression_common_ratio_l321_32160

/-- A geometric progression where each term is positive and any given term 
    is equal to the sum of the next three following terms. -/
structure GeometricProgression where
  a : ℝ  -- First term
  r : ℝ  -- Common ratio
  a_pos : 0 < a  -- Each term is positive
  r_pos : 0 < r  -- Common ratio is positive (to ensure all terms are positive)
  sum_property : ∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)

theorem geometric_progression_common_ratio 
  (gp : GeometricProgression) : 
  gp.r^3 + gp.r^2 + gp.r - 1 = 0 ∧ 
  abs (gp.r - 0.5437) < 0.0001 := by
sorry

end geometric_progression_common_ratio_l321_32160


namespace maurice_horseback_rides_l321_32194

theorem maurice_horseback_rides (maurice_visit_rides : ℕ) 
                                (matt_with_maurice : ℕ) 
                                (matt_alone_rides : ℕ) : 
  maurice_visit_rides = 8 →
  matt_with_maurice = 8 →
  matt_alone_rides = 16 →
  matt_with_maurice + matt_alone_rides = 3 * maurice_before_visit →
  maurice_before_visit = 8 := by
  sorry

def maurice_before_visit : ℕ := 8

end maurice_horseback_rides_l321_32194


namespace smartphone_price_difference_l321_32103

/-- Calculates the final price after discount and tax --/
def finalPrice (basePrice : ℝ) (quantity : ℕ) (discount : ℝ) (taxRate : ℝ) : ℝ :=
  let discountedPrice := basePrice * quantity * (1 - discount)
  discountedPrice * (1 + taxRate)

/-- Proves that the difference between Jane's and Tom's total costs is $112.68 --/
theorem smartphone_price_difference : 
  let storeAPrice := 125
  let storeBPrice := 130
  let storeADiscount := 0.12
  let storeBDiscount := 0.15
  let storeATaxRate := 0.07
  let storeBTaxRate := 0.05
  let tomQuantity := 2
  let janeQuantity := 3
  abs (finalPrice storeBPrice janeQuantity storeBDiscount storeBTaxRate - 
       finalPrice storeAPrice tomQuantity storeADiscount storeATaxRate - 112.68) < 0.01 := by
  sorry

end smartphone_price_difference_l321_32103


namespace subtraction_of_fractions_l321_32133

theorem subtraction_of_fractions : (8 : ℚ) / 19 - (5 : ℚ) / 57 = (1 : ℚ) / 3 := by
  sorry

end subtraction_of_fractions_l321_32133


namespace closest_whole_number_to_ratio_l321_32170

theorem closest_whole_number_to_ratio : 
  let ratio := (10^4000 + 3*10^4002) / (2*10^4001 + 4*10^4001)
  ∃ (n : ℕ), n = 5 ∧ ∀ (m : ℕ), m ≠ n → |ratio - (n : ℝ)| < |ratio - (m : ℝ)| :=
by sorry

end closest_whole_number_to_ratio_l321_32170


namespace max_value_theorem_equality_condition_l321_32152

theorem max_value_theorem (x : ℝ) (h : x > 0) : 2 - x - 4 / x ≤ -2 :=
sorry

theorem equality_condition (x : ℝ) (h : x > 0) : 2 - x - 4 / x = -2 ↔ x = 2 :=
sorry

end max_value_theorem_equality_condition_l321_32152


namespace parabola_and_intersection_l321_32161

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/2 - y^2/2 = 1

-- Define the line passing through P(3,1) with slope 1
def line (x y : ℝ) : Prop := y = x - 2

-- Theorem statement
theorem parabola_and_intersection :
  -- Conditions
  (∀ x y, parabola_C x y → (x = 0 ∧ y = 0) → True) → -- Vertex at origin
  (∀ x, parabola_C x 0 → True) → -- Axis of symmetry is coordinate axis
  (∃ x₀, x₀ = -2 ∧ (∀ x y, parabola_C x y → |x - x₀| = y^2/(4*x₀))) → -- Directrix passes through left focus of hyperbola
  -- Conclusions
  (∀ x y, parabola_C x y ↔ y^2 = 8*x) ∧ -- Equation of parabola C
  (∃ x₁ y₁ x₂ y₂, 
    parabola_C x₁ y₁ ∧ parabola_C x₂ y₂ ∧ 
    line x₁ y₁ ∧ line x₂ y₂ ∧ 
    ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2 : ℝ) = 16) -- Length of MN is 16
  := by sorry

end parabola_and_intersection_l321_32161


namespace dental_cleaning_theorem_l321_32199

/-- Represents the number of teeth for different animals --/
structure AnimalTeeth where
  dog : ℕ
  cat : ℕ
  pig : ℕ

/-- Represents the number of animals to be cleaned --/
structure AnimalsToClean where
  dogs : ℕ
  cats : ℕ
  pigs : ℕ

/-- Calculates the total number of teeth cleaned --/
def totalTeethCleaned (teeth : AnimalTeeth) (animals : AnimalsToClean) : ℕ :=
  teeth.dog * animals.dogs + teeth.cat * animals.cats + teeth.pig * animals.pigs

/-- Theorem stating that given the conditions, 5 dogs result in 706 teeth cleaned --/
theorem dental_cleaning_theorem (teeth : AnimalTeeth) (animals : AnimalsToClean) :
  teeth.dog = 42 →
  teeth.cat = 30 →
  teeth.pig = 28 →
  animals.cats = 10 →
  animals.pigs = 7 →
  totalTeethCleaned teeth { dogs := 5, cats := animals.cats, pigs := animals.pigs } = 706 :=
by
  sorry

#check dental_cleaning_theorem

end dental_cleaning_theorem_l321_32199


namespace gcd_lcm_product_150_180_l321_32193

theorem gcd_lcm_product_150_180 : Nat.gcd 150 180 * Nat.lcm 150 180 = 27000 := by
  sorry

end gcd_lcm_product_150_180_l321_32193


namespace log_ratio_squared_l321_32186

theorem log_ratio_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1) 
  (h1 : Real.log x / Real.log 3 = Real.log 81 / Real.log y) (h2 : x * y = 243) :
  (Real.log (x / y) / Real.log 3)^2 = 9 := by
  sorry

end log_ratio_squared_l321_32186


namespace inequality_solution_l321_32100

theorem inequality_solution (y : ℝ) : 
  (1 / (y * (y + 2)) - 1 / ((y + 2) * (y + 4)) < 1 / 4) ↔ 
  (y < -4 ∨ (-2 < y ∧ y < 0) ∨ 2 < y) :=
sorry

end inequality_solution_l321_32100


namespace right_angled_triangle_l321_32175

theorem right_angled_triangle (h₁ h₂ h₃ : ℝ) (h_positive : h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0)
  (h_altitudes : h₁ = 12 ∧ h₂ = 15 ∧ h₃ = 20) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a * h₁ = 2 * (b * c / 2)) ∧
    (b * h₂ = 2 * (a * c / 2)) ∧
    (c * h₃ = 2 * (a * b / 2)) ∧
    a^2 + b^2 = c^2 :=
by sorry

end right_angled_triangle_l321_32175


namespace smallest_root_of_quadratic_l321_32157

theorem smallest_root_of_quadratic (x : ℝ) :
  (4 * x^2 - 20 * x + 24 = 0) → (x ≥ 2) :=
by
  sorry

end smallest_root_of_quadratic_l321_32157


namespace ship_journey_distance_l321_32191

theorem ship_journey_distance : 
  let day1_distance : ℕ := 100
  let day2_distance : ℕ := 3 * day1_distance
  let day3_distance : ℕ := day2_distance + 110
  let total_distance : ℕ := day1_distance + day2_distance + day3_distance
  total_distance = 810 := by sorry

end ship_journey_distance_l321_32191


namespace fifth_term_geometric_progression_l321_32111

theorem fifth_term_geometric_progression :
  ∀ (b : ℕ → ℝ),
  (∀ n, b (n + 1) = b (n + 2) - b n) →  -- Each term from the second is the difference of adjacent terms
  b 1 = 7 - 3 * Real.sqrt 5 →           -- First term
  (∀ n, b (n + 1) > b n) →              -- Increasing progression
  b 5 = 2 :=                            -- Fifth term is 2
by
  sorry

end fifth_term_geometric_progression_l321_32111


namespace chord_length_l321_32146

/-- Theorem: The length of a chord in a circle is √(2ar), where r is the radius of the circle
    and a is the distance from one end of the chord to the tangent drawn through its other end. -/
theorem chord_length (r a : ℝ) (hr : r > 0) (ha : a > 0) :
  ∃ (chord_length : ℝ), chord_length = Real.sqrt (2 * a * r) := by
  sorry

end chord_length_l321_32146


namespace perpendicular_line_equation_l321_32189

/-- Given a line L1 with equation x - 2y + 3 = 0 and a point A(1, 2),
    the line L2 passing through A and perpendicular to L1 has the equation 2x + y - 4 = 0 -/
theorem perpendicular_line_equation (x y : ℝ) : 
  let L1 : ℝ → ℝ → Prop := λ x y ↦ x - 2*y + 3 = 0
  let A : ℝ × ℝ := (1, 2)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ 2*x + y - 4 = 0
  (∀ x y, L1 x y ↔ x - 2*y + 3 = 0) →
  (L2 A.1 A.2) →
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → L2 x₁ y₁ → L2 x₂ y₂ → 
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁)) * ((x₂ - x₁) * (x₂ - x₁)) = 0) →
  ∀ x y, L2 x y ↔ 2*x + y - 4 = 0 := by
sorry


end perpendicular_line_equation_l321_32189


namespace min_value_S_l321_32113

theorem min_value_S (a b c : ℤ) (h1 : a + b + c = 2) 
  (h2 : (2*a + b*c)*(2*b + c*a)*(2*c + a*b) > 200) : 
  ∃ (m : ℤ), m = 256 ∧ 
  ∀ (x y z : ℤ), x + y + z = 2 → 
  (2*x + y*z)*(2*y + z*x)*(2*z + x*y) > 200 → 
  (2*x + y*z)*(2*y + z*x)*(2*z + x*y) ≥ m :=
sorry

end min_value_S_l321_32113


namespace remainder_theorem_l321_32129

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - x^2 + 3*x + 4

-- State the theorem
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, p x = (x + 2) * q x + 10 := by
  sorry

end remainder_theorem_l321_32129


namespace father_age_triple_marika_age_2014_l321_32137

/-- Represents a person with their birth year -/
structure Person where
  birthYear : ℕ

/-- Marika, born in 1996 -/
def marika : Person := ⟨1996⟩

/-- Marika's father, born in 1961 -/
def father : Person := ⟨1961⟩

/-- The year when Marika was 10 years old -/
def baseYear : ℕ := 2006

/-- Calculates a person's age in a given year -/
def age (p : Person) (year : ℕ) : ℕ :=
  year - p.birthYear

/-- Theorem stating that 2014 is the first year when the father's age is exactly three times Marika's age -/
theorem father_age_triple_marika_age_2014 :
  (∀ y : ℕ, y < 2014 → y ≥ baseYear → age father y ≠ 3 * age marika y) ∧
  age father 2014 = 3 * age marika 2014 :=
sorry

end father_age_triple_marika_age_2014_l321_32137


namespace trapezoid_area_property_l321_32114

/-- Represents the area of a trapezoid with bases and altitude in arithmetic progression -/
def trapezoid_area (a : ℝ) : ℝ := a ^ 2

/-- The area of a trapezoid with bases and altitude in arithmetic progression
    can be any non-negative real number -/
theorem trapezoid_area_property :
  ∀ (J : ℝ), J ≥ 0 → ∃ (a : ℝ), trapezoid_area a = J :=
by sorry

end trapezoid_area_property_l321_32114


namespace pascal_triangle_row20_element5_l321_32151

theorem pascal_triangle_row20_element5 : Nat.choose 20 4 = 4845 := by
  sorry

end pascal_triangle_row20_element5_l321_32151


namespace fraction_to_decimal_l321_32105

theorem fraction_to_decimal (h : 343 = 7^3) : 7 / 343 = 0.056 := by
  sorry

end fraction_to_decimal_l321_32105


namespace gcd_180_270_l321_32159

theorem gcd_180_270 : Nat.gcd 180 270 = 90 := by
  sorry

end gcd_180_270_l321_32159


namespace perimeter_of_eight_squares_l321_32124

theorem perimeter_of_eight_squares (total_area : ℝ) (num_squares : ℕ) :
  total_area = 512 →
  num_squares = 8 →
  let square_area := total_area / num_squares
  let side_length := Real.sqrt square_area
  let perimeter := (2 * num_squares - 2) * side_length + 2 * side_length
  perimeter = 112 := by
  sorry

end perimeter_of_eight_squares_l321_32124


namespace georgie_enter_exit_ways_l321_32181

/-- The number of windows in the haunted mansion -/
def num_windows : ℕ := 8

/-- The number of ways Georgie can enter and exit the mansion -/
def num_ways : ℕ := num_windows * (num_windows - 1)

/-- Theorem stating that the number of ways Georgie can enter and exit is 56 -/
theorem georgie_enter_exit_ways : num_ways = 56 := by
  sorry

end georgie_enter_exit_ways_l321_32181


namespace subset_condition_disjoint_condition_l321_32198

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem 1: B ⊆ A ⇔ m ∈ (-∞, 3]
theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by sorry

-- Theorem 2: A ∩ B = ∅ ⇔ m ∈ (-∞, 2) ∪ (4, +∞)
theorem disjoint_condition (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end subset_condition_disjoint_condition_l321_32198


namespace pickle_ratio_l321_32176

/-- Prove the ratio of pickle slices Tammy can eat to Sammy can eat -/
theorem pickle_ratio (sammy tammy ron : ℕ) : 
  sammy = 15 → 
  ron = 24 → 
  ron = (80 * tammy) / 100 → 
  tammy / sammy = 2 := by
  sorry

end pickle_ratio_l321_32176


namespace hundred_to_fifty_equals_ten_to_hundred_l321_32173

theorem hundred_to_fifty_equals_ten_to_hundred : 100 ^ 50 = 10 ^ 100 := by
  sorry

end hundred_to_fifty_equals_ten_to_hundred_l321_32173


namespace roberto_outfits_l321_32174

/-- Represents the number of trousers Roberto has -/
def num_trousers : ℕ := 4

/-- Represents the number of shirts Roberto has -/
def num_shirts : ℕ := 7

/-- Represents the number of jackets Roberto has -/
def num_jackets : ℕ := 5

/-- Represents the number of hat options Roberto has (wear or not wear) -/
def num_hat_options : ℕ := 2

/-- Calculates the total number of outfit combinations -/
def total_outfits : ℕ := num_trousers * num_shirts * num_jackets * num_hat_options

/-- Theorem stating that the total number of outfits Roberto can create is 280 -/
theorem roberto_outfits : total_outfits = 280 := by
  sorry

end roberto_outfits_l321_32174


namespace trigonometric_simplification_l321_32196

theorem trigonometric_simplification :
  (Real.sin (15 * π / 180) + Real.sin (30 * π / 180) + Real.sin (45 * π / 180) + 
   Real.sin (60 * π / 180) + Real.sin (75 * π / 180)) / 
  (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180)) = 
  (Real.sqrt 2 * (4 * Real.cos (22.5 * π / 180) * Real.cos (7.5 * π / 180) + 1)) / 
  (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180)) :=
by sorry

end trigonometric_simplification_l321_32196


namespace complex_division_l321_32120

theorem complex_division : ((-2 : ℂ) - I) / I = -1 + 2*I := by sorry

end complex_division_l321_32120


namespace polynomial_divisibility_l321_32154

theorem polynomial_divisibility (a b : ℕ) (h1 : a ≥ 2 * b) (h2 : b > 1) :
  ∃ P : Polynomial ℕ, (Polynomial.degree P > 0) ∧ 
    (∀ i, Polynomial.coeff P i < b) ∧
    (P.eval a % P.eval b = 0) := by
  sorry

end polynomial_divisibility_l321_32154


namespace chef_inventory_solution_l321_32190

def chef_inventory (initial_apples initial_flour initial_sugar initial_butter : ℕ) : Prop :=
  let used_apples : ℕ := 15
  let used_flour : ℕ := 6
  let used_sugar : ℕ := 14  -- 10 initially + 4 from newly bought
  let used_butter : ℕ := 3
  let remaining_apples : ℕ := 4
  let remaining_flour : ℕ := 3
  let remaining_sugar : ℕ := 13
  let remaining_butter : ℕ := 2
  let given_away_apples : ℕ := 2
  (initial_apples = used_apples + given_away_apples + remaining_apples) ∧
  (initial_flour = 2 * (used_flour + remaining_flour)) ∧
  (initial_sugar = used_sugar + remaining_sugar) ∧
  (initial_butter = used_butter + remaining_butter)

theorem chef_inventory_solution :
  ∃ (initial_apples initial_flour initial_sugar initial_butter : ℕ),
    chef_inventory initial_apples initial_flour initial_sugar initial_butter ∧
    initial_apples = 21 ∧
    initial_flour = 18 ∧
    initial_sugar = 27 ∧
    initial_butter = 5 ∧
    initial_apples + initial_flour + initial_sugar + initial_butter = 71 :=
by sorry

end chef_inventory_solution_l321_32190


namespace product_sum_not_1001_l321_32171

theorem product_sum_not_1001 (a b c d : ℤ) (h1 : a + b = 100) (h2 : c + d = 100) : 
  a * b + c * d ≠ 1001 := by
sorry

end product_sum_not_1001_l321_32171


namespace bacon_count_l321_32102

/-- The number of students who suggested adding mashed potatoes -/
def mashed_potatoes : ℕ := 330

/-- The difference between the number of students who suggested mashed potatoes and bacon -/
def difference : ℕ := 61

/-- The number of students who suggested adding bacon -/
def bacon : ℕ := mashed_potatoes - difference

theorem bacon_count : bacon = 269 := by
  sorry

end bacon_count_l321_32102


namespace greatest_line_segment_length_l321_32192

/-- The greatest possible length of a line segment joining two points on a circle -/
theorem greatest_line_segment_length (r : ℝ) (h : r = 4) : 
  ∃ (d : ℝ), d = 2 * r ∧ ∀ (l : ℝ), l ≤ d := by sorry

end greatest_line_segment_length_l321_32192


namespace symmetry_coordinates_l321_32148

/-- Two points are symmetric about the origin if their coordinates are negatives of each other -/
def symmetric_about_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

theorem symmetry_coordinates :
  ∀ (m n : ℝ), symmetric_about_origin (m, 4) (-2, n) → m = 2 ∧ n = -4 := by
  sorry

end symmetry_coordinates_l321_32148


namespace ceil_zero_exists_ceil_minus_self_eq_point_two_l321_32125

-- Define the ceiling function [x)
noncomputable def ceil (x : ℝ) : ℤ :=
  Int.ceil x

-- Theorem 1: [0) = 1
theorem ceil_zero : ceil 0 = 1 := by sorry

-- Theorem 2: There exists an x such that [x) - x = 0.2
theorem exists_ceil_minus_self_eq_point_two :
  ∃ x : ℝ, (ceil x : ℝ) - x = 0.2 := by sorry

end ceil_zero_exists_ceil_minus_self_eq_point_two_l321_32125


namespace x_fourth_minus_four_x_cubed_plus_four_x_squared_plus_four_equals_five_l321_32143

theorem x_fourth_minus_four_x_cubed_plus_four_x_squared_plus_four_equals_five :
  ∀ x : ℝ, x = 1 + Real.sqrt 2 → x^4 - 4*x^3 + 4*x^2 + 4 = 5 := by
  sorry

end x_fourth_minus_four_x_cubed_plus_four_x_squared_plus_four_equals_five_l321_32143


namespace equations_represent_problem_l321_32195

/-- Represents the money each person brought -/
structure Money where
  a : ℝ  -- Amount A brought
  b : ℝ  -- Amount B brought

/-- Checks if the given equations satisfy the conditions of the problem -/
def satisfies_conditions (m : Money) : Prop :=
  (m.a + (1/2) * m.b = 50) ∧ (m.b + (2/3) * m.a = 50)

/-- Theorem stating that the equations correctly represent the problem -/
theorem equations_represent_problem :
  ∃ (m : Money), satisfies_conditions m :=
sorry

end equations_represent_problem_l321_32195


namespace solve_equation_1_solve_equation_2_solve_equation_3_l321_32147

-- Problem 1
theorem solve_equation_1 : 
  let f : ℝ → ℝ := λ x => -3*x*(2*x-3)+(2*x-3)
  ∃ x₁ x₂ : ℝ, x₁ = 3/2 ∧ x₂ = 1/3 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

-- Problem 2
theorem solve_equation_2 :
  let f : ℝ → ℝ := λ x => x^2 - 6*x + 4
  ∃ x₁ x₂ : ℝ, x₁ = 3 + Real.sqrt 5 ∧ x₂ = 3 - Real.sqrt 5 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

-- Problem 3
theorem solve_equation_3 :
  let f : ℝ → ℝ := λ x => 4 / (x^2 - 4) + 1 / (x - 2) + 1
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ -2 → f x ≠ 0 :=
by sorry

end solve_equation_1_solve_equation_2_solve_equation_3_l321_32147


namespace geometric_sequence_sum_l321_32126

/-- Given a geometric sequence {a_n} with a₁ = 2 and q = 2,
    prove that if the sum of the first n terms Sn = 126, then n = 6 -/
theorem geometric_sequence_sum (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  a 1 = 2 →
  (∀ k, a (k + 1) = 2 * a k) →
  S n = (a 1) * (1 - 2^n) / (1 - 2) →
  S n = 126 →
  n = 6 := by
sorry

end geometric_sequence_sum_l321_32126


namespace ab_value_l321_32112

theorem ab_value (a b c d : ℝ) 
  (h1 : a - b = 3)
  (h2 : a^2 + b^2 = 25)
  (h3 : a = 2*c + Real.sqrt d) :
  a * b = 8 := by
  sorry

end ab_value_l321_32112


namespace sweater_discount_percentage_l321_32184

theorem sweater_discount_percentage (final_price saved : ℝ) : 
  final_price = 27 → saved = 3 → (saved / (final_price + saved)) * 100 = 10 := by
sorry

end sweater_discount_percentage_l321_32184


namespace sin_18_cos_36_eq_quarter_l321_32139

theorem sin_18_cos_36_eq_quarter : Real.sin (18 * π / 180) * Real.cos (36 * π / 180) = 1/4 := by
  sorry

end sin_18_cos_36_eq_quarter_l321_32139


namespace poker_loss_l321_32149

theorem poker_loss (initial_amount winnings debt : ℤ) : 
  initial_amount = 100 → winnings = 65 → debt = 50 → 
  (initial_amount + winnings + debt) = 215 := by
sorry

end poker_loss_l321_32149


namespace inequality_proof_l321_32123

theorem inequality_proof (k n : ℕ) (hk : k > 1) (hn : n > 1) :
  (1 : ℝ) / ((n + 1 : ℝ) ^ (1 / k : ℝ)) + (1 : ℝ) / ((k + 1 : ℝ) ^ (1 / n : ℝ)) > 1 := by
  sorry

end inequality_proof_l321_32123


namespace circle_line_distance_range_l321_32169

/-- Given a circle x^2 + y^2 = 16 and a line y = x + b, if there are at least three points
    on the circle with a distance of 1 from the line, then -3√2 ≤ b ≤ 3√2 -/
theorem circle_line_distance_range (b : ℝ) :
  (∃ (p q r : ℝ × ℝ),
    p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    (p.1^2 + p.2^2 = 16) ∧ (q.1^2 + q.2^2 = 16) ∧ (r.1^2 + r.2^2 = 16) ∧
    (abs (p.1 - p.2 + b) / Real.sqrt 2 = 1) ∧
    (abs (q.1 - q.2 + b) / Real.sqrt 2 = 1) ∧
    (abs (r.1 - r.2 + b) / Real.sqrt 2 = 1)) →
  -3 * Real.sqrt 2 ≤ b ∧ b ≤ 3 * Real.sqrt 2 := by
  sorry


end circle_line_distance_range_l321_32169


namespace max_rectangle_division_ratio_l321_32138

/-- The number of ways to divide a rectangle with side lengths a and b into smaller rectangles with integer side lengths -/
def D (a b : ℕ+) : ℕ := sorry

/-- The theorem stating that D(a,b)/(2(a+b)) ≤ 3/8 for all positive integers a and b, 
    with equality if and only if a = b = 2 -/
theorem max_rectangle_division_ratio 
  (a b : ℕ+) : 
  (D a b : ℚ) / (2 * ((a:ℚ) + (b:ℚ))) ≤ 3/8 ∧ 
  ((D a b : ℚ) / (2 * ((a:ℚ) + (b:ℚ))) = 3/8 ↔ a = 2 ∧ b = 2) := by
  sorry

end max_rectangle_division_ratio_l321_32138


namespace unique_prime_seventh_power_l321_32115

theorem unique_prime_seventh_power (p : ℕ) : 
  Prime p ∧ ∃ q, Prime q ∧ p + 25 = q^7 ↔ p = 103 := by
sorry

end unique_prime_seventh_power_l321_32115


namespace soccer_sideline_time_l321_32108

/-- Given a soccer game duration and a player's playing times, calculate the time spent on the sideline -/
theorem soccer_sideline_time (game_duration playing_time1 playing_time2 : ℕ) 
  (h1 : game_duration = 90)
  (h2 : playing_time1 = 20)
  (h3 : playing_time2 = 35) :
  game_duration - (playing_time1 + playing_time2) = 35 := by
  sorry

end soccer_sideline_time_l321_32108


namespace log_stack_sum_l321_32172

theorem log_stack_sum (n : ℕ) (a l : ℕ) (h1 : n = 12) (h2 : a = 4) (h3 : l = 15) :
  n * (a + l) / 2 = 114 := by
  sorry

end log_stack_sum_l321_32172


namespace derivative_sin_cos_x_l321_32130

theorem derivative_sin_cos_x (x : ℝ) : 
  deriv (λ x => Real.sin x * Real.cos x) x = Real.cos (2 * x) := by
  sorry

end derivative_sin_cos_x_l321_32130


namespace unique_consecutive_digit_square_swap_l321_32187

/-- A function that checks if a number is formed by four consecutive digits -/
def is_consecutive_digits (n : ℕ) : Prop :=
  ∃ a : ℕ, n = 1000 * a + 100 * (a + 1) + 10 * (a + 2) + (a + 3)

/-- A function that swaps the first two digits of a four-digit number -/
def swap_first_two_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let last_two := n % 100
  1000 * d2 + 100 * d1 + last_two

/-- The main theorem stating that 3456 is the only number satisfying the conditions -/
theorem unique_consecutive_digit_square_swap :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 →
    (is_consecutive_digits n ∧ ∃ m : ℕ, swap_first_two_digits n = m ^ 2) ↔ n = 3456 :=
sorry

end unique_consecutive_digit_square_swap_l321_32187


namespace tangent_equality_solution_l321_32167

theorem tangent_equality_solution (x : Real) : 
  0 < x ∧ x < 360 →
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) →
  x = 100 ∨ x = 220 := by
sorry

end tangent_equality_solution_l321_32167


namespace problem_statement_l321_32136

def f (x : ℝ) : ℝ := x^2 - 1

theorem problem_statement :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 →
    (∀ m : ℝ, (4 * m^2 * |f x| + 4 * f m ≤ |f (x - 1)| ↔ -1/2 ≤ m ∧ m ≤ 1/2))) ∧
  (∀ a : ℝ, (∀ x₁ : ℝ, 1 ≤ x₁ ∧ x₁ ≤ 2 →
    ∃ x₂ : ℝ, 1 ≤ x₂ ∧ x₂ ≤ 2 ∧ f x₁ = |2 * f x₂ - a * x₂|) ↔
      ((0 ≤ a ∧ a ≤ 3/2) ∨ a = 3)) :=
by sorry

end problem_statement_l321_32136


namespace triangle_side_length_l321_32132

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  a = 1 →
  b = Real.sqrt 3 →
  Real.sin B = Real.sin (2 * A) →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a * Real.sin C = b * Real.sin A →
  a * Real.sin C = c * Real.sin B →
  c = 2 := by
  sorry

end triangle_side_length_l321_32132


namespace triangle_with_prime_angles_exists_l321_32153

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Theorem statement
theorem triangle_with_prime_angles_exists : ∃ p q r : ℕ, 
  isPrime p ∧ isPrime q ∧ isPrime r ∧ p + q + r = 180 := by
  sorry

end triangle_with_prime_angles_exists_l321_32153


namespace mike_practice_hours_l321_32134

/-- Calculates the total practice hours for a goalkeeper before a game -/
def total_practice_hours (weekday_hours : ℕ) (saturday_hours : ℕ) (weeks_until_game : ℕ) : ℕ :=
  (weekday_hours * 5 + saturday_hours) * weeks_until_game

/-- Theorem: Mike's total practice hours before the next game -/
theorem mike_practice_hours :
  total_practice_hours 3 5 3 = 60 := by
  sorry

#eval total_practice_hours 3 5 3

end mike_practice_hours_l321_32134


namespace sum_ratio_equals_half_l321_32150

theorem sum_ratio_equals_half
  (a b c x y z : ℝ)
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares_abc : a^2 + b^2 + c^2 = 10)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 40)
  (sum_products : a*x + b*y + c*z = 20) :
  (a + b + c) / (x + y + z) = 1/2 := by
  sorry

end sum_ratio_equals_half_l321_32150


namespace polynomial_divisibility_l321_32110

-- Define the polynomial
def f (x : ℝ) : ℝ := 9*x^3 - 5*x^2 - 48*x + 54

-- Define divisibility by (x - p)^2
def is_divisible_by_square (p : ℝ) : Prop :=
  ∃ (q : ℝ → ℝ), ∀ x, f x = (x - p)^2 * q x

-- Theorem statement
theorem polynomial_divisibility :
  ∀ p : ℝ, is_divisible_by_square p → p = 8/3 :=
by sorry

end polynomial_divisibility_l321_32110


namespace fraction_equality_l321_32165

theorem fraction_equality : (1 / 5 - 1 / 6) / (1 / 3 - 1 / 4) = 2 / 5 := by
  sorry

end fraction_equality_l321_32165


namespace negative_cube_root_of_negative_square_minus_one_l321_32118

theorem negative_cube_root_of_negative_square_minus_one (a : ℝ) :
  ∃ x : ℝ, x < 0 ∧ x^3 = -a^2 - 1 := by
  sorry

end negative_cube_root_of_negative_square_minus_one_l321_32118


namespace baseball_cost_l321_32162

def marbles_cost : ℚ := 9.05
def football_cost : ℚ := 4.95
def total_cost : ℚ := 20.52

theorem baseball_cost : total_cost - (marbles_cost + football_cost) = 6.52 := by
  sorry

end baseball_cost_l321_32162


namespace rectangle_area_diagonal_l321_32121

theorem rectangle_area_diagonal (l w d : ℝ) (h_ratio : l / w = 5 / 4) (h_diagonal : l^2 + w^2 = d^2) :
  l * w = (20 / 41) * d^2 := by
  sorry

end rectangle_area_diagonal_l321_32121


namespace quadratic_sum_l321_32109

/-- Given a quadratic function f(x) = 4x^2 - 40x + 100, 
    there exist constants a, b, and c such that 
    f(x) = a(x+b)^2 + c for all x, and a + b + c = -1 -/
theorem quadratic_sum (x : ℝ) : 
  ∃ (a b c : ℝ), (∀ x, 4*x^2 - 40*x + 100 = a*(x+b)^2 + c) ∧ (a + b + c = -1) := by
sorry

end quadratic_sum_l321_32109


namespace find_b_value_l321_32158

theorem find_b_value (a b : ℚ) (eq1 : 3 * a + 3 = 0) (eq2 : 2 * b - a = 4) : b = 3 / 2 := by
  sorry

end find_b_value_l321_32158


namespace M_equals_set_l321_32163

def M : Set ℕ := {m : ℕ | m > 0 ∧ (∃ k : ℤ, 10 = k * (m + 1))}

theorem M_equals_set : M = {1, 4, 9} := by
  sorry

end M_equals_set_l321_32163


namespace hexagon_diagonals_from_vertex_l321_32141

/-- The number of diagonals from a single vertex in a polygon -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

theorem hexagon_diagonals_from_vertex :
  diagonals_from_vertex hexagon_sides = 3 := by
  sorry

end hexagon_diagonals_from_vertex_l321_32141


namespace pure_imaginary_complex_number_l321_32166

theorem pure_imaginary_complex_number (a : ℝ) : 
  (((2 : ℂ) - a * Complex.I) / (1 + Complex.I)).re = 0 → a = 2 := by
  sorry

end pure_imaginary_complex_number_l321_32166


namespace fraction_addition_l321_32131

theorem fraction_addition : (4 : ℚ) / 510 + 25 / 34 = 379 / 510 := by
  sorry

end fraction_addition_l321_32131
