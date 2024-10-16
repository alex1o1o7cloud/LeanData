import Mathlib

namespace NUMINAMATH_CALUDE_tennis_ball_count_l3677_367744

theorem tennis_ball_count : 
  ∀ (lily frodo brian sam : ℕ),
  lily = 12 →
  frodo = lily + 15 →
  brian = 3 * frodo →
  sam = frodo + lily - 5 →
  sam = 34 := by
sorry

end NUMINAMATH_CALUDE_tennis_ball_count_l3677_367744


namespace NUMINAMATH_CALUDE_f_properties_l3677_367703

def f (a x : ℝ) : ℝ := |1 - x - a| + |2 * a - x|

theorem f_properties (a x : ℝ) :
  (f a 1 < 3 ↔ a > -2/3 ∧ a < 4/3) ∧
  (a ≥ 2/3 → f a x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3677_367703


namespace NUMINAMATH_CALUDE_average_weight_section_A_l3677_367798

theorem average_weight_section_A (students_A : ℕ) (students_B : ℕ) 
  (avg_weight_B : ℝ) (avg_weight_total : ℝ) :
  students_A = 24 →
  students_B = 16 →
  avg_weight_B = 35 →
  avg_weight_total = 38 →
  (students_A * avg_weight_section_A + students_B * avg_weight_B) / (students_A + students_B) = avg_weight_total →
  avg_weight_section_A = 40 := by
  sorry

#check average_weight_section_A

end NUMINAMATH_CALUDE_average_weight_section_A_l3677_367798


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3677_367720

theorem opposite_of_negative_fraction :
  -(-(1 / 2023)) = 1 / 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3677_367720


namespace NUMINAMATH_CALUDE_unimodal_peak_interval_peak_interval_length_specific_peak_interval_l3677_367709

/-- A unimodal function on [0,1] is a function that is monotonically increasing
    on [0,x*] and monotonically decreasing on [x*,1] for some x* in (0,1) -/
def UnimodalFunction (f : ℝ → ℝ) : Prop := 
  ∃ x_star : ℝ, 0 < x_star ∧ x_star < 1 ∧ 
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ x_star → f x ≤ f y) ∧
  (∀ x y : ℝ, x_star ≤ x ∧ x < y ∧ y ≤ 1 → f x ≥ f y)

/-- The peak interval of a unimodal function contains the peak point -/
def PeakInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  UnimodalFunction f ∧ 0 ≤ a ∧ b ≤ 1 ∧
  ∃ x_star : ℝ, a < x_star ∧ x_star < b ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ x_star → f x ≤ f y) ∧
  (∀ x y : ℝ, x_star ≤ x ∧ x < y ∧ y ≤ 1 → f x ≥ f y)

theorem unimodal_peak_interval 
  (f : ℝ → ℝ) (x₁ x₂ : ℝ) 
  (h_unimodal : UnimodalFunction f)
  (h_x₁ : 0 < x₁) (h_x₂ : x₂ < 1) (h_order : x₁ < x₂) :
  (f x₁ ≥ f x₂ → PeakInterval f 0 x₂) ∧
  (f x₁ ≤ f x₂ → PeakInterval f x₁ 1) := by sorry

theorem peak_interval_length 
  (f : ℝ → ℝ) (r : ℝ) 
  (h_unimodal : UnimodalFunction f)
  (h_r : 0 < r ∧ r < 0.5) :
  ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₂ < 1 ∧ x₁ < x₂ ∧ x₂ - x₁ ≥ 2*r ∧
  ((PeakInterval f 0 x₂ ∧ x₂ ≤ 0.5 + r) ∨
   (PeakInterval f x₁ 1 ∧ 1 - x₁ ≤ 0.5 + r)) := by sorry

theorem specific_peak_interval 
  (f : ℝ → ℝ) 
  (h_unimodal : UnimodalFunction f) :
  ∃ x₁ x₂ x₃ : ℝ, 
    x₁ = 0.34 ∧ x₂ = 0.66 ∧ x₃ = 0.32 ∧
    PeakInterval f 0 x₂ ∧
    PeakInterval f 0 x₁ ∧
    |x₁ - x₂| ≥ 0.02 ∧ |x₁ - x₃| ≥ 0.02 ∧ |x₂ - x₃| ≥ 0.02 := by sorry

end NUMINAMATH_CALUDE_unimodal_peak_interval_peak_interval_length_specific_peak_interval_l3677_367709


namespace NUMINAMATH_CALUDE_a5_equals_6_l3677_367727

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + (n - 1) * d

/-- The theorem stating that a5 = 6 in the given arithmetic sequence -/
theorem a5_equals_6 (a : ℕ → ℝ) (h1 : arithmetic_sequence a) (h2 : a 2 + a 8 = 12) :
  a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_a5_equals_6_l3677_367727


namespace NUMINAMATH_CALUDE_cube_root_of_negative_one_twenty_seventh_l3677_367787

theorem cube_root_of_negative_one_twenty_seventh :
  ((-1 / 3 : ℝ) : ℝ)^3 = -1 / 27 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_one_twenty_seventh_l3677_367787


namespace NUMINAMATH_CALUDE_commercial_viewers_l3677_367706

/-- Calculates the number of commercial viewers given revenue data -/
theorem commercial_viewers (revenue_per_view : ℚ) (revenue_per_sub : ℚ) 
  (num_subs : ℕ) (total_revenue : ℚ) : 
  revenue_per_view > 0 → 
  (total_revenue - revenue_per_sub * num_subs) / revenue_per_view = 100 → 
  ∃ (num_viewers : ℕ), num_viewers = 100 :=
by
  sorry

#check commercial_viewers (1/2) 1 27 77

end NUMINAMATH_CALUDE_commercial_viewers_l3677_367706


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3677_367778

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (5 * x + 9) = 11 → x = 22.4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3677_367778


namespace NUMINAMATH_CALUDE_fresh_mushroom_mass_calculation_l3677_367772

/-- The mass of fresh mushrooms in kg that, when dried, become 15 kg lighter
    and have a moisture content of 60%, given that fresh mushrooms contain 90% water. -/
def fresh_mushroom_mass : ℝ := 20

/-- The water content of fresh mushrooms as a percentage. -/
def fresh_water_content : ℝ := 90

/-- The water content of dried mushrooms as a percentage. -/
def dried_water_content : ℝ := 60

/-- The mass reduction after drying in kg. -/
def mass_reduction : ℝ := 15

theorem fresh_mushroom_mass_calculation :
  fresh_mushroom_mass * (1 - fresh_water_content / 100) =
  (fresh_mushroom_mass - mass_reduction) * (1 - dried_water_content / 100) :=
by sorry

end NUMINAMATH_CALUDE_fresh_mushroom_mass_calculation_l3677_367772


namespace NUMINAMATH_CALUDE_john_needs_additional_money_l3677_367792

/-- The amount of money John needs -/
def money_needed : ℚ := 2.50

/-- The amount of money John has -/
def money_has : ℚ := 0.75

/-- The additional money John needs -/
def additional_money : ℚ := money_needed - money_has

theorem john_needs_additional_money : additional_money = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_additional_money_l3677_367792


namespace NUMINAMATH_CALUDE_machine_depreciation_rate_l3677_367701

/-- The annual depreciation rate of a machine given its initial value,
    selling price after two years, and profit. -/
theorem machine_depreciation_rate
  (initial_value : ℝ)
  (selling_price : ℝ)
  (profit : ℝ)
  (h1 : initial_value = 150000)
  (h2 : selling_price = 113935)
  (h3 : profit = 24000)
  : ∃ (r : ℝ), initial_value * (1 - r / 100)^2 = selling_price - profit :=
sorry

end NUMINAMATH_CALUDE_machine_depreciation_rate_l3677_367701


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l3677_367797

theorem isosceles_triangle_side_lengths 
  (perimeter : ℝ) 
  (height : ℝ) 
  (is_isosceles : Bool) 
  (h1 : perimeter = 16) 
  (h2 : height = 4) 
  (h3 : is_isosceles = true) : 
  ∃ (a b c : ℝ), a = 5 ∧ b = 5 ∧ c = 6 ∧ a + b + c = perimeter := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l3677_367797


namespace NUMINAMATH_CALUDE_cubic_divisibility_l3677_367712

theorem cubic_divisibility : ∃ (n : ℕ), n > 0 ∧ 84^3 % n = 0 ∧ n = 592704 := by
  sorry

end NUMINAMATH_CALUDE_cubic_divisibility_l3677_367712


namespace NUMINAMATH_CALUDE_collinear_implies_relation_vector_relation_implies_coordinates_l3677_367738

-- Define points A, B, and C
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (3, -1)
def C : ℝ → ℝ → ℝ × ℝ := λ a b => (a, b)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, r.1 - p.1 = t * (q.1 - p.1) ∧ r.2 - p.2 = t * (q.2 - p.2)

-- Define vector multiplication
def vec_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Part 1: Collinearity implies a = 2 - b
theorem collinear_implies_relation (a b : ℝ) :
  collinear A B (C a b) → a = 2 - b := by sorry

-- Part 2: AC = 2AB implies C = (5, -3)
theorem vector_relation_implies_coordinates (a b : ℝ) :
  C a b - A = vec_mult 2 (B - A) → C a b = (5, -3) := by sorry

end NUMINAMATH_CALUDE_collinear_implies_relation_vector_relation_implies_coordinates_l3677_367738


namespace NUMINAMATH_CALUDE_sqrt_floor_problem_l3677_367761

theorem sqrt_floor_problem (a b c : ℝ) : 
  (abs a = 4) → 
  (b^2 = 9) → 
  (c^3 = -8) → 
  (a > c) → 
  (c > b) → 
  Int.floor (Real.sqrt (a - b - 2*c)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_floor_problem_l3677_367761


namespace NUMINAMATH_CALUDE_empty_solution_set_l3677_367758

def f (x : ℝ) : ℝ := x^2 + x

theorem empty_solution_set :
  {x : ℝ | f (x - 2) + f x < 0} = ∅ := by sorry

end NUMINAMATH_CALUDE_empty_solution_set_l3677_367758


namespace NUMINAMATH_CALUDE_remaining_item_is_bead_l3677_367718

/-- Represents the three types of items --/
inductive Item
  | GoldBar
  | Pearl
  | Bead

/-- Represents the state of the tribe's possessions --/
structure TribeState where
  goldBars : Nat
  pearls : Nat
  beads : Nat

/-- Represents the possible exchanges --/
inductive Exchange
  | Cortes    -- 1 gold bar + 1 pearl → 1 bead
  | Montezuma -- 1 gold bar + 1 bead → 1 pearl
  | Totonacs  -- 1 pearl + 1 bead → 1 gold bar

def initialState : TribeState :=
  { goldBars := 24, pearls := 26, beads := 25 }

def applyExchange (state : TribeState) (exchange : Exchange) : TribeState :=
  match exchange with
  | Exchange.Cortes =>
      { goldBars := state.goldBars - 1, pearls := state.pearls - 1, beads := state.beads + 1 }
  | Exchange.Montezuma =>
      { goldBars := state.goldBars - 1, pearls := state.pearls + 1, beads := state.beads - 1 }
  | Exchange.Totonacs =>
      { goldBars := state.goldBars + 1, pearls := state.pearls - 1, beads := state.beads - 1 }

def remainingItem (state : TribeState) : Option Item :=
  if state.goldBars > 0 && state.pearls = 0 && state.beads = 0 then some Item.GoldBar
  else if state.goldBars = 0 && state.pearls > 0 && state.beads = 0 then some Item.Pearl
  else if state.goldBars = 0 && state.pearls = 0 && state.beads > 0 then some Item.Bead
  else none

/-- Theorem stating that if only one item type remains after any number of exchanges, it must be beads --/
theorem remaining_item_is_bead (exchanges : List Exchange) :
  let finalState := exchanges.foldl applyExchange initialState
  remainingItem finalState = some Item.Bead ∨ remainingItem finalState = none := by
  sorry

end NUMINAMATH_CALUDE_remaining_item_is_bead_l3677_367718


namespace NUMINAMATH_CALUDE_euler_totient_power_of_two_l3677_367722

theorem euler_totient_power_of_two (n : ℕ) : 
  Odd n → 
  ∃ k m : ℕ, Nat.totient n = 2^k ∧ Nat.totient (n+1) = 2^m → 
  ∃ p : ℕ, n + 1 = 2^p ∨ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_euler_totient_power_of_two_l3677_367722


namespace NUMINAMATH_CALUDE_quadratic_shift_l3677_367793

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 1

/-- Shift a function to the left -/
def shiftLeft (g : ℝ → ℝ) (d : ℝ) : ℝ → ℝ := λ x ↦ g (x + d)

/-- Shift a function down -/
def shiftDown (g : ℝ → ℝ) (d : ℝ) : ℝ → ℝ := λ x ↦ g x - d

/-- The resulting function after shifts -/
def g (x : ℝ) : ℝ := (x + 1)^2 - 2

theorem quadratic_shift :
  shiftDown (shiftLeft f 2) 3 = g := by sorry

end NUMINAMATH_CALUDE_quadratic_shift_l3677_367793


namespace NUMINAMATH_CALUDE_lottery_is_systematic_sampling_l3677_367729

-- Define the lottery range
def lottery_range : Set ℕ := {n | 0 ≤ n ∧ n < 100000}

-- Define the winning number criteria
def is_winning_number (n : ℕ) : Prop :=
  n ∈ lottery_range ∧ (n % 100 = 88 ∨ n % 100 = 68)

-- Define systematic sampling
def systematic_sampling (S : Set ℕ) (f : ℕ → Prop) : Prop :=
  ∃ k : ℕ, k > 0 ∧ ∀ n ∈ S, f n ↔ ∃ m : ℕ, n = m * k

-- Theorem statement
theorem lottery_is_systematic_sampling :
  systematic_sampling lottery_range is_winning_number := by
  sorry


end NUMINAMATH_CALUDE_lottery_is_systematic_sampling_l3677_367729


namespace NUMINAMATH_CALUDE_y_squared_value_l3677_367733

theorem y_squared_value (x y : ℤ) 
  (eq1 : 4 * x + y = 34) 
  (eq2 : 2 * x - y = 20) : 
  y ^ 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_y_squared_value_l3677_367733


namespace NUMINAMATH_CALUDE_x_squared_mod_25_l3677_367762

theorem x_squared_mod_25 (x : ℤ) (h1 : 5 * x ≡ 15 [ZMOD 25]) (h2 : 4 * x ≡ 12 [ZMOD 25]) :
  x^2 ≡ 9 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_25_l3677_367762


namespace NUMINAMATH_CALUDE_one_minus_repeating_third_equals_two_thirds_l3677_367746

-- Define the repeating decimal 0.3333...
def repeating_third : ℚ := 1/3

-- Theorem statement
theorem one_minus_repeating_third_equals_two_thirds :
  1 - repeating_third = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_one_minus_repeating_third_equals_two_thirds_l3677_367746


namespace NUMINAMATH_CALUDE_subset_sum_exists_l3677_367790

theorem subset_sum_exists (A : ℕ) (h1 : ∀ i ∈ Finset.range 9, A % (i + 1) = 0)
  (h2 : ∃ (S : Finset ℕ), (∀ x ∈ S, x ∈ Finset.range 9) ∧ S.sum id = 2 * A) :
  ∃ (T : Finset ℕ), T ⊆ S ∧ T.sum id = A :=
sorry

end NUMINAMATH_CALUDE_subset_sum_exists_l3677_367790


namespace NUMINAMATH_CALUDE_hilt_bread_flour_l3677_367743

/-- The amount of flour needed for baking bread -/
def flour_for_bread (loaves : ℕ) (flour_per_loaf : ℚ) : ℚ :=
  loaves * flour_per_loaf

/-- Theorem: Mrs. Hilt needs 5 cups of flour to bake 2 loaves of bread -/
theorem hilt_bread_flour :
  flour_for_bread 2 (5/2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_hilt_bread_flour_l3677_367743


namespace NUMINAMATH_CALUDE_drink_composition_l3677_367760

theorem drink_composition (coke sprite mountain_dew : ℕ) 
  (h1 : coke = 2)
  (h2 : sprite = 1)
  (h3 : mountain_dew = 3)
  (h4 : (6 : ℚ) / (coke / (coke + sprite + mountain_dew)) = 18) :
  (6 : ℚ) / ((coke : ℚ) / (coke + sprite + mountain_dew)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_drink_composition_l3677_367760


namespace NUMINAMATH_CALUDE_markup_percentage_l3677_367723

/-- Proves that given a cost price of 540, a selling price of 459, and a discount percentage
    of 26.08695652173913%, the percentage marked above the cost price is 15%. -/
theorem markup_percentage
  (cost_price : ℝ)
  (selling_price : ℝ)
  (discount_percentage : ℝ)
  (h_cost_price : cost_price = 540)
  (h_selling_price : selling_price = 459)
  (h_discount_percentage : discount_percentage = 26.08695652173913) :
  let marked_price := selling_price / (1 - discount_percentage / 100)
  (marked_price - cost_price) / cost_price * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_markup_percentage_l3677_367723


namespace NUMINAMATH_CALUDE_not_cheap_necessary_for_good_quality_l3677_367783

-- Define the propositions
variable (cheap : Prop) (good_quality : Prop)

-- Define the given condition
axiom cheap_implies_not_good : cheap → ¬good_quality

-- Theorem to prove
theorem not_cheap_necessary_for_good_quality :
  good_quality → ¬cheap :=
sorry

end NUMINAMATH_CALUDE_not_cheap_necessary_for_good_quality_l3677_367783


namespace NUMINAMATH_CALUDE_dans_remaining_money_l3677_367773

/-- 
Given an initial amount of money and the cost of a candy bar, 
calculate the remaining amount after purchasing the candy bar.
-/
def remaining_money (initial_amount : ℝ) (candy_cost : ℝ) : ℝ :=
  initial_amount - candy_cost

/-- 
Theorem: Given an initial amount of $4 and a candy bar cost of $1, 
the remaining amount after purchasing the candy bar is $3.
-/
theorem dans_remaining_money : 
  remaining_money 4 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_money_l3677_367773


namespace NUMINAMATH_CALUDE_nilpotent_matrix_square_zero_l3677_367749

theorem nilpotent_matrix_square_zero 
  (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A ^ 4 = 0) : 
  A ^ 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_nilpotent_matrix_square_zero_l3677_367749


namespace NUMINAMATH_CALUDE_heptagon_exterior_angles_sum_l3677_367764

-- Define a heptagon
structure Heptagon where
  sides : Fin 7 → ℝ × ℝ

-- Define the sum of exterior angles
def sum_of_exterior_angles (h : Heptagon) : ℝ := sorry

-- Theorem statement
theorem heptagon_exterior_angles_sum (h : Heptagon) :
  sum_of_exterior_angles h = 360 := by sorry

end NUMINAMATH_CALUDE_heptagon_exterior_angles_sum_l3677_367764


namespace NUMINAMATH_CALUDE_watch_sale_gain_percentage_l3677_367756

/-- Calculates the selling price given the cost price and loss percentage -/
def sellingPriceWithLoss (costPrice : ℚ) (lossPercentage : ℚ) : ℚ :=
  costPrice * (1 - lossPercentage / 100)

/-- Calculates the gain percentage given the cost price and selling price -/
def gainPercentage (costPrice : ℚ) (sellingPrice : ℚ) : ℚ :=
  (sellingPrice - costPrice) / costPrice * 100

theorem watch_sale_gain_percentage 
  (costPrice : ℚ) 
  (lossPercentage : ℚ) 
  (additionalAmount : ℚ) : 
  costPrice = 3000 →
  lossPercentage = 10 →
  additionalAmount = 540 →
  gainPercentage costPrice (sellingPriceWithLoss costPrice lossPercentage + additionalAmount) = 8 := by
  sorry

end NUMINAMATH_CALUDE_watch_sale_gain_percentage_l3677_367756


namespace NUMINAMATH_CALUDE_ellipse_equation_parabola_equation_l3677_367753

-- Problem 1
theorem ellipse_equation (focal_distance : ℝ) (point : ℝ × ℝ) : 
  focal_distance = 4 ∧ point = (3, -2 * Real.sqrt 6) →
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧
    (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ x^2/36 + y^2/32 = 1) :=
sorry

-- Problem 2
theorem parabola_equation (hyperbola : ℝ → ℝ → Prop) (directrix : ℝ) :
  (∀ x y : ℝ, hyperbola x y ↔ x^2 - y^2/3 = 1) ∧
  directrix = -1/2 →
  ∀ x y : ℝ, y^2 = 2*x ↔ y^2 = 2*x ∧ x ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_parabola_equation_l3677_367753


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_in_square_midpoint_triangle_l3677_367770

/-- Given a square with side length 12, this theorem proves that the radius of the circle
    inscribed in the triangle formed by connecting midpoints of adjacent sides to each other
    and to the opposite side is 2√5 - √2. -/
theorem inscribed_circle_radius_in_square_midpoint_triangle :
  let square_side : ℝ := 12
  let midpoint_triangle_area : ℝ := 54
  let midpoint_triangle_semiperimeter : ℝ := 6 * Real.sqrt 5 + 3 * Real.sqrt 2
  let inscribed_circle_radius : ℝ := midpoint_triangle_area / midpoint_triangle_semiperimeter
  inscribed_circle_radius = 2 * Real.sqrt 5 - Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_radius_in_square_midpoint_triangle_l3677_367770


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3677_367719

/-- Given positive real numbers c and d where c > d, the sum of the infinite series
    1/(cd) + 1/(c(3c-d)) + 1/((3c-d)(5c-2d)) + 1/((5c-2d)(7c-3d)) + ...
    is equal to 1/((c-d)d). -/
theorem infinite_series_sum (c d : ℝ) (hc : c > 0) (hd : d > 0) (h : c > d) :
  let series := fun n : ℕ => 1 / ((2 * n - 1) * c - (n - 1) * d) / ((2 * n + 1) * c - n * d)
  ∑' n, series n = 1 / ((c - d) * d) := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3677_367719


namespace NUMINAMATH_CALUDE_roots_expression_l3677_367702

theorem roots_expression (p q : ℝ) (α β γ δ : ℝ) 
  (hαβ : α^2 + p*α - 1 = 0 ∧ β^2 + p*β - 1 = 0)
  (hγδ : γ^2 + q*γ - 1 = 0 ∧ δ^2 + q*δ - 1 = 0) :
  (α - γ)*(β - γ)*(α - δ)*(β - δ) = -(p - q)^2 := by
sorry

end NUMINAMATH_CALUDE_roots_expression_l3677_367702


namespace NUMINAMATH_CALUDE_max_value_implies_a_range_l3677_367771

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - (a + 2) * x

theorem max_value_implies_a_range (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x > 0, f a x ≤ f a (1/2)) : 0 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_a_range_l3677_367771


namespace NUMINAMATH_CALUDE_fifteen_percent_less_than_80_l3677_367735

theorem fifteen_percent_less_than_80 : ∃ x : ℝ, x + (1/4) * x = 80 - 0.15 * 80 ∧ x = 54 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_less_than_80_l3677_367735


namespace NUMINAMATH_CALUDE_composite_product_equals_twelve_over_pi_squared_l3677_367780

-- Define the sequence of composite numbers
def composite : ℕ → ℕ
  | 0 => 4  -- First composite number
  | n + 1 => sorry  -- Definition of subsequent composite numbers

-- Define the infinite product
def infinite_product : ℝ := sorry

-- Define the infinite sum of reciprocal squares
def reciprocal_squares_sum : ℝ := sorry

-- Theorem statement
theorem composite_product_equals_twelve_over_pi_squared :
  (reciprocal_squares_sum = Real.pi^2 / 6) →
  infinite_product = 12 / Real.pi^2 := by
  sorry

end NUMINAMATH_CALUDE_composite_product_equals_twelve_over_pi_squared_l3677_367780


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_expression_l3677_367705

theorem greatest_prime_factor_of_expression : 
  ∃ p : ℕ, p.Prime ∧ p ∣ (3^8 + 6^7) ∧ ∀ q : ℕ, q.Prime → q ∣ (3^8 + 6^7) → q ≤ p ∧ p = 131 :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_expression_l3677_367705


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l3677_367785

theorem right_triangle_third_side (a b x : ℝ) :
  (a - 3)^2 + |b - 4| = 0 →
  (x^2 = a^2 + b^2 ∨ x^2 + a^2 = b^2 ∨ x^2 + b^2 = a^2) →
  x = 5 ∨ x = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l3677_367785


namespace NUMINAMATH_CALUDE_hyperbola_iff_m_negative_l3677_367769

/-- A conic section represented by the equation x^2 + my^2 = m -/
structure ConicSection (m : ℝ) where
  x : ℝ
  y : ℝ
  eq : x^2 + m * y^2 = m

/-- Predicate to determine if a conic section is a hyperbola -/
def IsHyperbola (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 + m * y^2 = m

/-- Theorem stating that the equation x^2 + my^2 = m represents a hyperbola if and only if m < 0 -/
theorem hyperbola_iff_m_negative (m : ℝ) : IsHyperbola m ↔ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_iff_m_negative_l3677_367769


namespace NUMINAMATH_CALUDE_binomial_coefficient_seven_two_l3677_367782

theorem binomial_coefficient_seven_two : Nat.choose 7 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_seven_two_l3677_367782


namespace NUMINAMATH_CALUDE_candy_distribution_theorem_l3677_367768

/-- Represents the number of candy bars of each type --/
structure CandyBars where
  chocolate : ℕ
  caramel : ℕ
  nougat : ℕ

/-- Represents the ratio of candy bars in a bag --/
structure BagRatio where
  chocolate : ℕ
  caramel : ℕ
  nougat : ℕ

/-- Checks if the given ratio is valid for the total number of candy bars --/
def isValidRatio (total : CandyBars) (ratio : BagRatio) (bags : ℕ) : Prop :=
  total.chocolate = ratio.chocolate * bags ∧
  total.caramel = ratio.caramel * bags ∧
  total.nougat = ratio.nougat * bags

/-- The main theorem to be proved --/
theorem candy_distribution_theorem (total : CandyBars) 
  (h1 : total.chocolate = 12) 
  (h2 : total.caramel = 18) 
  (h3 : total.nougat = 15) :
  ∃ (ratio : BagRatio) (bags : ℕ), 
    bags = 5 ∧ 
    ratio.chocolate = 2 ∧ 
    ratio.caramel = 3 ∧ 
    ratio.nougat = 3 ∧
    isValidRatio total ratio bags ∧
    ∀ (other_ratio : BagRatio) (other_bags : ℕ), 
      isValidRatio total other_ratio other_bags → other_bags ≤ bags :=
by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_theorem_l3677_367768


namespace NUMINAMATH_CALUDE_no_prime_generating_pair_l3677_367710

theorem no_prime_generating_pair : ∀ a b : ℕ+, ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ p > 1000 ∧ q > 1000 ∧ p ≠ q ∧ ¬(Prime (a * p + b * q)) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_generating_pair_l3677_367710


namespace NUMINAMATH_CALUDE_inverse_function_range_l3677_367721

def is_inverse_function (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) * f (a - x) = 1

theorem inverse_function_range 
  (f : ℝ → ℝ) 
  (h0 : is_inverse_function f 0)
  (h1 : is_inverse_function f 1)
  (h_range : ∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc 1 2) :
  ∀ x ∈ Set.Icc (-2016) 2016, f x ∈ Set.Icc (1/2) 2 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_range_l3677_367721


namespace NUMINAMATH_CALUDE_function_is_identity_l3677_367774

-- Define the property that the function f must satisfy
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (2 * x + f y) = x + y + f x

-- Theorem statement
theorem function_is_identity 
  (f : ℝ → ℝ) 
  (h : satisfies_equation f) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_is_identity_l3677_367774


namespace NUMINAMATH_CALUDE_election_winner_votes_l3677_367757

theorem election_winner_votes 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (vote_difference : ℕ) :
  winner_percentage = 55 / 100 →
  vote_difference = 100 →
  (winner_percentage * total_votes).num = 
    (1 - winner_percentage) * total_votes + vote_difference →
  (winner_percentage * total_votes).num = 550 := by
sorry

end NUMINAMATH_CALUDE_election_winner_votes_l3677_367757


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l3677_367750

theorem logarithm_expression_equality : 
  9^(Real.log 2 / Real.log 3) - 4 * (Real.log 3 / Real.log 4) * (Real.log 8 / Real.log 27) + 
  (1/3) * (Real.log 8 / Real.log 6) - 2 * (Real.log (Real.sqrt 3) / Real.log (1/6)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l3677_367750


namespace NUMINAMATH_CALUDE_common_inner_tangent_l3677_367713

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 16
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y + 24 = 0

-- Define the proposed tangent line
def tangent_line (x y : ℝ) : Prop := 3*x - 4*y - 20 = 0

-- Theorem statement
theorem common_inner_tangent :
  ∀ x y : ℝ, 
  (circle1 x y ∨ circle2 x y) → 
  (tangent_line x y ↔ 
    (∃ t : ℝ, 
      (circle1 (x + t) (y + t) ∧ tangent_line (x + t) (y + t)) ∨
      (circle2 (x + t) (y + t) ∧ tangent_line (x + t) (y + t))))
  := by sorry

end NUMINAMATH_CALUDE_common_inner_tangent_l3677_367713


namespace NUMINAMATH_CALUDE_power_of_two_greater_than_n_and_factorial_greater_than_power_of_two_l3677_367777

theorem power_of_two_greater_than_n_and_factorial_greater_than_power_of_two :
  (∀ n : ℕ, 2^n > n) ∧
  (∀ n : ℕ, n ≥ 4 → n.factorial > 2^n) := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_greater_than_n_and_factorial_greater_than_power_of_two_l3677_367777


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3677_367708

theorem age_ratio_proof (parent_age son_age : ℕ) : 
  parent_age = 45 →
  son_age = 15 →
  parent_age + 5 = (5/2) * (son_age + 5) →
  parent_age / son_age = 3 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3677_367708


namespace NUMINAMATH_CALUDE_second_class_end_time_l3677_367754

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60
  , minutes := totalMinutes % 60 }

theorem second_class_end_time :
  let start_time : Time := { hours := 9, minutes := 25 }
  let class_duration : Nat := 35
  let end_time := addMinutes start_time class_duration
  end_time = { hours := 10, minutes := 0 } := by
  sorry

end NUMINAMATH_CALUDE_second_class_end_time_l3677_367754


namespace NUMINAMATH_CALUDE_line_slope_is_two_l3677_367725

/-- A line in the xy-plane with y-intercept 2 and passing through (239, 480) has slope 2 -/
theorem line_slope_is_two :
  ∀ (m : ℝ) (f : ℝ → ℝ),
  (∀ x, f x = m * x + 2) →  -- Line equation with y-intercept 2
  f 239 = 480 →            -- Line passes through (239, 480)
  m = 2 :=                 -- Slope is 2
by
  sorry

end NUMINAMATH_CALUDE_line_slope_is_two_l3677_367725


namespace NUMINAMATH_CALUDE_inequality_range_l3677_367732

theorem inequality_range (t : ℝ) (h1 : t > 0) :
  (∀ x > 0, Real.exp (2 * t * x) - (Real.log 2 + Real.log x) / t ≥ 0) ↔ t ≥ 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l3677_367732


namespace NUMINAMATH_CALUDE_remainder_3m_mod_5_l3677_367716

theorem remainder_3m_mod_5 (m : ℤ) (h : m % 5 = 2) : (3 * m) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3m_mod_5_l3677_367716


namespace NUMINAMATH_CALUDE_grass_seed_cost_l3677_367740

/-- Represents the cost and weight of a bag of grass seed -/
structure GrassSeedBag where
  weight : Nat
  cost : Float

/-- Represents a purchase of grass seed bags -/
structure Purchase where
  bags : List GrassSeedBag
  totalWeight : Nat
  totalCost : Float

def tenPoundBag : GrassSeedBag := { weight := 10, cost := 20.43 }
def twentyFivePoundBag : GrassSeedBag := { weight := 25, cost := 32.20 }

/-- The optimal purchase satisfying the given conditions -/
def optimalPurchase (fivePoundBagCost : Float) : Purchase :=
  { bags := [twentyFivePoundBag, twentyFivePoundBag, twentyFivePoundBag, 
             { weight := 5, cost := fivePoundBagCost }],
    totalWeight := 80,
    totalCost := 3 * 32.20 + fivePoundBagCost }

theorem grass_seed_cost 
  (h1 : ∀ p : Purchase, p.totalWeight ≥ 65 ∧ p.totalWeight ≤ 80 → p.totalCost ≥ 98.68)
  (h2 : (optimalPurchase 2.08).totalCost = 98.68) :
  ∃ fivePoundBagCost : Float, fivePoundBagCost = 2.08 ∧ 
    (optimalPurchase fivePoundBagCost).totalCost = 98.68 ∧
    (optimalPurchase fivePoundBagCost).totalWeight ≥ 65 ∧
    (optimalPurchase fivePoundBagCost).totalWeight ≤ 80 := by
  sorry

end NUMINAMATH_CALUDE_grass_seed_cost_l3677_367740


namespace NUMINAMATH_CALUDE_inequality_proof_l3677_367711

theorem inequality_proof (a b : ℤ) 
  (h1 : a > b) 
  (h2 : b > 1) 
  (h3 : (a + b) ∣ (a * b + 1)) 
  (h4 : (a - b) ∣ (a * b - 1)) : 
  a < Real.sqrt 3 * b := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3677_367711


namespace NUMINAMATH_CALUDE_function_graphs_common_point_l3677_367796

/-- Given real numbers a, b, c, and d, if the graphs of y = 2a + 1/(x-b) and y = 2c + 1/(x-d) 
    have exactly one common point, then the graphs of y = 2b + 1/(x-a) and y = 2d + 1/(x-c) 
    also have exactly one common point. -/
theorem function_graphs_common_point (a b c d : ℝ) :
  (∃! x : ℝ, 2 * a + 1 / (x - b) = 2 * c + 1 / (x - d)) →
  (∃! x : ℝ, 2 * b + 1 / (x - a) = 2 * d + 1 / (x - c)) :=
by sorry

end NUMINAMATH_CALUDE_function_graphs_common_point_l3677_367796


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l3677_367767

theorem fraction_sum_equals_decimal : 
  (3 : ℚ) / 100 + 5 / 1000 + 8 / 10000 + 2 / 100000 = 0.03582 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l3677_367767


namespace NUMINAMATH_CALUDE_giraffe_count_prove_giraffe_count_l3677_367739

/-- The number of giraffes at a zoo, given certain conditions. -/
theorem giraffe_count : ℕ → ℕ → Prop :=
  fun (giraffes : ℕ) (other_animals : ℕ) =>
    giraffes = 3 * other_animals ∧
    giraffes = other_animals + 290 →
    giraffes = 435

/-- Proof of the giraffe count theorem. -/
theorem prove_giraffe_count : ∃ (g o : ℕ), giraffe_count g o :=
  sorry

end NUMINAMATH_CALUDE_giraffe_count_prove_giraffe_count_l3677_367739


namespace NUMINAMATH_CALUDE_smallest_gcd_bc_l3677_367751

theorem smallest_gcd_bc (a b c : ℕ+) (hab : Nat.gcd a b = 168) (hac : Nat.gcd a c = 693) :
  ∃ (d : ℕ), d = Nat.gcd b c ∧ d ≥ 21 ∧ ∀ (e : ℕ), e = Nat.gcd b c → e ≥ d :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_bc_l3677_367751


namespace NUMINAMATH_CALUDE_equation_solutions_l3677_367759

theorem equation_solutions :
  (∀ x : ℝ, 6*x - 7 = 4*x - 5 ↔ x = 1) ∧
  (∀ x : ℝ, 5*(x + 8) - 5 = 6*(2*x - 7) ↔ x = 11) ∧
  (∀ x : ℝ, x - (x - 1)/2 = 2 - (x + 2)/5 ↔ x = 11/7) ∧
  (∀ x : ℝ, x^2 - 64 = 0 ↔ x = 8 ∨ x = -8) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3677_367759


namespace NUMINAMATH_CALUDE_circle_area_ratio_l3677_367745

/-- Given a triangle with sides 13, 14, and 15, the ratio of the area of its 
    circumscribed circle to the area of its inscribed circle is (65/32)^2 -/
theorem circle_area_ratio (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) :
  let p := (a + b + c) / 2
  let s := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let r := s / p
  let R := (a * b * c) / (4 * s)
  (R / r) ^ 2 = (65 / 32) ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_circle_area_ratio_l3677_367745


namespace NUMINAMATH_CALUDE_one_third_to_fifth_power_l3677_367736

theorem one_third_to_fifth_power :
  (1 / 3 : ℚ) ^ 5 = 1 / 243 := by sorry

end NUMINAMATH_CALUDE_one_third_to_fifth_power_l3677_367736


namespace NUMINAMATH_CALUDE_equation_solution_l3677_367714

theorem equation_solution : 
  ∀ x : ℝ, (x + 2) * (x + 1) = 3 * (x + 1) ↔ x = -1 ∨ x = 1 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3677_367714


namespace NUMINAMATH_CALUDE_f_equals_g_l3677_367795

/-- Two functions are considered the same if they have the same domain, codomain, and function value for all inputs. -/
def same_function (α β : Type) (f g : α → β) : Prop :=
  ∀ x, f x = g x

/-- Function f defined as f(x) = x - 1 -/
def f : ℝ → ℝ := λ x ↦ x - 1

/-- Function g defined as g(t) = t - 1 -/
def g : ℝ → ℝ := λ t ↦ t - 1

/-- Theorem stating that f and g are the same function -/
theorem f_equals_g : same_function ℝ ℝ f g := by
  sorry


end NUMINAMATH_CALUDE_f_equals_g_l3677_367795


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l3677_367789

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1 / a + 4 / b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l3677_367789


namespace NUMINAMATH_CALUDE_sqrt_sum_difference_eighth_power_l3677_367700

theorem sqrt_sum_difference_eighth_power : 
  (Real.sqrt 11 + Real.sqrt 5)^8 + (Real.sqrt 11 - Real.sqrt 5)^8 = 903712 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_difference_eighth_power_l3677_367700


namespace NUMINAMATH_CALUDE_square_perimeter_sum_l3677_367747

theorem square_perimeter_sum (a b : ℝ) (h1 : a + b = 130) (h2 : a - b = 42) :
  4 * (Real.sqrt a.toNNReal + Real.sqrt b.toNNReal) = 4 * (Real.sqrt 86 + 2 * Real.sqrt 11) :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_sum_l3677_367747


namespace NUMINAMATH_CALUDE_triangle_existence_l3677_367715

theorem triangle_existence (y : ℕ+) : 
  (∃ (a b c : ℝ), a = 8 ∧ b = 12 ∧ c = y.val^2 ∧ 
   a + b > c ∧ a + c > b ∧ b + c > a) ↔ (y = 3 ∨ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_existence_l3677_367715


namespace NUMINAMATH_CALUDE_prob_hit_both_l3677_367730

variable (p : ℝ)

-- Define the probability of hitting a single basket in 6 throws
def prob_hit_single (p : ℝ) : ℝ := 1 - (1 - p)^6

-- Define the probability of hitting at least one of two baskets in 6 throws
def prob_hit_at_least_one (p : ℝ) : ℝ := 1 - (1 - 2*p)^6

-- State the theorem
theorem prob_hit_both (hp : 0 ≤ p ∧ p ≤ 1/2) :
  prob_hit_single p + prob_hit_single p - prob_hit_at_least_one p = 1 - 2*(1 - p)^6 + (1 - 2*p)^6 := by
  sorry

end NUMINAMATH_CALUDE_prob_hit_both_l3677_367730


namespace NUMINAMATH_CALUDE_problem_statement_l3677_367794

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 3) :
  (∀ a b, a > 0 ∧ b > 0 ∧ a + 2*b = 3 → y/x + 3/y ≥ 4) ∧
  (∀ a b, a > 0 ∧ b > 0 ∧ a + 2*b = 3 → x*y ≤ 9/8) ∧
  (∀ ε > 0, ∃ a b, a > 0 ∧ b > 0 ∧ a + 2*b = 3 ∧ Real.sqrt a + Real.sqrt (2*b) > 2 - ε) ∧
  (∀ a b, a > 0 ∧ b > 0 ∧ a + 2*b = 3 → a^2 + 4*b^2 ≥ 9/2) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3677_367794


namespace NUMINAMATH_CALUDE_angles_with_same_terminal_side_l3677_367799

-- Define the range
def angle_range : Set ℝ := {x | -720 ≤ x ∧ x < 0}

-- Define the property of having the same terminal side as 45°
def same_terminal_side (θ : ℝ) : Prop :=
  ∃ k : ℤ, θ = 45 + k * 360

-- Theorem statement
theorem angles_with_same_terminal_side :
  {θ : ℝ | θ ∈ angle_range ∧ same_terminal_side θ} = {-675, -315} := by
  sorry

end NUMINAMATH_CALUDE_angles_with_same_terminal_side_l3677_367799


namespace NUMINAMATH_CALUDE_regular_20gon_symmetry_sum_l3677_367786

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Add any necessary fields here

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := sorry

/-- The smallest positive angle of rotational symmetry in degrees -/
def smallestRotationAngle (p : RegularPolygon n) : ℝ := sorry

theorem regular_20gon_symmetry_sum :
  ∀ (p : RegularPolygon 20),
    (linesOfSymmetry p : ℝ) + smallestRotationAngle p = 38 := by sorry

end NUMINAMATH_CALUDE_regular_20gon_symmetry_sum_l3677_367786


namespace NUMINAMATH_CALUDE_square_field_area_l3677_367752

/-- The area of a square field with side length 6 meters is 36 square meters. -/
theorem square_field_area :
  let side_length : ℝ := 6
  let field_area : ℝ := side_length ^ 2
  field_area = 36 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l3677_367752


namespace NUMINAMATH_CALUDE_prime_octuple_sum_product_relation_l3677_367724

theorem prime_octuple_sum_product_relation :
  ∀ (p₁ p₂ p₃ p₄ p₅ p₆ p₇ p₈ : ℕ),
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧
    Prime p₅ ∧ Prime p₆ ∧ Prime p₇ ∧ Prime p₈ →
    (p₁^2 + p₂^2 + p₃^2 + p₄^2 + p₅^2 + p₆^2 + p₇^2 + p₈^2 = 4 * (p₁ * p₂ * p₃ * p₄ * p₅ * p₆ * p₇ * p₈) - 992) →
    p₁ = 2 ∧ p₂ = 2 ∧ p₃ = 2 ∧ p₄ = 2 ∧ p₅ = 2 ∧ p₆ = 2 ∧ p₇ = 2 ∧ p₈ = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_prime_octuple_sum_product_relation_l3677_367724


namespace NUMINAMATH_CALUDE_det_of_matrix_l3677_367731

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![7, -2; -3, 6]

theorem det_of_matrix : Matrix.det matrix = 36 := by sorry

end NUMINAMATH_CALUDE_det_of_matrix_l3677_367731


namespace NUMINAMATH_CALUDE_gcd_problem_l3677_367784

/-- The greatest common divisor of (123^2 + 235^2 + 347^2) and (122^2 + 234^2 + 348^2) is 1 -/
theorem gcd_problem : Nat.gcd (123^2 + 235^2 + 347^2) (122^2 + 234^2 + 348^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3677_367784


namespace NUMINAMATH_CALUDE_tailor_cut_l3677_367766

theorem tailor_cut (skirt_cut pants_cut : ℝ) : 
  skirt_cut = 0.75 → 
  skirt_cut = pants_cut + 0.25 → 
  pants_cut = 0.50 := by
sorry

end NUMINAMATH_CALUDE_tailor_cut_l3677_367766


namespace NUMINAMATH_CALUDE_brianna_marbles_l3677_367755

/-- The number of marbles Brianna has remaining after a series of events -/
def remaining_marbles (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost - (2 * lost) - (lost / 2)

/-- Theorem stating that Brianna has 10 marbles remaining -/
theorem brianna_marbles : remaining_marbles 24 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_brianna_marbles_l3677_367755


namespace NUMINAMATH_CALUDE_sum_of_valid_a_l3677_367741

theorem sum_of_valid_a : ∃ (S : Finset ℤ), 
  (∀ a ∈ S, 
    (∃ x : ℝ, x > 0 ∧ x ≠ 3 ∧ (3*x - a)/(x - 3) + (x + 1)/(3 - x) = 1) ∧
    (∀ y : ℝ, (y + 9 ≤ 2*(y + 2) ∧ (2*y - a)/3 > 1) ↔ y ≥ 5)) ∧
  (∀ a : ℤ, 
    ((∃ x : ℝ, x > 0 ∧ x ≠ 3 ∧ (3*x - a)/(x - 3) + (x + 1)/(3 - x) = 1) ∧
    (∀ y : ℝ, (y + 9 ≤ 2*(y + 2) ∧ (2*y - a)/3 > 1) ↔ y ≥ 5)) → a ∈ S) ∧
  Finset.sum S id = 13 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_valid_a_l3677_367741


namespace NUMINAMATH_CALUDE_num_tuba_players_l3677_367763

/-- The weight carried by each trumpet or clarinet player -/
def trumpet_clarinet_weight : ℕ := 5

/-- The weight carried by each trombone player -/
def trombone_weight : ℕ := 10

/-- The weight carried by each tuba player -/
def tuba_weight : ℕ := 20

/-- The weight carried by each drum player -/
def drum_weight : ℕ := 15

/-- The number of trumpet players -/
def num_trumpets : ℕ := 6

/-- The number of clarinet players -/
def num_clarinets : ℕ := 9

/-- The number of trombone players -/
def num_trombones : ℕ := 8

/-- The number of drum players -/
def num_drums : ℕ := 2

/-- The total weight carried by the marching band -/
def total_weight : ℕ := 245

/-- Theorem: The number of tuba players in the marching band is 3 -/
theorem num_tuba_players : 
  ∃ (n : ℕ), n * tuba_weight = 
    total_weight - 
    (num_trumpets * trumpet_clarinet_weight + 
     num_clarinets * trumpet_clarinet_weight + 
     num_trombones * trombone_weight + 
     num_drums * drum_weight) ∧ 
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_num_tuba_players_l3677_367763


namespace NUMINAMATH_CALUDE_tangent_line_to_ln_l3677_367779

-- Define the natural logarithm function
noncomputable def ln : ℝ → ℝ := Real.log

-- Define the tangent line
def tangent_line (a : ℝ) (x : ℝ) : ℝ := x + a

-- State the theorem
theorem tangent_line_to_ln (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ 
    tangent_line a x = ln x ∧ 
    (∀ y : ℝ, y > 0 → tangent_line a y ≥ ln y)) →
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_ln_l3677_367779


namespace NUMINAMATH_CALUDE_complex_fraction_ratio_l3677_367791

theorem complex_fraction_ratio : 
  let z : ℂ := (2 + I) / I
  ∃ a b : ℝ, z = a + b * I ∧ b / a = -2 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_ratio_l3677_367791


namespace NUMINAMATH_CALUDE_jacks_purchase_cost_l3677_367776

/-- The cost of Jack's purchase of a squat rack and barbell -/
theorem jacks_purchase_cost (squat_rack_cost : ℝ) (barbell_cost : ℝ) : 
  squat_rack_cost = 2500 →
  barbell_cost = squat_rack_cost / 10 →
  squat_rack_cost + barbell_cost = 2750 := by
sorry

end NUMINAMATH_CALUDE_jacks_purchase_cost_l3677_367776


namespace NUMINAMATH_CALUDE_ball_distribution_probability_ratio_l3677_367781

theorem ball_distribution_probability_ratio :
  let total_balls : ℕ := 25
  let total_bins : ℕ := 6
  let p := (Nat.choose total_bins 2 * Nat.choose total_balls 3 * Nat.choose (total_balls - 3) 3 *
            Nat.choose (total_balls - 6) 4 * Nat.choose (total_balls - 10) 4 *
            Nat.choose (total_balls - 14) 4 * Nat.choose (total_balls - 18) 4) / 
           (Nat.factorial 4 * Nat.pow total_bins total_balls)
  let q := (Nat.choose total_bins 1 * Nat.choose total_balls 5 *
            Nat.choose (total_balls - 5) 4 * Nat.choose (total_balls - 9) 4 *
            Nat.choose (total_balls - 13) 4 * Nat.choose (total_balls - 17) 4 *
            Nat.choose (total_balls - 21) 4) / 
           Nat.pow total_bins total_balls
  p / q = 8 := by
sorry

end NUMINAMATH_CALUDE_ball_distribution_probability_ratio_l3677_367781


namespace NUMINAMATH_CALUDE_complex_magnitude_l3677_367734

theorem complex_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 30)
  (h2 : Complex.abs (z + 3 * w) = 15)
  (h3 : Complex.abs (z - w) = 10) :
  Complex.abs z = 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3677_367734


namespace NUMINAMATH_CALUDE_table_length_is_77_l3677_367728

/-- The length of a rectangular table covered with sheets of paper -/
def table_length (table_width sheet_width sheet_height : ℕ) : ℕ :=
  sheet_height + (table_width - sheet_width)

/-- Theorem stating that the length of the table is 77 cm -/
theorem table_length_is_77 :
  table_length 80 8 5 = 77 :=
by sorry

end NUMINAMATH_CALUDE_table_length_is_77_l3677_367728


namespace NUMINAMATH_CALUDE_johns_hotel_cost_l3677_367765

/-- Calculates the total cost of a hotel stay with a discount -/
def hotel_cost (nights : ℕ) (cost_per_night : ℕ) (discount : ℕ) : ℕ :=
  nights * cost_per_night - discount

/-- Proves that John's 3-night stay at $250 per night with a $100 discount costs $650 -/
theorem johns_hotel_cost : hotel_cost 3 250 100 = 650 := by
  sorry

end NUMINAMATH_CALUDE_johns_hotel_cost_l3677_367765


namespace NUMINAMATH_CALUDE_equal_population_after_14_years_second_village_initial_population_is_correct_l3677_367748

/-- The initial population of Village X -/
def village_x_initial : ℕ := 70000

/-- The yearly decrease in population of Village X -/
def village_x_decrease : ℕ := 1200

/-- The yearly increase in population of the second village -/
def village_2_increase : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years_until_equal : ℕ := 14

/-- The initial population of the second village -/
def village_2_initial : ℕ := 42000

theorem equal_population_after_14_years :
  village_x_initial - village_x_decrease * years_until_equal = 
  village_2_initial + village_2_increase * years_until_equal :=
by sorry

/-- The theorem stating that the calculated initial population of the second village is correct -/
theorem second_village_initial_population_is_correct : village_2_initial = 42000 :=
by sorry

end NUMINAMATH_CALUDE_equal_population_after_14_years_second_village_initial_population_is_correct_l3677_367748


namespace NUMINAMATH_CALUDE_inverse_functions_l3677_367788

-- Define the properties of each function
def is_v_shaped (f : ℝ → ℝ) : Prop := sorry

def is_closed_loop (f : ℝ → ℝ) : Prop := sorry

def is_linear_increasing (f : ℝ → ℝ) : Prop := sorry

def is_semicircle_left (f : ℝ → ℝ) : Prop := sorry

def is_semicircle_right (f : ℝ → ℝ) : Prop := sorry

def is_cubic (f : ℝ → ℝ) : Prop := sorry

-- Define the condition for a function to have an inverse
def has_inverse (f : ℝ → ℝ) : Prop := sorry

-- Define the six functions
def F : ℝ → ℝ := sorry
def G : ℝ → ℝ := sorry
def H : ℝ → ℝ := sorry
def I : ℝ → ℝ := sorry
def J : ℝ → ℝ := sorry
def K : ℝ → ℝ := sorry

-- Theorem statement
theorem inverse_functions :
  is_v_shaped F ∧
  is_closed_loop G ∧
  is_linear_increasing H ∧
  is_semicircle_left I ∧
  is_semicircle_right J ∧
  is_cubic K →
  (has_inverse F ∧ has_inverse H ∧ has_inverse I ∧ has_inverse J) ∧
  (¬ has_inverse G ∧ ¬ has_inverse K) := by
  sorry

end NUMINAMATH_CALUDE_inverse_functions_l3677_367788


namespace NUMINAMATH_CALUDE_project_hours_theorem_l3677_367717

theorem project_hours_theorem (kate mark pat : ℕ) 
  (h1 : pat = 2 * kate)
  (h2 : 3 * pat = mark)
  (h3 : mark = kate + 120) :
  kate + mark + pat = 216 := by
  sorry

end NUMINAMATH_CALUDE_project_hours_theorem_l3677_367717


namespace NUMINAMATH_CALUDE_solution_set_l3677_367726

def satisfies_equations (x y : ℝ) : Prop :=
  y^2 - y*x^2 = 0 ∧ x^5 + x^4 = 0

theorem solution_set :
  ∀ x y : ℝ, satisfies_equations x y ↔ (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = 0) ∨ (x = -1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l3677_367726


namespace NUMINAMATH_CALUDE_hoseok_position_l3677_367704

theorem hoseok_position (n : Nat) (h : n = 9) :
  ∀ (position_tallest : Nat), position_tallest = 5 →
    n + 1 - position_tallest = 5 :=
by sorry

end NUMINAMATH_CALUDE_hoseok_position_l3677_367704


namespace NUMINAMATH_CALUDE_symmetry_about_x_equals_one_l3677_367707

/-- Given a function f(x) = x, if its graph is symmetric about the line x = 1,
    then the corresponding function g(x) is equal to 3x - 2. -/
theorem symmetry_about_x_equals_one (f g : ℝ → ℝ) :
  (∀ x, f x = x) →
  (∀ x, f (2 - x) = g x) →
  (∀ x, g x = 3*x - 2) := by
sorry

end NUMINAMATH_CALUDE_symmetry_about_x_equals_one_l3677_367707


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3677_367775

theorem fraction_evaluation : (3^6 - 9 * 3^3 + 27) / (3^3 - 3) = 24.75 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3677_367775


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3677_367737

/-- Definition of a point in the fourth quadrant -/
def in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

/-- The point (3, -2) -/
def point : ℝ × ℝ := (3, -2)

/-- Theorem: The point (3, -2) is in the fourth quadrant -/
theorem point_in_fourth_quadrant : 
  in_fourth_quadrant point.1 point.2 := by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3677_367737


namespace NUMINAMATH_CALUDE_sqrt_two_minus_a_l3677_367742

theorem sqrt_two_minus_a (a : ℝ) : a = -2 → Real.sqrt (2 - a) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_a_l3677_367742
