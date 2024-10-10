import Mathlib

namespace no_solution_exists_l3573_357374

theorem no_solution_exists : ¬∃ (x : ℝ), 3 * (2*x)^2 - 2 * (2*x) + 5 = 2 * (6*x^2 - 3*(2*x) + 3) := by
  sorry

end no_solution_exists_l3573_357374


namespace congruence_and_infinite_primes_l3573_357384

theorem congruence_and_infinite_primes (p : ℕ) (hp : Prime p) (hp3 : p > 3) :
  (∃ x : ℕ, (x^2 + x + 1) % p = 0) →
  (p % 6 = 1 ∧ ∀ n : ℕ, ∃ q > n, Prime q ∧ q % 6 = 1) := by
  sorry

end congruence_and_infinite_primes_l3573_357384


namespace jim_travels_two_miles_l3573_357326

def john_distance : ℝ := 15

def jill_distance (john_dist : ℝ) : ℝ := john_dist - 5

def jim_distance (jill_dist : ℝ) : ℝ := 0.20 * jill_dist

theorem jim_travels_two_miles :
  jim_distance (jill_distance john_distance) = 2 := by
  sorry

end jim_travels_two_miles_l3573_357326


namespace min_people_needed_is_30_l3573_357308

/-- Represents the types of vehicles --/
inductive VehicleType
| SmallCar
| MediumCar
| LargeCar
| LightTruck
| HeavyTruck

/-- Returns the weight of a vehicle type in pounds --/
def vehicleWeight (v : VehicleType) : ℕ :=
  match v with
  | .SmallCar => 2000
  | .MediumCar => 3000
  | .LargeCar => 4000
  | .LightTruck => 10000
  | .HeavyTruck => 15000

/-- Represents the fleet of vehicles --/
def fleet : List (VehicleType × ℕ) :=
  [(VehicleType.SmallCar, 2), (VehicleType.MediumCar, 2), (VehicleType.LargeCar, 2),
   (VehicleType.LightTruck, 1), (VehicleType.HeavyTruck, 2)]

/-- The maximum lifting capacity of a person in pounds --/
def maxLiftingCapacity : ℕ := 1000

/-- Calculates the total weight of the fleet --/
def totalFleetWeight : ℕ :=
  fleet.foldl (fun acc (v, count) => acc + vehicleWeight v * count) 0

/-- Theorem: The minimum number of people needed to lift all vehicles is 30 --/
theorem min_people_needed_is_30 :
  ∃ (n : ℕ), n = 30 ∧
  n * maxLiftingCapacity ≥ totalFleetWeight ∧
  ∀ (m : ℕ), m * maxLiftingCapacity ≥ totalFleetWeight → m ≥ n :=
sorry

end min_people_needed_is_30_l3573_357308


namespace cryptarithmetic_solution_l3573_357373

theorem cryptarithmetic_solution : 
  ∃! (K I S : Nat), 
    K < 10 ∧ I < 10 ∧ S < 10 ∧
    K ≠ I ∧ K ≠ S ∧ I ≠ S ∧
    100 * K + 10 * I + S + 100 * K + 10 * S + I = 100 * I + 10 * S + K ∧
    K = 4 ∧ I = 9 ∧ S = 5 := by
  sorry

end cryptarithmetic_solution_l3573_357373


namespace cubic_identity_l3573_357338

theorem cubic_identity (x : ℝ) (h : x^3 + 1/x^3 = 116) : x + 1/x = 4 := by
  sorry

end cubic_identity_l3573_357338


namespace ant_meeting_point_l3573_357387

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point on the perimeter of the triangle -/
structure PerimeterPoint where
  distanceFromP : ℝ

/-- The theorem statement -/
theorem ant_meeting_point (t : Triangle) (s : PerimeterPoint) : 
  t.a = 7 ∧ t.b = 8 ∧ t.c = 9 →
  s.distanceFromP = (t.a + t.b + t.c) / 2 →
  s.distanceFromP - t.a = 5 := by
  sorry

end ant_meeting_point_l3573_357387


namespace perfect_square_polynomial_l3573_357397

theorem perfect_square_polynomial (m : ℤ) : 
  1 + 2*m + 3*m^2 + 4*m^3 + 5*m^4 + 4*m^5 + 3*m^6 + 2*m^7 + m^8 = (1 + m + m^2 + m^3 + m^4)^2 := by
  sorry

end perfect_square_polynomial_l3573_357397


namespace complex_product_magnitude_l3573_357313

theorem complex_product_magnitude (a b : ℂ) (t : ℝ) :
  (Complex.abs a = 3) →
  (Complex.abs b = Real.sqrt 10) →
  (a * b = t - 3 * Complex.I) →
  (t > 0) →
  t = 9 := by
sorry

end complex_product_magnitude_l3573_357313


namespace line_equation_theorem_l3573_357389

-- Define the line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def has_slope (l : Line) (m : ℝ) : Prop :=
  l.a ≠ 0 ∧ -l.b / l.a = m

def triangle_area (l : Line) (area : ℝ) : Prop :=
  l.c ≠ 0 ∧ abs (l.c / l.a) * abs (l.c / l.b) / 2 = area

def passes_through (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

def equal_absolute_intercepts (l : Line) : Prop :=
  abs (l.c / l.a) = abs (l.c / l.b)

-- Define the theorem
theorem line_equation_theorem (l : Line) :
  has_slope l (3/4) ∧
  triangle_area l 6 ∧
  passes_through l 4 (-3) ∧
  equal_absolute_intercepts l →
  (l.a = 1 ∧ l.b = 1 ∧ l.c = -1) ∨
  (l.a = 1 ∧ l.b = -1 ∧ l.c = 7) ∨
  (l.a = 3 ∧ l.b = 4 ∧ l.c = 0) :=
sorry

end line_equation_theorem_l3573_357389


namespace total_shark_teeth_l3573_357370

def tiger_shark_teeth : ℕ := 180

def hammerhead_shark_teeth : ℕ := tiger_shark_teeth / 6

def great_white_shark_teeth : ℕ := 2 * (tiger_shark_teeth + hammerhead_shark_teeth)

def mako_shark_teeth : ℕ := (5 * hammerhead_shark_teeth) / 3

theorem total_shark_teeth : 
  tiger_shark_teeth + hammerhead_shark_teeth + great_white_shark_teeth + mako_shark_teeth = 680 := by
  sorry

end total_shark_teeth_l3573_357370


namespace smallest_integer_satisfying_inequality_l3573_357315

theorem smallest_integer_satisfying_inequality : 
  (∃ (x : ℤ), x / 4 + 3 / 7 > 2 / 3 ∧ ∀ (y : ℤ), y < x → y / 4 + 3 / 7 ≤ 2 / 3) ∧
  (∀ (x : ℤ), x / 4 + 3 / 7 > 2 / 3 → x ≥ 1) :=
by sorry

end smallest_integer_satisfying_inequality_l3573_357315


namespace distance_between_centers_l3573_357364

/-- Right triangle ABC with given side lengths -/
structure RightTriangle where
  AB : ℝ
  BC : ℝ
  AC : ℝ
  right_angle : AB^2 = BC^2 + AC^2

/-- Circle tangent to a side of the triangle and passing through the opposite vertex -/
structure TangentCircle (t : RightTriangle) where
  center : ℝ × ℝ
  tangent_point : ℝ × ℝ
  passing_point : ℝ × ℝ

/-- Configuration of the problem -/
structure TriangleCirclesConfig where
  triangle : RightTriangle
  circle_Q : TangentCircle triangle
  circle_R : TangentCircle triangle
  h_Q_tangent_BC : circle_Q.tangent_point.1 = 0 ∧ circle_Q.tangent_point.2 = 0
  h_Q_passes_A : circle_Q.passing_point.1 = 0 ∧ circle_Q.passing_point.2 = triangle.AC
  h_R_tangent_AC : circle_R.tangent_point.1 = 0 ∧ circle_R.tangent_point.2 = triangle.AC
  h_R_passes_B : circle_R.passing_point.1 = triangle.BC ∧ circle_R.passing_point.2 = 0

/-- The main theorem -/
theorem distance_between_centers (config : TriangleCirclesConfig)
  (h_triangle : config.triangle.AB = 13 ∧ config.triangle.BC = 5 ∧ config.triangle.AC = 12) :
  Real.sqrt ((config.circle_Q.center.1 - config.circle_R.center.1)^2 +
             (config.circle_Q.center.2 - config.circle_R.center.2)^2) = 33.8 := by
  sorry

end distance_between_centers_l3573_357364


namespace floor_plus_self_eq_18_3_l3573_357340

theorem floor_plus_self_eq_18_3 :
  ∃! s : ℝ, ⌊s⌋ + s = 18.3 ∧ s = 9.3 := by sorry

end floor_plus_self_eq_18_3_l3573_357340


namespace classroom_ratio_l3573_357361

theorem classroom_ratio : 
  ∀ (boys girls : ℕ),
  boys = girls →
  boys + girls = 32 →
  (boys : ℚ) / (girls - 8 : ℚ) = 2 := by
sorry

end classroom_ratio_l3573_357361


namespace equal_roots_quadratic_l3573_357367

/-- 
If the quadratic equation 2x^2 - x + c = 0 has two equal real roots, 
then c = 1/8.
-/
theorem equal_roots_quadratic (c : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - x + c = 0 ∧ 
   ∀ y : ℝ, 2 * y^2 - y + c = 0 → y = x) → 
  c = 1/8 := by
sorry

end equal_roots_quadratic_l3573_357367


namespace prime_factor_count_l3573_357377

/-- Given an expression 4^11 * x^5 * 11^2 with a total of 29 prime factors, x must be a prime number -/
theorem prime_factor_count (x : ℕ) : 
  (∀ (p : ℕ), Prime p → (Nat.factorization (4^11 * x^5 * 11^2)).sum (λ _ e => e) = 29) → 
  Prime x := by
sorry

end prime_factor_count_l3573_357377


namespace marthas_children_l3573_357341

/-- Given that Martha needs to buy a total number of cakes and each child should receive a specific number of cakes, calculate the number of children Martha has. -/
theorem marthas_children (total_cakes : ℕ) (cakes_per_child : ℚ) : 
  total_cakes = 54 → cakes_per_child = 18 → (total_cakes : ℚ) / cakes_per_child = 3 := by
  sorry

end marthas_children_l3573_357341


namespace monotonicity_and_range_l3573_357358

noncomputable def f (a b x : ℝ) : ℝ := 2 * a * x + b * x - 1 - 2 * Real.log x

theorem monotonicity_and_range :
  (∀ a ≤ 0, ∀ x > 0, (deriv (f a 0)) x < 0) ∧
  (∀ a > 0, ∀ x ∈ Set.Ioo 0 (1/a), (deriv (f a 0)) x < 0) ∧
  (∀ a > 0, ∀ x ∈ Set.Ioi (1/a), (deriv (f a 0)) x > 0) ∧
  (∀ x > 0, f 1 b x ≥ 2 * b * x - 3 → b ≤ 2 - 2 / Real.exp 2) :=
by sorry

end monotonicity_and_range_l3573_357358


namespace tom_found_15_seashells_l3573_357328

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := 43

/-- The difference between Fred's and Tom's seashell counts -/
def difference : ℕ := 28

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := fred_seashells - difference

theorem tom_found_15_seashells : tom_seashells = 15 := by
  sorry

end tom_found_15_seashells_l3573_357328


namespace total_earnings_l3573_357363

/-- The total earnings of Salvadore and Santo, given Salvadore's earnings and that Santo earned half of Salvadore's earnings -/
theorem total_earnings (salvadore_earnings : ℕ) (h : salvadore_earnings = 1956) :
  salvadore_earnings + (salvadore_earnings / 2) = 2934 := by
  sorry

end total_earnings_l3573_357363


namespace exists_solution_l3573_357380

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + Real.exp (x - a)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem exists_solution (a : ℝ) : 
  (∃ x₀ : ℝ, f a x₀ - g a x₀ = 3) → a = -Real.log 2 - 1 := by sorry

end exists_solution_l3573_357380


namespace derivative_of_odd_function_l3573_357353

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
variable (hf : Differentiable ℝ f)
variable (hodd : ∀ x, f (-x) = -f x)

-- Define x₀ and k
variable (x₀ : ℝ)
variable (k : ℝ)
variable (hk : k ≠ 0)

-- State the hypothesis about f'(-x₀)
variable (hderiv : deriv f (-x₀) = k)

-- State the theorem
theorem derivative_of_odd_function :
  deriv f x₀ = k := by sorry

end derivative_of_odd_function_l3573_357353


namespace presidency_meeting_arrangements_l3573_357310

/-- Represents the number of schools --/
def num_schools : ℕ := 3

/-- Represents the number of members per school --/
def members_per_school : ℕ := 5

/-- Calculates the number of ways to choose r items from n items --/
def choose (n : ℕ) (r : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))

/-- Represents the number of ways to choose representatives from the host school --/
def host_school_choices : ℕ := choose members_per_school 2

/-- Represents the number of ways to choose representatives from non-host schools --/
def non_host_school_choices : ℕ := (choose members_per_school 1) ^ 2

/-- Represents the total number of ways to arrange the presidency meeting --/
def total_arrangements : ℕ := num_schools * host_school_choices * non_host_school_choices

theorem presidency_meeting_arrangements :
  total_arrangements = 750 :=
sorry

end presidency_meeting_arrangements_l3573_357310


namespace expression_value_l3573_357354

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 2) :
  3 * x^2 - 4 * y + 2 = 21 := by
  sorry

end expression_value_l3573_357354


namespace markup_calculation_l3573_357309

/-- Given a purchase price, overhead percentage, and desired net profit, 
    calculate the required markup. -/
def calculate_markup (purchase_price : ℝ) (overhead_percentage : ℝ) (net_profit : ℝ) : ℝ :=
  let overhead_cost := overhead_percentage * purchase_price
  let total_cost := purchase_price + overhead_cost
  let selling_price := total_cost + net_profit
  selling_price - purchase_price

/-- Theorem stating that the markup for the given conditions is $14.40 -/
theorem markup_calculation : 
  calculate_markup 48 0.05 12 = 14.40 := by
  sorry

end markup_calculation_l3573_357309


namespace whole_milk_fat_percentage_l3573_357366

/-- The percentage of fat in low-fat milk -/
def low_fat_milk_percentage : ℝ := 3

/-- The percentage difference between low-fat and semi-skimmed milk -/
def low_fat_semi_skimmed_difference : ℝ := 25

/-- The percentage difference between semi-skimmed and whole milk -/
def semi_skimmed_whole_difference : ℝ := 20

/-- The percentage of fat in whole milk -/
def whole_milk_percentage : ℝ := 5

theorem whole_milk_fat_percentage :
  (low_fat_milk_percentage / (1 - low_fat_semi_skimmed_difference / 100)) / (1 - semi_skimmed_whole_difference / 100) = whole_milk_percentage := by
  sorry

end whole_milk_fat_percentage_l3573_357366


namespace quadratic_roots_theorem_l3573_357395

-- Define the quadratic equation
def quadratic (m : ℤ) (x : ℤ) : Prop :=
  x^2 - 2*(2*m-3)*x + 4*m^2 - 14*m + 8 = 0

-- Define the theorem
theorem quadratic_roots_theorem :
  ∀ m : ℤ, 4 < m → m < 40 →
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ quadratic m x₁ ∧ quadratic m x₂) →
  ((m = 12 ∧ ∃ x₁ x₂ : ℤ, x₁ = 26 ∧ x₂ = 16 ∧ quadratic m x₁ ∧ quadratic m x₂) ∨
   (m = 24 ∧ ∃ x₁ x₂ : ℤ, x₁ = 52 ∧ x₂ = 38 ∧ quadratic m x₁ ∧ quadratic m x₂)) :=
by sorry

end quadratic_roots_theorem_l3573_357395


namespace caitlin_bracelets_l3573_357399

/-- The number of bracelets Caitlin can make given the conditions -/
def num_bracelets : ℕ :=
  let total_beads : ℕ := 528
  let large_beads_per_bracelet : ℕ := 12
  let small_beads_per_bracelet : ℕ := 2 * large_beads_per_bracelet
  let large_beads : ℕ := total_beads / 2
  let small_beads : ℕ := total_beads / 2
  min (large_beads / large_beads_per_bracelet) (small_beads / small_beads_per_bracelet)

/-- Theorem stating that Caitlin can make 22 bracelets -/
theorem caitlin_bracelets : num_bracelets = 22 := by
  sorry

end caitlin_bracelets_l3573_357399


namespace gas_price_increase_l3573_357321

theorem gas_price_increase (x : ℝ) : 
  (1 + x / 100) * 1.1 * (1 - 27.27272727272727 / 100) = 1 → x = 25 := by
  sorry

end gas_price_increase_l3573_357321


namespace total_weight_on_scale_l3573_357372

theorem total_weight_on_scale (alexa_weight katerina_weight : ℕ) 
  (h1 : alexa_weight = 46)
  (h2 : katerina_weight = 49) :
  alexa_weight + katerina_weight = 95 := by
  sorry

end total_weight_on_scale_l3573_357372


namespace absolute_value_inequality_range_l3573_357335

theorem absolute_value_inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 3| ≥ a^2 + a) ↔ -2 ≤ a ∧ a ≤ 1 :=
sorry

end absolute_value_inequality_range_l3573_357335


namespace meeting_participants_l3573_357347

theorem meeting_participants :
  ∀ (F M : ℕ),
  F > 0 →
  M > 0 →
  F / 2 = 125 →
  F / 2 + M / 4 = (F + M) / 3 →
  F + M = 1750 :=
by
  sorry

end meeting_participants_l3573_357347


namespace value_of_x_l3573_357346

theorem value_of_x :
  ∀ (x y z w u : ℤ),
    x = y + 3 →
    y = z + 15 →
    z = w + 25 →
    w = u + 10 →
    u = 90 →
    x = 143 := by
  sorry

end value_of_x_l3573_357346


namespace students_both_correct_l3573_357385

theorem students_both_correct (total : ℕ) (physics_correct : ℕ) (chemistry_correct : ℕ) (both_incorrect : ℕ)
  (h1 : total = 50)
  (h2 : physics_correct = 40)
  (h3 : chemistry_correct = 31)
  (h4 : both_incorrect = 4) :
  total - both_incorrect - (physics_correct + chemistry_correct - total + both_incorrect) = 25 := by
  sorry

end students_both_correct_l3573_357385


namespace sqrt_product_plus_one_l3573_357316

theorem sqrt_product_plus_one : 
  Real.sqrt ((26 : ℝ) * 25 * 24 * 23 + 1) = 599 := by sorry

end sqrt_product_plus_one_l3573_357316


namespace illustration_project_time_l3573_357382

/-- Calculates the total time spent on an illustration project with three phases -/
def total_illustration_time (
  landscape_count : ℕ)
  (landscape_draw_time : ℝ)
  (landscape_color_reduction : ℝ)
  (landscape_enhance_time : ℝ)
  (portrait_count : ℕ)
  (portrait_draw_time : ℝ)
  (portrait_color_reduction : ℝ)
  (portrait_enhance_time : ℝ)
  (abstract_count : ℕ)
  (abstract_draw_time : ℝ)
  (abstract_color_reduction : ℝ)
  (abstract_enhance_time : ℝ) : ℝ :=
  let landscape_time := landscape_count * (landscape_draw_time + landscape_draw_time * (1 - landscape_color_reduction) + landscape_enhance_time)
  let portrait_time := portrait_count * (portrait_draw_time + portrait_draw_time * (1 - portrait_color_reduction) + portrait_enhance_time)
  let abstract_time := abstract_count * (abstract_draw_time + abstract_draw_time * (1 - abstract_color_reduction) + abstract_enhance_time)
  landscape_time + portrait_time + abstract_time

theorem illustration_project_time :
  total_illustration_time 10 2 0.3 0.75 15 3 0.25 1 20 1.5 0.4 0.5 = 193.25 := by
  sorry

end illustration_project_time_l3573_357382


namespace croissant_making_time_l3573_357398

/-- Proves that the total time for making croissants is 6 hours -/
theorem croissant_making_time : 
  let fold_time : ℕ := 4 * 5
  let rest_time : ℕ := 4 * 75
  let mix_time : ℕ := 10
  let bake_time : ℕ := 30
  let minutes_per_hour : ℕ := 60
  (fold_time + rest_time + mix_time + bake_time) / minutes_per_hour = 6 := by
  sorry

end croissant_making_time_l3573_357398


namespace ages_sum_l3573_357317

/-- Given the ages of Al, Bob, and Carl satisfying certain conditions, prove their sum is 80 -/
theorem ages_sum (a b c : ℕ) : 
  a = b + c + 20 → 
  a^2 = (b + c)^2 + 2000 → 
  a + b + c = 80 := by
sorry

end ages_sum_l3573_357317


namespace angle_properties_l3573_357378

def isObtuseAngle (α : ℝ) : Prop := 90 < α ∧ α < 180

def isSecondQuadrantAngle (β : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 + 90 < β ∧ β < k * 360 + 180

def isFirstQuadrantAngle (γ : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 < γ ∧ γ < k * 360 + 90

theorem angle_properties :
  (∀ α : ℝ, isObtuseAngle α → isSecondQuadrantAngle α) ∧
  (∃ β γ : ℝ, isSecondQuadrantAngle β ∧ isFirstQuadrantAngle γ ∧ β < γ) ∧
  (∃ δ : ℝ, 90 < δ ∧ ¬ isObtuseAngle δ) ∧
  ¬ isSecondQuadrantAngle (-165) :=
by sorry

end angle_properties_l3573_357378


namespace fish_value_in_rice_l3573_357320

/-- Represents the trade value of items in terms of bags of rice -/
structure TradeValue where
  fish : ℚ
  bread : ℚ

/-- Defines the trade rates in the distant realm -/
def trade_rates : TradeValue where
  fish := 5⁻¹ * 3 * 6  -- 5 fish = 3 bread, 1 bread = 6 rice
  bread := 6           -- 1 bread = 6 rice

/-- Theorem stating that one fish is equivalent to 3 3/5 bags of rice -/
theorem fish_value_in_rice : trade_rates.fish = 18/5 := by
  sorry

#eval trade_rates.fish

end fish_value_in_rice_l3573_357320


namespace minimum_married_men_l3573_357337

theorem minimum_married_men (total_men : ℕ) (tv_men : ℕ) (radio_men : ℕ) (ac_men : ℕ) (married_with_all : ℕ)
  (h_total : total_men = 100)
  (h_tv : tv_men = 75)
  (h_radio : radio_men = 85)
  (h_ac : ac_men = 70)
  (h_married_all : married_with_all = 11)
  (h_tv_le : tv_men ≤ total_men)
  (h_radio_le : radio_men ≤ total_men)
  (h_ac_le : ac_men ≤ total_men)
  (h_married_all_le : married_with_all ≤ tv_men ∧ married_with_all ≤ radio_men ∧ married_with_all ≤ ac_men) :
  ∃ (married_men : ℕ), married_men ≥ married_with_all ∧ married_men ≤ total_men := by
  sorry

end minimum_married_men_l3573_357337


namespace sports_conference_games_l3573_357333

theorem sports_conference_games (n : ℕ) (d : ℕ) (intra : ℕ) (inter : ℕ) 
  (h1 : n = 16)
  (h2 : d = 2)
  (h3 : n = d * 8)
  (h4 : intra = 3)
  (h5 : inter = 2) :
  d * (Nat.choose 8 2 * intra) + (n / 2) * (n / 2) * inter = 296 := by
  sorry

end sports_conference_games_l3573_357333


namespace beetles_eaten_in_forest_l3573_357356

/-- The number of beetles eaten in a forest each day -/
def beetles_eaten_per_day (jaguars : ℕ) (snakes_per_jaguar : ℕ) (birds_per_snake : ℕ) (beetles_per_bird : ℕ) : ℕ :=
  jaguars * snakes_per_jaguar * birds_per_snake * beetles_per_bird

/-- Theorem stating the number of beetles eaten in a specific forest scenario -/
theorem beetles_eaten_in_forest :
  beetles_eaten_per_day 6 5 3 12 = 1080 := by
  sorry

#eval beetles_eaten_per_day 6 5 3 12

end beetles_eaten_in_forest_l3573_357356


namespace least_prime_angle_in_square_triangle_l3573_357369

theorem least_prime_angle_in_square_triangle (a b : ℕ) : 
  (a > b) →
  (Nat.Prime a) →
  (Nat.Prime b) →
  (a + b = 90) →
  (∀ p, Nat.Prime p → p < b → ¬(∃ q, Nat.Prime q ∧ p + q = 90)) →
  b = 7 := by
sorry

end least_prime_angle_in_square_triangle_l3573_357369


namespace inequality_proof_equality_condition_l3573_357311

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + 3*c)) + (b / (8*c + 4*a)) + (9*c / (3*a + 2*b)) ≥ 47/48 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + 3*c)) + (b / (8*c + 4*a)) + (9*c / (3*a + 2*b)) = 47/48 ↔ 
  ∃ (k : ℝ), k > 0 ∧ a = 10*k ∧ b = 21*k ∧ c = k :=
by sorry

end inequality_proof_equality_condition_l3573_357311


namespace comprehensive_investigation_is_census_l3573_357303

/-- A comprehensive investigation conducted on the subject of examination for a specific purpose -/
def comprehensive_investigation : Type := Unit

/-- Census as a type -/
def census : Type := Unit

/-- Theorem stating that a comprehensive investigation is equivalent to a census -/
theorem comprehensive_investigation_is_census : 
  comprehensive_investigation ≃ census := by sorry

end comprehensive_investigation_is_census_l3573_357303


namespace chain_merge_time_theorem_l3573_357343

/-- Represents a chain with a certain number of links -/
structure Chain where
  links : ℕ

/-- Represents the time required for chain operations -/
structure ChainOperationTime where
  openLinkTime : ℕ
  closeLinkTime : ℕ

/-- Calculates the minimum time required to merge chains -/
def minTimeMergeChains (chains : List Chain) (opTime : ChainOperationTime) : ℕ :=
  sorry

/-- Theorem statement for the chain merging problem -/
theorem chain_merge_time_theorem (chains : List Chain) (opTime : ChainOperationTime) :
  chains.length = 6 ∧ 
  chains.all (λ c => c.links = 4) ∧
  opTime.openLinkTime = 1 ∧
  opTime.closeLinkTime = 3 →
  minTimeMergeChains chains opTime = 20 :=
sorry

end chain_merge_time_theorem_l3573_357343


namespace condition_necessary_not_sufficient_l3573_357388

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x^2 - x - 2 = 0 → -1 ≤ x ∧ x ≤ 2) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ x^2 - x - 2 ≠ 0) :=
by sorry

end condition_necessary_not_sufficient_l3573_357388


namespace shadow_problem_l3573_357355

theorem shadow_problem (cube_edge : ℝ) (shadow_area : ℝ) (y : ℝ) : 
  cube_edge = 2 →
  shadow_area = 200 →
  y > 0 →
  y = (Real.sqrt (shadow_area + cube_edge ^ 2)) →
  ⌊1000 * y⌋ = 14280 := by
sorry

end shadow_problem_l3573_357355


namespace largest_six_digit_number_l3573_357322

def digit_product (n : ℕ) : ℕ :=
  if n = 0 then 1 else (n % 10) * digit_product (n / 10)

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem largest_six_digit_number : ∀ n : ℕ, 
  100000 ≤ n ∧ n ≤ 999999 ∧ digit_product n = factorial 8 → n ≤ 987744 :=
by
  sorry

end largest_six_digit_number_l3573_357322


namespace imaginary_part_of_complex_l3573_357375

theorem imaginary_part_of_complex (z : ℂ) (h : z = -4 * Complex.I + 3) : 
  z.im = -4 := by sorry

end imaginary_part_of_complex_l3573_357375


namespace cubic_equation_solution_l3573_357305

theorem cubic_equation_solution :
  ∃! x : ℝ, x^3 + 12*x = 6*x^2 + 35 :=
by
  -- The unique solution is x = 5
  use 5
  constructor
  · -- Prove that x = 5 satisfies the equation
    simp
    -- Additional steps to prove 5^3 + 12*5 = 6*5^2 + 35
    sorry
  · -- Prove that any solution must equal 5
    intro y hy
    -- Steps to show that if y satisfies the equation, then y = 5
    sorry


end cubic_equation_solution_l3573_357305


namespace new_selling_price_l3573_357396

theorem new_selling_price (old_price : ℝ) (old_profit_rate new_profit_rate : ℝ) : 
  old_price = 88 →
  old_profit_rate = 0.1 →
  new_profit_rate = 0.15 →
  let cost := old_price / (1 + old_profit_rate)
  let new_price := cost * (1 + new_profit_rate)
  new_price = 92 := by
sorry

end new_selling_price_l3573_357396


namespace intersection_points_10_5_l3573_357394

/-- The number of intersection points formed by line segments connecting points on x and y axes -/
def intersection_points (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- Theorem stating that 10 points on x-axis and 5 points on y-axis result in 450 intersection points -/
theorem intersection_points_10_5 :
  intersection_points 10 5 = 450 := by sorry

end intersection_points_10_5_l3573_357394


namespace cost_reduction_over_two_years_l3573_357334

theorem cost_reduction_over_two_years (total_reduction : ℝ) (annual_reduction : ℝ) :
  total_reduction = 0.19 →
  (1 - annual_reduction) * (1 - annual_reduction) = 1 - total_reduction →
  annual_reduction = 0.1 := by
sorry

end cost_reduction_over_two_years_l3573_357334


namespace set_operation_result_l3573_357386

def A : Set Nat := {1, 2, 6}
def B : Set Nat := {2, 4}
def C : Set Nat := {1, 2, 3, 4}

theorem set_operation_result : (A ∪ B) ∩ C = {1, 2, 4} := by sorry

end set_operation_result_l3573_357386


namespace martins_berry_consumption_l3573_357371

/-- Given the cost of berries and Martin's spending habits, calculate his daily berry consumption --/
theorem martins_berry_consumption
  (package_cost : ℚ)
  (total_spent : ℚ)
  (num_days : ℕ)
  (h1 : package_cost = 2)
  (h2 : total_spent = 30)
  (h3 : num_days = 30)
  : (total_spent / package_cost) / num_days = 1/2 := by
  sorry

end martins_berry_consumption_l3573_357371


namespace geometric_sequence_ratio_l3573_357327

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def four_consecutive_terms (b : ℕ → ℝ) (S : Set ℝ) : Prop :=
  ∃ k, (b k ∈ S) ∧ (b (k + 1) ∈ S) ∧ (b (k + 2) ∈ S) ∧ (b (k + 3) ∈ S)

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (b : ℕ → ℝ) :
  is_geometric_sequence a q →
  (∀ n, b n = a n + 1) →
  |q| > 1 →
  four_consecutive_terms b {-53, -23, 19, 37, 82} →
  q = -3/2 := by
  sorry

end geometric_sequence_ratio_l3573_357327


namespace triangle_is_equilateral_l3573_357314

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  (t.a^3 + t.b^3 - t.c^3) / (t.a + t.b - t.c) = t.c^2

def condition2 (t : Triangle) : Prop :=
  Real.sin t.α * Real.sin t.β = 3/4

-- Theorem statement
theorem triangle_is_equilateral (t : Triangle) 
  (h1 : condition1 t) (h2 : condition2 t) :
  t.a = t.b ∧ t.b = t.c ∧ t.α = π/3 ∧ t.β = π/3 ∧ t.γ = π/3 :=
sorry

end triangle_is_equilateral_l3573_357314


namespace inequality_proof_l3573_357323

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 * a) / (a^2 + b * c) + (2 * b) / (b^2 + c * a) + (2 * c) / (c^2 + a * b) ≤
  a / (b * c) + b / (c * a) + c / (a * b) := by
  sorry

end inequality_proof_l3573_357323


namespace regular_polyhedra_symmetry_axes_l3573_357349

-- Define the types of regular polyhedra
inductive RegularPolyhedron
  | Tetrahedron
  | Hexahedron
  | Octahedron
  | Dodecahedron
  | Icosahedron

-- Define a structure for symmetry axis information
structure SymmetryAxis where
  order : ℕ
  count : ℕ

-- Define a function that returns the symmetry axes for a given polyhedron
def symmetryAxes (p : RegularPolyhedron) : List SymmetryAxis :=
  match p with
  | RegularPolyhedron.Tetrahedron => [
      { order := 3, count := 4 },
      { order := 2, count := 3 }
    ]
  | RegularPolyhedron.Hexahedron => [
      { order := 4, count := 3 },
      { order := 3, count := 4 },
      { order := 2, count := 6 }
    ]
  | RegularPolyhedron.Octahedron => [
      { order := 4, count := 3 },
      { order := 3, count := 4 },
      { order := 2, count := 6 }
    ]
  | RegularPolyhedron.Dodecahedron => [
      { order := 5, count := 6 },
      { order := 3, count := 10 },
      { order := 2, count := 15 }
    ]
  | RegularPolyhedron.Icosahedron => [
      { order := 5, count := 6 },
      { order := 3, count := 10 },
      { order := 2, count := 15 }
    ]

-- Theorem stating that the symmetry axes for each polyhedron are correct
theorem regular_polyhedra_symmetry_axes :
  ∀ p : RegularPolyhedron, 
    (symmetryAxes p).length > 0 ∧
    (∀ axis ∈ symmetryAxes p, axis.order ≥ 2 ∧ axis.count > 0) :=
by sorry

end regular_polyhedra_symmetry_axes_l3573_357349


namespace greatest_power_under_500_l3573_357324

theorem greatest_power_under_500 :
  ∃ (a b : ℕ), 
    a > 0 ∧ 
    b > 1 ∧ 
    a^b < 500 ∧ 
    (∀ (c d : ℕ), c > 0 → d > 1 → c^d < 500 → c^d ≤ a^b) ∧ 
    a + b = 24 :=
by sorry

end greatest_power_under_500_l3573_357324


namespace sum_of_roots_quadratic_sum_of_roots_specific_equation_l3573_357391

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ s = -b / a) :=
sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 + 2000 * x - 2001
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ s = -2000) :=
sorry

end sum_of_roots_quadratic_sum_of_roots_specific_equation_l3573_357391


namespace p_range_q_range_p_or_q_range_l3573_357344

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m - 3 > 0
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*m*x + m + 2 < 0

-- Theorem for the range of m when p is true
theorem p_range (m : ℝ) : p m ↔ m > 3/2 := by sorry

-- Theorem for the range of m when q is true
theorem q_range (m : ℝ) : q m ↔ m < -1 ∨ m > 2 := by sorry

-- Theorem for the range of m when at least one of p or q is true
theorem p_or_q_range (m : ℝ) : p m ∨ q m ↔ m < -1 ∨ m > 3/2 := by sorry

end p_range_q_range_p_or_q_range_l3573_357344


namespace taxi_fare_equality_l3573_357301

/-- Taxi fare calculation problem -/
theorem taxi_fare_equality (mike_base_fare annie_base_fare toll_fee : ℚ)
  (per_mile_rate : ℚ) (annie_miles : ℕ) :
  mike_base_fare = 2.5 ∧
  annie_base_fare = 2.5 ∧
  toll_fee = 5 ∧
  per_mile_rate = 0.25 ∧
  annie_miles = 16 →
  ∃ (mike_miles : ℕ),
    mike_base_fare + per_mile_rate * mike_miles =
    annie_base_fare + toll_fee + per_mile_rate * annie_miles ∧
    mike_miles = 36 := by
  sorry

end taxi_fare_equality_l3573_357301


namespace some_number_value_l3573_357376

theorem some_number_value (a x : ℚ) : 
  a = 105 → 
  a^3 = x * 25 * 45 * 49 → 
  x = 7/3 := by
sorry

end some_number_value_l3573_357376


namespace remainder_theorem_l3573_357319

theorem remainder_theorem (d : ℚ) : 
  (∃! d, ∀ x, (3 * x^3 + d * x^2 - 6 * x + 25) % (3 * x + 5) = 3) → d = 2 := by
  sorry

end remainder_theorem_l3573_357319


namespace max_annual_profit_l3573_357381

/-- Additional investment function R -/
noncomputable def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 40 then 10 * x^2 + 300 * x
  else (901 * x^2 - 9450 * x + 10000) / x

/-- Annual profit function W -/
noncomputable def W (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 40 then -10 * x^2 + 600 * x - 260
  else -x - 10000 / x + 9190

/-- Theorem stating the maximum annual profit and corresponding production volume -/
theorem max_annual_profit :
  ∃ (x : ℝ), x = 100 ∧ W x = 8990 ∧ ∀ y, W y ≤ W x :=
sorry

end max_annual_profit_l3573_357381


namespace motorist_journey_l3573_357365

theorem motorist_journey (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : 
  total_time = 6 → speed1 = 60 → speed2 = 48 → 
  (total_time / 2 * speed1) + (total_time / 2 * speed2) = 324 := by
sorry

end motorist_journey_l3573_357365


namespace area_of_ω_l3573_357329

-- Define the circle ω
def ω : Set (ℝ × ℝ) := sorry

-- Define points A and B
def A : ℝ × ℝ := (4, 15)
def B : ℝ × ℝ := (12, 9)

-- State that A and B lie on ω
axiom A_on_ω : A ∈ ω
axiom B_on_ω : B ∈ ω

-- Define the tangent lines at A and B
def tangent_A : Set (ℝ × ℝ) := sorry
def tangent_B : Set (ℝ × ℝ) := sorry

-- State that the tangent lines intersect at a point on the x-axis
axiom tangents_intersect_x_axis : ∃ x : ℝ, (x, 0) ∈ tangent_A ∩ tangent_B

-- Define the area of a circle
def circle_area (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_of_ω : circle_area ω = 306 * Real.pi := sorry

end area_of_ω_l3573_357329


namespace luke_trivia_rounds_l3573_357304

/-- Given that Luke gained 46 points per round and scored 8142 points in total,
    prove that he played 177 rounds. -/
theorem luke_trivia_rounds (points_per_round : ℕ) (total_points : ℕ) 
    (h1 : points_per_round = 46) 
    (h2 : total_points = 8142) : 
  total_points / points_per_round = 177 := by
  sorry

end luke_trivia_rounds_l3573_357304


namespace no_positive_integral_solutions_l3573_357342

theorem no_positive_integral_solutions : 
  ¬ ∃ (x y : ℕ+), x.val^6 * y.val^6 - 13 * x.val^3 * y.val^3 + 36 = 0 :=
by sorry

end no_positive_integral_solutions_l3573_357342


namespace flu_infection_model_l3573_357392

/-- 
Given two rounds of flu infection where:
- In each round, on average, one person infects x people
- After two rounds, a total of 144 people had the flu

This theorem states that the equation (1+x)^2 = 144 correctly models 
the total number of people infected after two rounds.
-/
theorem flu_infection_model (x : ℝ) : 
  (∃ (infected_first_round infected_second_round : ℕ),
    infected_first_round = x ∧ 
    infected_second_round = x * infected_first_round ∧
    1 + infected_first_round + infected_second_round = 144) ↔ 
  (1 + x)^2 = 144 :=
sorry

end flu_infection_model_l3573_357392


namespace quadruple_equation_solutions_l3573_357351

theorem quadruple_equation_solutions :
  let equation (a b c d : ℕ) := 2*a + 2*b + 2*c + 2*d = d^2 - c^2 + b^2 - a^2
  ∀ (a b c d : ℕ), a < b → b < c → c < d →
  (
    (equation 2 4 5 7) ∧
    (∀ x : ℕ, equation (2*x) (2*x+2) (2*x+4) (2*x+6))
  ) := by sorry

end quadruple_equation_solutions_l3573_357351


namespace circle_intersection_condition_l3573_357331

-- Define the circles B and C
def circle_B (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + b = 0}

def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 6*p.1 + 8*p.2 + 16 = 0}

-- Define the condition for no common points
def no_common_points (b : ℝ) : Prop :=
  circle_B b ∩ circle_C = ∅

-- State the theorem
theorem circle_intersection_condition :
  ∀ b : ℝ, no_common_points b ↔ (-4 < b ∧ b < 0) ∨ b < -64 :=
sorry

end circle_intersection_condition_l3573_357331


namespace ceiling_product_sqrt_l3573_357306

theorem ceiling_product_sqrt : ⌈Real.sqrt 3⌉ * ⌈Real.sqrt 12⌉ * ⌈Real.sqrt 120⌉ = 88 := by
  sorry

end ceiling_product_sqrt_l3573_357306


namespace ducks_and_dogs_total_l3573_357339

theorem ducks_and_dogs_total (d g : ℕ) : 
  d = g + 2 →                   -- number of ducks is 2 more than dogs
  4 * g - 2 * d = 10 →          -- dogs have 10 more legs than ducks
  d + g = 16 := by              -- total number of ducks and dogs is 16
sorry

end ducks_and_dogs_total_l3573_357339


namespace quadratic_discriminant_l3573_357330

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 - 11x - 18 -/
def a : ℝ := 5
def b : ℝ := -11
def c : ℝ := -18

theorem quadratic_discriminant : discriminant a b c = 481 := by
  sorry

end quadratic_discriminant_l3573_357330


namespace cube_sum_relation_l3573_357360

theorem cube_sum_relation : 
  (2^3 + 4^3 + 6^3 + 8^3 + 10^3 + 12^3 + 14^3 + 16^3 + 18^3 = 16200) →
  (3^3 + 6^3 + 9^3 + 12^3 + 15^3 + 18^3 + 21^3 + 24^3 + 27^3 = 54675) :=
by
  sorry

end cube_sum_relation_l3573_357360


namespace abacus_problem_solution_l3573_357332

/-- Represents a three-digit number -/
def ThreeDigitNumber := { n : ℕ // n ≥ 100 ∧ n < 1000 }

/-- Check if a three-digit number has distinct digits -/
def has_distinct_digits (n : ThreeDigitNumber) : Prop :=
  let digits := [n.val / 100, (n.val / 10) % 10, n.val % 10]
  digits.Nodup

/-- The abacus problem solution -/
theorem abacus_problem_solution :
  ∃! (top bottom : ThreeDigitNumber),
    has_distinct_digits top ∧
    ∃ (k : ℕ), k > 1 ∧ top.val = k * bottom.val ∧
    top.val + bottom.val = 1110 ∧
    top.val = 925 := by
  sorry

end abacus_problem_solution_l3573_357332


namespace tumbler_payment_denomination_l3573_357325

/-- Proves that the denomination of bills used to pay for tumblers is $100 given the specified conditions -/
theorem tumbler_payment_denomination :
  ∀ (num_tumblers : ℕ) (cost_per_tumbler : ℕ) (num_bills : ℕ) (change : ℕ),
    num_tumblers = 10 →
    cost_per_tumbler = 45 →
    num_bills = 5 →
    change = 50 →
    (num_tumblers * cost_per_tumbler + change) / num_bills = 100 :=
by
  sorry

end tumbler_payment_denomination_l3573_357325


namespace unique_x_with_three_prime_divisors_including_13_l3573_357318

theorem unique_x_with_three_prime_divisors_including_13 :
  ∀ (x n : ℕ),
    x = 9^n - 1 →
    (∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r) →
    13 ∣ x →
    x = 728 := by
  sorry

end unique_x_with_three_prime_divisors_including_13_l3573_357318


namespace floor_of_2_7_l3573_357352

theorem floor_of_2_7 :
  ⌊(2.7 : ℝ)⌋ = 2 := by sorry

end floor_of_2_7_l3573_357352


namespace fair_haired_women_percentage_l3573_357357

/-- Given that 30% of employees are women with fair hair and 75% of employees have fair hair,
    prove that 40% of fair-haired employees are women. -/
theorem fair_haired_women_percentage
  (total_employees : ℝ)
  (women_fair_hair_percentage : ℝ)
  (fair_hair_percentage : ℝ)
  (h1 : women_fair_hair_percentage = 30 / 100)
  (h2 : fair_hair_percentage = 75 / 100) :
  (women_fair_hair_percentage * total_employees) / (fair_hair_percentage * total_employees) = 40 / 100 :=
by sorry

end fair_haired_women_percentage_l3573_357357


namespace unique_solution_quadratic_l3573_357345

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = m + 3 * x) ↔ m = 6 := by
  sorry

end unique_solution_quadratic_l3573_357345


namespace smallest_multiple_with_conditions_l3573_357350

theorem smallest_multiple_with_conditions : ∃! n : ℕ, 
  n > 0 ∧ 
  47 ∣ n ∧ 
  n % 97 = 7 ∧ 
  n % 31 = 28 ∧ 
  ∀ m : ℕ, m > 0 → 47 ∣ m → m % 97 = 7 → m % 31 = 28 → n ≤ m :=
by
  use 79618
  sorry

end smallest_multiple_with_conditions_l3573_357350


namespace gcd_of_three_numbers_l3573_357348

theorem gcd_of_three_numbers : Nat.gcd 4560 (Nat.gcd 6080 16560) = 80 := by
  sorry

end gcd_of_three_numbers_l3573_357348


namespace annual_salary_calculation_l3573_357390

theorem annual_salary_calculation (hourly_wage : ℝ) (hours_per_day : ℕ) (days_per_month : ℕ) :
  hourly_wage = 8.50 →
  hours_per_day = 8 →
  days_per_month = 20 →
  hourly_wage * hours_per_day * days_per_month * 12 = 16320 :=
by
  sorry

end annual_salary_calculation_l3573_357390


namespace rectangle_triangle_area_equality_l3573_357359

theorem rectangle_triangle_area_equality (l w h : ℝ) (l_pos : l > 0) (w_pos : w > 0) (h_pos : h > 0) :
  l * w = (1 / 2) * l * h → h = 2 * w := by
  sorry

end rectangle_triangle_area_equality_l3573_357359


namespace planar_graph_edge_count_l3573_357368

/-- A planar graph -/
structure PlanarGraph where
  V : Type* -- Vertex set
  E : Type* -- Edge set
  n : ℕ     -- Number of vertices
  m : ℕ     -- Number of edges
  is_planar : Bool
  vertex_count : n ≥ 3

/-- A planar triangulation -/
structure PlanarTriangulation extends PlanarGraph where
  is_triangulation : Bool

/-- Theorem about the number of edges in planar graphs and planar triangulations -/
theorem planar_graph_edge_count (G : PlanarGraph) :
  G.m ≤ 3 * G.n - 6 ∧
  (∀ (T : PlanarTriangulation), T.toPlanarGraph = G → T.m = 3 * T.n - 6) :=
sorry

end planar_graph_edge_count_l3573_357368


namespace truck_distance_on_rough_terrain_truck_travel_distance_l3573_357302

/-- Calculates the distance a truck can travel on rough terrain given its performance on a smooth highway and the efficiency decrease on rough terrain. -/
theorem truck_distance_on_rough_terrain 
  (highway_distance : ℝ) 
  (highway_gas : ℝ) 
  (rough_terrain_efficiency_decrease : ℝ) 
  (rough_terrain_gas : ℝ) : ℝ :=
  let highway_efficiency := highway_distance / highway_gas
  let rough_terrain_efficiency := highway_efficiency * (1 - rough_terrain_efficiency_decrease)
  rough_terrain_efficiency * rough_terrain_gas

/-- Proves that a truck traveling 300 miles on 10 gallons of gas on a smooth highway can travel 405 miles on 15 gallons of gas on rough terrain with a 10% efficiency decrease. -/
theorem truck_travel_distance : 
  truck_distance_on_rough_terrain 300 10 0.1 15 = 405 := by
  sorry

end truck_distance_on_rough_terrain_truck_travel_distance_l3573_357302


namespace max_obtuse_dihedral_angles_l3573_357307

/-- A tetrahedron is a polyhedron with four faces. -/
structure Tetrahedron where
  -- We don't need to define the internal structure for this problem

/-- A dihedral angle is the angle between two intersecting planes. -/
structure DihedralAngle where
  -- We don't need to define the internal structure for this problem

/-- An obtuse angle is an angle greater than 90 degrees but less than 180 degrees. -/
def isObtuse (angle : DihedralAngle) : Prop :=
  sorry  -- Definition of obtuse angle

/-- A tetrahedron has exactly 6 dihedral angles. -/
axiom tetrahedron_has_six_dihedral_angles (t : Tetrahedron) :
  ∃ (angles : Finset DihedralAngle), angles.card = 6

/-- The maximum number of obtuse dihedral angles in a tetrahedron is 3. -/
theorem max_obtuse_dihedral_angles (t : Tetrahedron) :
  ∃ (angles : Finset DihedralAngle),
    (∀ a ∈ angles, isObtuse a) ∧
    angles.card = 3 ∧
    ∀ (other_angles : Finset DihedralAngle),
      (∀ a ∈ other_angles, isObtuse a) →
      other_angles.card ≤ 3 :=
sorry

end max_obtuse_dihedral_angles_l3573_357307


namespace quadratic_roots_problem_l3573_357383

theorem quadratic_roots_problem (k : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, x^2 - 2*(k+1)*x + k^2 + 3 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  1/x₁ + 1/x₂ = 6/7 →
  k = 2 ∧ x₁^2 + x₂^2 > 8 :=
by sorry

end quadratic_roots_problem_l3573_357383


namespace smaller_number_is_42_l3573_357312

theorem smaller_number_is_42 (x y : ℕ) (h1 : x + y = 96) (h2 : y = x + 12) : x = 42 := by
  sorry

end smaller_number_is_42_l3573_357312


namespace functional_equation_solution_l3573_357300

/-- The functional equation for f and g -/
def FunctionalEquation (f g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + g y) = x * f y - y * f x + g x

/-- The solution forms for f and g -/
def SolutionForms (f g : ℝ → ℝ) : Prop :=
  ∃ t : ℝ, t ≠ -1 ∧
    (∀ x : ℝ, f x = (t * (x - t)) / (t + 1)) ∧
    (∀ x : ℝ, g x = t * (x - t))

theorem functional_equation_solution :
    ∀ f g : ℝ → ℝ, FunctionalEquation f g → SolutionForms f g :=
  sorry

end functional_equation_solution_l3573_357300


namespace four_points_not_coplanar_iff_any_three_not_collinear_lines_no_common_point_iff_skew_l3573_357393

-- Define the types for points and lines in space
variable (Point Line : Type)

-- Define the properties
variable (coplanar : Point → Point → Point → Point → Prop)
variable (collinear : Point → Point → Point → Prop)
variable (have_common_point : Line → Line → Prop)
variable (skew : Line → Line → Prop)

-- Theorem 1
theorem four_points_not_coplanar_iff_any_three_not_collinear 
  (p1 p2 p3 p4 : Point) : 
  ¬(coplanar p1 p2 p3 p4) ↔ 
  (¬(collinear p1 p2 p3) ∧ ¬(collinear p1 p2 p4) ∧ 
   ¬(collinear p1 p3 p4) ∧ ¬(collinear p2 p3 p4)) :=
sorry

-- Theorem 2
theorem lines_no_common_point_iff_skew (l1 l2 : Line) :
  ¬(have_common_point l1 l2) ↔ skew l1 l2 :=
sorry

end four_points_not_coplanar_iff_any_three_not_collinear_lines_no_common_point_iff_skew_l3573_357393


namespace basketball_team_age_stats_l3573_357362

/-- Represents the age distribution of players in a basketball team -/
structure AgeDistribution :=
  (age18 : ℕ)
  (age19 : ℕ)
  (age20 : ℕ)
  (age21 : ℕ)
  (total : ℕ)
  (sum : ℕ)
  (h_total : age18 + age19 + age20 + age21 = total)
  (h_sum : 18 * age18 + 19 * age19 + 20 * age20 + 21 * age21 = sum)

/-- The mode of a set of ages -/
def mode (d : AgeDistribution) : ℕ :=
  max (max d.age18 d.age19) (max d.age20 d.age21)

/-- The mean of a set of ages -/
def mean (d : AgeDistribution) : ℚ :=
  d.sum / d.total

/-- Theorem stating the mode and mean of the given age distribution -/
theorem basketball_team_age_stats :
  ∃ d : AgeDistribution,
    d.age18 = 5 ∧
    d.age19 = 4 ∧
    d.age20 = 1 ∧
    d.age21 = 2 ∧
    d.total = 12 ∧
    mode d = 18 ∧
    mean d = 19 := by
  sorry

end basketball_team_age_stats_l3573_357362


namespace car_rental_cost_per_mile_l3573_357336

/-- Proves that the cost per mile for a car rental is $0.20 given specific conditions --/
theorem car_rental_cost_per_mile 
  (daily_fee : ℝ) 
  (daily_budget : ℝ) 
  (max_distance : ℝ) 
  (h1 : daily_fee = 50) 
  (h2 : daily_budget = 88) 
  (h3 : max_distance = 190) : 
  ∃ (cost_per_mile : ℝ), 
    cost_per_mile = 0.20 ∧ 
    daily_fee + cost_per_mile * max_distance = daily_budget :=
by
  sorry

end car_rental_cost_per_mile_l3573_357336


namespace remainder_777_pow_777_mod_13_l3573_357379

theorem remainder_777_pow_777_mod_13 : 777^777 % 13 = 1 := by
  sorry

end remainder_777_pow_777_mod_13_l3573_357379
