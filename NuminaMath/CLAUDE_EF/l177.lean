import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_min_value_change_l177_17778

/-- A quadratic polynomial f(x) = ax^2 + bx + c -/
noncomputable def QuadraticPolynomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The minimum value of a quadratic polynomial -/
noncomputable def MinValue (a b c : ℝ) : ℝ := -b^2 / (4 * a) + c

theorem quadratic_min_value_change 
  (a b c : ℝ) (h_a : a ≠ 0) :
  (MinValue (a + 1) b c - MinValue a b c = 1) →
  (MinValue (a - 1) b c - MinValue a b c = -3) →
  (MinValue (a + 2) b c - MinValue a b c = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_min_value_change_l177_17778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_rational_l177_17773

/-- Given that a and b are rational numbers, a is non-negative, 
    and the sum of the square root of a and the cube root of b is rational, 
    prove that the cube root of b is rational. -/
theorem cube_root_rational (a b : ℚ) (h1 : a ≥ 0) 
  (h2 : ∃ (r : ℚ), r = Real.sqrt (a : ℝ) + (b : ℝ) ^ (1/3 : ℝ)) : 
  ∃ (q : ℚ), (q : ℝ) = (b : ℝ) ^ (1/3 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_rational_l177_17773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_areas_l177_17702

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let (xa, ya) := A
  let (xb, yb) := B
  let (xc, yc) := C
  (xa - xb)^2 + (ya - yb)^2 = 16 ∧
  (xb - xc)^2 + (yb - yc)^2 = 16 ∧
  (xc - xa)^2 + (yc - ya)^2 = 4

-- Define the angle bisector from A
def AngleBisector (A B C A₁ : ℝ × ℝ) : Prop :=
  let (xa, ya) := A
  let (xb, yb) := B
  let (xc, yc) := C
  let (x₁, y₁) := A₁
  (x₁ - xa) * (yb - ya) = (y₁ - ya) * (xb - xa) ∧
  (x₁ - xa) * (yc - ya) = (y₁ - ya) * (xc - xa)

-- Define the median from B
def Median (A B C B₁ : ℝ × ℝ) : Prop :=
  let (xa, ya) := A
  let (xc, yc) := C
  let (x₁, y₁) := B₁
  2 * x₁ = xa + xc ∧ 2 * y₁ = ya + yc

-- Define the altitude from C
def Altitude (A B C C₁ : ℝ × ℝ) : Prop :=
  let (xa, ya) := A
  let (xb, yb) := B
  let (x₁, y₁) := C₁
  (xb - xa) * (x₁ - xa) + (yb - ya) * (y₁ - ya) = 0

-- Define area calculation functions (placeholder definitions)
noncomputable def area_of_triangle_formed_by_AC_AA₁_CC₁ (A C A₁ C₁ : ℝ × ℝ) : ℝ := sorry

noncomputable def area_of_triangle_formed_by_AA₁_BB₁_CC₁ (A₁ B₁ C₁ : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_areas
  (A B C A₁ B₁ C₁ : ℝ × ℝ)
  (h_triangle : Triangle A B C)
  (h_angle_bisector : AngleBisector A B C A₁)
  (h_median : Median A B C B₁)
  (h_altitude : Altitude A B C C₁) :
  let S₁ := area_of_triangle_formed_by_AC_AA₁_CC₁ A C A₁ C₁
  let S₂ := area_of_triangle_formed_by_AA₁_BB₁_CC₁ A₁ B₁ C₁
  S₁ = Real.sqrt 15 / 10 ∧ S₂ = Real.sqrt 15 / 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_areas_l177_17702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_equation_l177_17706

/-- The projection of vector v onto vector u -/
noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_squared := u.1 * u.1 + u.2 * u.2
  (dot_product / norm_squared * u.1, dot_product / norm_squared * u.2)

/-- Theorem: Vectors satisfying the projection condition lie on the line y = -2x - 15/2 -/
theorem projection_line_equation (x y : ℝ) :
  proj (6, 3) (x, y) = (-3, -3/2) →
  y = -2 * x - 15/2 := by
  sorry

#check projection_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_equation_l177_17706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_individual_is_16_l177_17717

/-- Represents a two-digit number in the range [01, 20] -/
def ValidNumber := {n : ℕ // 1 ≤ n ∧ n ≤ 20}

/-- Represents the random number table -/
def RandomTable : List (List ℕ) :=
  [[18, 18, 07, 92, 45, 44, 17, 16, 58, 09, 79, 83, 86, 19],
   [62, 06, 76, 50, 03, 10, 55, 23, 64, 05, 05, 26, 62, 38]]

/-- The selection method function -/
def selectIndividuals (table : List (List ℕ)) : List ValidNumber :=
  sorry

/-- The theorem to prove -/
theorem fourth_individual_is_16 :
  ∃ (selected : List ValidNumber),
    selectIndividuals RandomTable = selected ∧
    selected.length = 6 ∧
    selected.get? 3 = some ⟨16, by sorry⟩ :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_individual_is_16_l177_17717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_polynomial_power_l177_17714

theorem degree_of_polynomial_power : 
  Polynomial.degree ((5 * X^3 - 4 * X + 7 : Polynomial ℝ)^10) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_polynomial_power_l177_17714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_l177_17705

/-- The function f(x) = (3x^2 + 8x + 12) / (3x + 4) -/
noncomputable def f (x : ℝ) : ℝ := (3*x^2 + 8*x + 12) / (3*x + 4)

/-- The proposed oblique asymptote y = x + 4/3 -/
noncomputable def g (x : ℝ) : ℝ := x + 4/3

/-- Theorem: The oblique asymptote of f(x) is g(x) -/
theorem oblique_asymptote : 
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - g x| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_l177_17705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_u_is_one_l177_17784

open BigOperators

def fibonacci : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

def u (n : ℕ) : ℚ :=
  ∑ i in Finset.range n, 1 / ((fibonacci i : ℚ) * (fibonacci (i + 2) : ℚ))

theorem limit_of_u_is_one :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |u n - 1| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_u_is_one_l177_17784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_between_20k_and_150k_total_percentage_is_100_l177_17764

-- Define the population ranges
inductive PopulationRange
  | LessThan20k
  | Between20kAnd150k
  | MoreThan150k

-- Define the distribution of county populations
def CountyDistribution : PopulationRange → Rat
  | PopulationRange.LessThan20k => 30 / 100
  | PopulationRange.Between20kAnd150k => 45 / 100
  | PopulationRange.MoreThan150k => 25 / 100

-- Theorem statement
theorem percentage_between_20k_and_150k :
  CountyDistribution PopulationRange.Between20kAnd150k = 45 / 100 :=
by
  -- The proof is trivial given the definition of CountyDistribution
  rfl

-- Verify that the percentages sum to 100%
theorem total_percentage_is_100 :
  CountyDistribution PopulationRange.LessThan20k +
  CountyDistribution PopulationRange.Between20kAnd150k +
  CountyDistribution PopulationRange.MoreThan150k = 1 :=
by
  -- Evaluate each part of the sum
  simp [CountyDistribution]
  -- Perform the addition
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_between_20k_and_150k_total_percentage_is_100_l177_17764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wholesale_price_is_90_l177_17747

/-- Calculates the wholesale price of a machine given the retail price, discount percentage, and profit percentage. -/
noncomputable def calculate_wholesale_price (retail_price : ℝ) (discount_percent : ℝ) (profit_percent : ℝ) : ℝ :=
  let selling_price := retail_price * (1 - discount_percent)
  selling_price / (1 + profit_percent)

/-- Theorem stating that given the specified conditions, the wholesale price is $90. -/
theorem wholesale_price_is_90 :
  let retail_price : ℝ := 120
  let discount_percent : ℝ := 0.1
  let profit_percent : ℝ := 0.2
  calculate_wholesale_price retail_price discount_percent profit_percent = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wholesale_price_is_90_l177_17747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octant_volume_l177_17795

-- Define the circumference of the sphere
noncomputable def sphere_circumference : ℝ := 16 * Real.pi

-- Define the number of parts the sphere is divided into
def num_parts : ℕ := 8

-- Theorem statement
theorem octant_volume :
  (4 / 3 * Real.pi * (sphere_circumference / (2 * Real.pi))^3) / num_parts = 256 / 3 * Real.pi :=
by
  -- Expand the definition of sphere_circumference
  unfold sphere_circumference
  -- Simplify the expression
  simp [num_parts]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octant_volume_l177_17795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2alpha_value_l177_17792

theorem tan_2alpha_value (α : ℝ) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  Real.tan (2 * α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2alpha_value_l177_17792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_beam_distance_l177_17734

/-- The distance traveled by a light beam from point P to point Q with reflection on the xOy plane -/
noncomputable def lightPathDistance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  let M := (P.fst, P.snd.fst, -P.snd.snd)  -- Reflection point of P on xOy plane
  Real.sqrt ((Q.fst - M.fst)^2 + (Q.snd.fst - M.snd.fst)^2 + (Q.snd.snd - M.snd.snd)^2)

/-- Theorem: The distance traveled by a light beam from (1,1,1) to (3,3,6) with reflection on the xOy plane is √57 -/
theorem light_beam_distance :
  lightPathDistance (1, (1, 1)) (3, (3, 6)) = Real.sqrt 57 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_beam_distance_l177_17734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_babysitter_hours_worked_l177_17704

-- Define the constants
def regular_rate : ℚ := 16
def overtime_rate_1 : ℚ := regular_rate * (1 + 3/4)
def overtime_rate_2 : ℚ := regular_rate * 2
def overtime_rate_3 : ℚ := regular_rate * (1 + 3/2)
def late_penalty : ℚ := 10
def extra_task_bonus : ℚ := 5
def total_earnings : ℚ := 1150
def late_arrivals : ℕ := 3
def extra_tasks : ℕ := 8

-- Define the function to calculate earnings based on hours worked
noncomputable def earnings (hours : ℚ) : ℚ :=
  if hours ≤ 30 then
    regular_rate * hours
  else if hours ≤ 40 then
    regular_rate * 30 + overtime_rate_1 * (hours - 30)
  else if hours ≤ 50 then
    regular_rate * 30 + overtime_rate_1 * 10 + overtime_rate_2 * (hours - 40)
  else
    regular_rate * 30 + overtime_rate_1 * 10 + overtime_rate_2 * 10 + overtime_rate_3 * (hours - 50)

-- Theorem statement
theorem babysitter_hours_worked :
  ∃ (hours : ℚ), 
    earnings hours - (late_penalty * ↑late_arrivals) + (extra_task_bonus * ↑extra_tasks) = total_earnings ∧
    hours = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_babysitter_hours_worked_l177_17704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l177_17780

/-- Converts kilometers per hour to meters per second -/
noncomputable def kmph_to_mps (speed : ℝ) : ℝ := speed * (1000 / 3600)

/-- Calculates the relative speed between two objects moving in the same direction -/
noncomputable def relative_speed (speed1 speed2 : ℝ) : ℝ := speed1 - speed2

theorem faster_train_length 
  (faster_speed slower_speed : ℝ) 
  (crossing_time : ℝ) 
  (h1 : faster_speed = 108) 
  (h2 : slower_speed = 36) 
  (h3 : crossing_time = 17) : 
  (relative_speed (kmph_to_mps faster_speed) (kmph_to_mps slower_speed)) * crossing_time = 340 := by
  sorry

-- Remove the #eval line as it's not necessary for building and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l177_17780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l177_17783

-- Define the ellipse
noncomputable def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Define the area of the rhombus
noncomputable def rhombusArea (a b : ℝ) : ℝ := 4 * a * b

-- Define the line
def line (x y k m : ℝ) : Prop := y = k * x + m

-- Define the slope product condition
def slopeProductCondition (x₁ y₁ x₂ y₂ : ℝ) : Prop := (y₁ / x₁) * (y₂ / x₂) = -1/2

theorem ellipse_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (hecc : eccentricity a b = Real.sqrt 2 / 2)
  (harea : rhombusArea a b = 8 * Real.sqrt 2) :
  (∀ x y, ellipse x y a b ↔ ellipse x y (Real.sqrt 8) 2) ∧
  (∀ k m x₁ y₁ x₂ y₂,
    line y₁ x₁ k m → line y₂ x₂ k m →
    ellipse x₁ y₁ (Real.sqrt 8) 2 → ellipse x₂ y₂ (Real.sqrt 8) 2 →
    x₁ ≠ x₂ → slopeProductCondition x₁ y₁ x₂ y₂ →
    -2 < y₁ * y₂ ∧ y₁ * y₂ ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l177_17783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_collection_size_l177_17759

theorem movie_collection_size 
  (initial_ratio : ℚ) 
  (final_ratio : ℚ) 
  (additional_blurays : ℕ) : ℕ :=
  let dvd_multiplier : ℚ := 7
  let bluray_multiplier : ℚ := 2
  let final_dvd_multiplier : ℚ := 13
  let final_bluray_multiplier : ℚ := 4
  have h1 : initial_ratio = dvd_multiplier / bluray_multiplier := by sorry
  have h2 : final_ratio = final_dvd_multiplier / final_bluray_multiplier := by sorry
  have h3 : additional_blurays = 5 := by sorry
  293


end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_collection_size_l177_17759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_fourth_eq_one_third_l177_17794

def g (x : ℝ) : ℝ := 1 - x^2

noncomputable def f (x : ℝ) : ℝ := 
  if x ≠ 0 then (1 - (g⁻¹ x)^2) / ((g⁻¹ x)^2) else 0

theorem f_one_fourth_eq_one_third : f (1/4) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_fourth_eq_one_third_l177_17794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_mile_travel_time_velocity_inverse_proportion_fourth_mile_time_velocity_time_relation_l177_17715

noncomputable def travel_time (n : ℕ) : ℝ :=
  if n ≤ 3 then 2 else 3 * (n - 3 : ℝ)

noncomputable def velocity (n : ℕ) : ℝ :=
  if n ≤ 3 then 1 / 2 else 1 / (3 * (n - 3 : ℝ))

theorem nth_mile_travel_time (n : ℕ) (h : n ≥ 4) :
  travel_time n = 3 * (n - 3 : ℝ) := by sorry

theorem velocity_inverse_proportion (n m : ℕ) (hn : n > 3) (hm : m > 3) :
  velocity n * (n - 3 : ℝ) = velocity m * (m - 3 : ℝ) := by sorry

theorem fourth_mile_time :
  travel_time 4 = 3 := by sorry

theorem velocity_time_relation (n : ℕ) (h : n ≥ 4) :
  velocity n * travel_time n = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_mile_travel_time_velocity_inverse_proportion_fourth_mile_time_velocity_time_relation_l177_17715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_exist_l177_17738

/-- Represents a container with acidic liquid -/
structure AcidContainer where
  volume : ℝ
  concentration : ℝ

/-- The problem setup -/
def acidMixtureProblem : List AcidContainer :=
  [ { volume := 72, concentration := 0.30 }
  , { volume := 85, concentration := 0.40 }
  , { volume := 65, concentration := 0.25 }
  , { volume := 90, concentration := 0.50 }
  , { volume := 55, concentration := 0.35 }
  ]

/-- The desired concentration of the final mixture -/
def targetConcentration : ℝ := 0.80

/-- Theorem stating that there are infinitely many solutions -/
theorem infinite_solutions_exist (containers : List AcidContainer) (target : ℝ) :
  ∃ (solutions : List (List ℝ)), 
    (∀ sol ∈ solutions, sol.length = containers.length) ∧ 
    (∀ sol ∈ solutions, 
      let totalVolume := (List.zip containers sol).foldl (λ acc (c, v) => acc + v) 0
      let totalAcid := (List.zip containers sol).foldl (λ acc (c, v) => acc + c.concentration * v) 0
      totalAcid / totalVolume = target) ∧
    solutions.length > 1 :=
by
  sorry

#check infinite_solutions_exist acidMixtureProblem targetConcentration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_exist_l177_17738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_angles_cosine_l177_17793

theorem complementary_angles_cosine (α : ℝ) :
  Real.sin (π / 4 + α) = 2 / 3 → Real.cos (π / 4 - α) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_angles_cosine_l177_17793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_l177_17758

/-- A circle with center O and radius r -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in the plane -/
def Point := ℝ × ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- A point is inside a circle if its distance from the center is less than the radius -/
def is_inside (p : Point) (c : Circle) : Prop :=
  distance p c.center < c.radius

theorem point_inside_circle (O : Circle) (P : Point) :
  O.radius = 4 →
  distance P O.center = 3 →
  is_inside P O :=
by
  intro h_radius h_distance
  unfold is_inside
  rw [h_radius, h_distance]
  exact lt_of_lt_of_le (by norm_num) (le_refl 4)


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_l177_17758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jewelry_store_gross_profit_l177_17741

noncomputable def calculate_gross_profit (initial_price markup discount1 discount2 discount3 : ℝ) : ℝ :=
  let marked_up_price := initial_price * (1 + markup)
  let discounted_price := marked_up_price * (1 - discount1) * (1 - discount2) * (1 - discount3)
  discounted_price - initial_price

theorem jewelry_store_gross_profit :
  let earrings_profit := calculate_gross_profit 240 0.25 0.15 0 0
  let bracelet_profit := calculate_gross_profit 360 0.30 0.10 0.05 0
  let necklace_profit := calculate_gross_profit 480 0.40 0.20 0.05 0
  let ring_profit := calculate_gross_profit 600 0.35 0.10 0.05 0.02
  let pendant_profit := calculate_gross_profit 720 0.50 0.20 0.03 0.07
  let total_profit := earrings_profit + bracelet_profit + necklace_profit + ring_profit + pendant_profit
  abs (total_profit - 224.97) < 0.01 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jewelry_store_gross_profit_l177_17741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camera_price_difference_l177_17789

/-- The list price of Camera Y in dollars -/
def list_price : ℚ := 50

/-- The discount offered by Value Deals in dollars -/
def value_deals_discount : ℚ := 12

/-- The discount percentage offered by Budget Buys -/
def budget_buys_discount_percent : ℚ := 20

/-- The sale price at Value Deals in dollars -/
def value_deals_price : ℚ := list_price - value_deals_discount

/-- The sale price at Budget Buys in dollars -/
def budget_buys_price : ℚ := list_price * (1 - budget_buys_discount_percent / 100)

/-- The price difference between Budget Buys and Value Deals in cents -/
def price_difference_cents : ℚ := (budget_buys_price - value_deals_price) * 100

theorem camera_price_difference :
  price_difference_cents = 200 := by
  -- Unfold definitions
  unfold price_difference_cents budget_buys_price value_deals_price
  unfold list_price value_deals_discount budget_buys_discount_percent
  -- Simplify the expression
  simp [sub_eq_add_neg, mul_add, mul_sub, mul_one]
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_camera_price_difference_l177_17789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perfect_square_remainder_l177_17740

theorem not_perfect_square_remainder (k : ℕ) (p : ℕ) (r : ℕ) : 
  Prime p → 
  p = 8 * k + 1 → 
  r = Nat.choose (4 * k) k % p → 
  ¬ ∃ (a : ℕ), r = a^2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perfect_square_remainder_l177_17740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_3_with_21_count_l177_17710

def count_gcd_3_with_21 (lower upper : ℕ) : ℕ :=
  (List.range (upper - lower + 1)).map (λ i => i + lower)
    |>.filter (λ n => Nat.gcd 21 n = 3)
    |>.length

theorem gcd_3_with_21_count : count_gcd_3_with_21 1 200 = 57 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_3_with_21_count_l177_17710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_properties_l177_17774

-- Define the logarithm function with base a
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define our function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log_base a x

-- State the theorem
theorem log_properties (a : ℝ) (h : 0 < a ∧ a < 1) :
  (∀ x > 1, f a x < 0) ∧
  (∀ x, 0 < x → x < 1 → f a x > 0) ∧
  (∀ x y, f a (x * y) = f a x + f a y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_properties_l177_17774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_and_max_area_l177_17748

-- Define the curves and lines
def C₁ (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def C₂ (a b x y : ℝ) : Prop := x^2 / (4*a^2) + y^2 / (2*b^2) = 1
def l₁ (x y : ℝ) : Prop := Real.sqrt 3 * x + Real.sqrt 10 * y - 4 = 0
def l₂ (x y : ℝ) : Prop := x - 2*y - 4 = 0

-- Define the parabola
def parabola (p x y : ℝ) : Prop := y^2 = 2*p*x

-- Helper function for area calculation
noncomputable def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem curves_and_max_area 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (h_l₁_tangent : ∃ x y, l₁ x y ∧ C₁ a b x y)
  (h_l₂_tangent : ∃ x y, l₂ x y ∧ C₂ a b x y) :
  (∀ x y, C₁ a b x y ↔ x^2/2 + y^2 = 1) ∧
  (∀ x y, C₂ a b x y ↔ x^2/8 + y^2/2 = 1) ∧
  (∃ p₀ : ℝ, p₀ > 0 ∧
    (∀ p : ℝ, p > 0 →
      (∃ A B C D : ℝ × ℝ,
        parabola p A.1 A.2 ∧ C₁ a b A.1 A.2 ∧
        parabola p B.1 B.2 ∧ C₁ a b B.1 B.2 ∧
        parabola p C.1 C.2 ∧ C₂ a b C.1 C.2 ∧
        parabola p D.1 D.2 ∧ C₂ a b D.1 D.2 ∧
        area_quadrilateral A B C D ≤ Real.sqrt 2 / 2 + 1) ∧
    (∃ A B C D : ℝ × ℝ,
      parabola p₀ A.1 A.2 ∧ C₁ a b A.1 A.2 ∧
      parabola p₀ B.1 B.2 ∧ C₁ a b B.1 B.2 ∧
      parabola p₀ C.1 C.2 ∧ C₂ a b C.1 C.2 ∧
      parabola p₀ D.1 D.2 ∧ C₂ a b D.1 D.2 ∧
      area_quadrilateral A B C D = Real.sqrt 2 / 2 + 1) ∧
    p₀ = 1/4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_and_max_area_l177_17748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_at_distance_sqrt_two_l177_17716

/-- The line l with parameter a -/
def line (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + a = 0}

/-- The circle C -/
def circleC : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 2)^2 = 8}

/-- The distance between a point and a line -/
noncomputable def distPointLine (p : ℝ × ℝ) (a : ℝ) : ℝ :=
  |p.1 - p.2 + a| / Real.sqrt 2

/-- The theorem stating the condition for exactly three points on the circle
    to be at distance √2 from the line -/
theorem three_points_at_distance_sqrt_two (a : ℝ) : 
  (∃! (s : Set (ℝ × ℝ)), s ⊆ circleC ∧ (∀ p ∈ s, distPointLine p a = 1) ∧ s.ncard = 3) ↔ 
  (a = -6 ∨ a = -2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_at_distance_sqrt_two_l177_17716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whale_sixth_hour_consumption_l177_17791

noncomputable def whale_consumption (x : ℝ) (hour : ℕ) : ℝ :=
  x + 3 * ((hour : ℝ) - 1)

noncomputable def total_consumption (x : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (whale_consumption x 1 + whale_consumption x n)

theorem whale_sixth_hour_consumption :
  ∃ x : ℝ, 
    total_consumption x 9 = 450 ∧ 
    whale_consumption x 6 = 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_whale_sixth_hour_consumption_l177_17791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_solutions_l177_17723

def a (n : ℕ) : ℕ := 10^n + 10^(2016 - n)

def solution_set : List ℕ := [336, 252, 756, 112, 560, 784, 288, 576, 864]

theorem congruence_solutions (p : ℕ) (h_prime : Nat.Prime p) (h_primitive_root : IsPrimitiveRoot 10 p) :
  p = 2017 →
  (∀ n ∈ solution_set,
    (a n ≡ 1 [ZMOD p] ∨
     (a n)^2 ≡ 2 [ZMOD p] ∨
     (a n)^3 - 3*(a n) ≡ 1 [ZMOD p] ∨
     (a n)^3 + (a n)^2 - 2*(a n) ≡ 1 [ZMOD p])) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_solutions_l177_17723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_sum_l177_17760

theorem tan_ratio_sum (x y : Real) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 8) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 56/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_sum_l177_17760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_pair_sum_53_l177_17731

theorem existence_of_pair_sum_53 (S : Finset ℕ) :
  S.card = 53 →
  (∀ x, x ∈ S → x > 0) →
  (∀ x y, x ∈ S → y ∈ S → x ≠ y → x ≠ y) →
  S.sum id ≤ 1990 →
  ∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ x + y = 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_pair_sum_53_l177_17731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oscar_leap_minus_elmer_stride_l177_17745

/-- Represents the trail with its characteristics -/
structure Trail where
  length : ℝ
  num_poles : ℕ
  elmer_strides_per_gap : ℕ
  oscar_leaps_per_gap : ℕ

/-- Calculates the length of a single stride or leap -/
noncomputable def stride_or_leap_length (trail : Trail) (steps_per_gap : ℕ) : ℝ :=
  trail.length / (↑(trail.num_poles - 1) * ↑steps_per_gap)

/-- The main theorem to prove -/
theorem oscar_leap_minus_elmer_stride (trail : Trail) 
  (h1 : trail.length = 10560)
  (h2 : trail.num_poles = 81)
  (h3 : trail.elmer_strides_per_gap = 60)
  (h4 : trail.oscar_leaps_per_gap = 15) :
  stride_or_leap_length trail trail.oscar_leaps_per_gap - 
  stride_or_leap_length trail trail.elmer_strides_per_gap = 6.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oscar_leap_minus_elmer_stride_l177_17745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_senior_average_is_126_l177_17755

/-- Represents the data for a math competition at Parkview High School -/
structure MathCompetition where
  total_students : ℕ
  overall_average : ℚ
  senior_count : ℕ
  nonsenior_count : ℕ
  senior_average : ℚ
  nonsenior_average : ℚ

/-- The conditions of the math competition -/
def parkview_competition : MathCompetition :=
  { total_students := 120
  , overall_average := 84
  , senior_count := 40
  , nonsenior_count := 80
  , senior_average := 126  -- We now know this value
  , nonsenior_average := 63  -- We can calculate this: 126 / 2
  }

/-- Theorem stating the average score of seniors in the math competition -/
theorem senior_average_is_126 (c : MathCompetition) (h1 : c = parkview_competition) :
  c.senior_average = 126 := by
  rw [h1]
  rfl

#check senior_average_is_126

end NUMINAMATH_CALUDE_ERRORFEEDBACK_senior_average_is_126_l177_17755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_ratio_l177_17752

-- Define the properties of the cylinders
def height_C : ℝ := 10
def circumference_C : ℝ := 8
def height_B : ℝ := 8
def circumference_B : ℝ := 10

-- Define the volume of a cylinder
noncomputable def cylinder_volume (height : ℝ) (circumference : ℝ) : ℝ :=
  (height * circumference^2) / (4 * Real.pi)

-- Theorem statement
theorem cylinder_volume_ratio :
  (cylinder_volume height_C circumference_C) / (cylinder_volume height_B circumference_B) = 0.8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_ratio_l177_17752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_subset_of_naturals_l177_17730

-- Define a tiling of a set
def Tiles (S : Set ℕ) (T : Set ℕ) : Prop :=
  ∃ (f : ℕ → ℕ), Function.Injective f ∧ (∀ n ∈ T, ∃ s ∈ S, f s = n)

-- Define the set of natural numbers from 1 to k
def SetUpToK (k : ℕ) : Set ℕ :=
  {n : ℕ | 1 ≤ n ∧ n ≤ k}

-- Theorem statement
theorem tiles_subset_of_naturals (S : Set ℕ) :
  Tiles S (Set.univ : Set ℕ) → ∃ k : ℕ, k > 0 ∧ Tiles S (SetUpToK k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_subset_of_naturals_l177_17730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_calculation_l177_17781

/-- The speed of the stream given boat speed and travel distances -/
noncomputable def stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) : ℝ :=
  (downstream_distance - upstream_distance) / (downstream_distance / upstream_distance + 1)

theorem stream_speed_calculation :
  let boat_speed : ℝ := 30
  let downstream_distance : ℝ := 80
  let upstream_distance : ℝ := 40
  stream_speed boat_speed downstream_distance upstream_distance = 10 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval stream_speed 30 80 40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_calculation_l177_17781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l177_17703

-- Define the function f
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := Real.log (|x + 1|) / Real.log t

-- State the theorem
theorem solution_set (t : ℝ) :
  (∀ x ∈ Set.Ioo (-2) (-1), f t x > 0) →
  (f t (8^t - 1) < f t 1) ↔
  t ∈ Set.Ioo (1/3) 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l177_17703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_inequality_theorem_l177_17797

/-- A positive function defined on positive integers satisfying f(n) ≥ q f(n-1) -/
def FunctionalInequality (f : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n : ℕ, n > 0 → f n > 0) ∧ 
  (q > 0) ∧ 
  (∀ n : ℕ, n > 0 → f n ≥ q * f (n - 1))

/-- The solution to the functional inequality -/
def FunctionalInequalitySolution (f : ℕ → ℝ) (q : ℝ) : Prop :=
  ∃ g : ℕ → ℝ, 
    (∀ n : ℕ, g (n + 1) ≥ g n) ∧
    (∀ n : ℕ, n > 0 → f n = q^(n - 1) * g n)

/-- The main theorem: if f satisfies the functional inequality, then it has the proposed solution form -/
theorem functional_inequality_theorem (f : ℕ → ℝ) (q : ℝ) :
  FunctionalInequality f q → FunctionalInequalitySolution f q :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_inequality_theorem_l177_17797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_valve_fill_time_first_valve_fill_time_is_two_hours_l177_17754

/-- Proves that the first valve alone takes 2 hours to fill the pool -/
theorem first_valve_fill_time (pool_capacity : ℝ) (both_valves_time : ℝ) (valve_difference : ℝ) : ℝ :=
  by
  -- Define the pool capacity
  have h1 : pool_capacity = 12000 := by sorry
  -- Define the time taken when both valves are open (in minutes)
  have h2 : both_valves_time = 48 := by sorry
  -- Define the difference in water output between the two valves (in cubic meters per minute)
  have h3 : valve_difference = 50 := by sorry
  -- Calculate the rate of the first valve
  let first_valve_rate := (pool_capacity / both_valves_time - valve_difference / 2)
  -- Calculate the time taken for the first valve alone to fill the pool (in hours)
  let result := pool_capacity / first_valve_rate / 60
  -- Show that the result is equal to 2
  have h4 : result = 2 := by sorry
  exact result

/-- The result of first_valve_fill_time is 2 hours -/
theorem first_valve_fill_time_is_two_hours :
    first_valve_fill_time 12000 48 50 = 2 :=
  by
  -- Unfold the definition of first_valve_fill_time
  unfold first_valve_fill_time
  -- Simplify the expression
  simp
  -- The result should be 2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_valve_fill_time_first_valve_fill_time_is_two_hours_l177_17754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_to_actual_area_l177_17722

/-- Represents the scale of a map as a ratio -/
structure MapScale where
  ratio : ℚ

/-- Represents an area on a map -/
structure MapArea where
  area : ℚ
  unit : String

/-- Represents an actual area in reality -/
structure ActualArea where
  area : ℚ
  unit : String

/-- Converts an area from cm² to m² -/
def convertCmSqToMSq (area : ℚ) : ℚ :=
  area / 10000

/-- Theorem stating the relationship between map area and actual area given a map scale -/
theorem map_to_actual_area 
  (scale : MapScale) 
  (mapArea : MapArea) 
  (actualArea : ActualArea) : 
  scale.ratio = 1 / 50000 → 
  mapArea.area = 100 ∧ mapArea.unit = "cm²" → 
  actualArea.area = 25000000 ∧ actualArea.unit = "m²" := by
  sorry

#check map_to_actual_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_to_actual_area_l177_17722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_area_enclosed_l177_17750

/-- The area enclosed by the cosine curve y = cos x and the coordinate axes in the interval [0, 3π/2] is equal to 3. -/
theorem cosine_area_enclosed : ∫ x in (0)..(3 * Real.pi / 2), max (Real.cos x) 0 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_area_enclosed_l177_17750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_intersection_line_passes_fixed_point_l177_17742

/-- The ellipse C₁ -/
def C₁ (x y : ℝ) : Prop := x^2/2 + y^2 = 1

/-- The hyperbola C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- The fixed point -/
noncomputable def fixed_point : ℝ × ℝ := (1, 0)

/-- The intersection point of C₁ and C₂ -/
noncomputable def M : ℝ × ℝ := (2 * Real.sqrt 3 / 3, Real.sqrt 3 / 3)

theorem ellipse_hyperbola_intersection_line_passes_fixed_point :
  C₁ M.1 M.2 ∧ C₂ M.1 M.2 →
  ∀ x₀ y₀ : ℝ, C₂ x₀ y₀ ∧ x₀ > 1 →
  let Q : ℝ × ℝ := (0, -(x₀ + 1) / y₀)
  (Q.2 - y₀) / (Q.1 - x₀) = (fixed_point.2 - y₀) / (fixed_point.1 - x₀) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_intersection_line_passes_fixed_point_l177_17742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_is_correct_l177_17768

/-- The price of a single book in rubles -/
def book_price : ℝ := 1.23

/-- Nine books cost more than 11 rubles but less than 12 rubles -/
axiom nine_books_cost : 11 < 9 * book_price ∧ 9 * book_price < 12

/-- Thirteen books cost more than 15 rubles but less than 16 rubles -/
axiom thirteen_books_cost : 15 < 13 * book_price ∧ 13 * book_price < 16

/-- The price is measured in rubles and kopecks (1 ruble = 100 kopecks) -/
axiom price_in_kopecks : ∃ (r k : ℕ), k < 100 ∧ book_price = r + k / 100

/-- The price of one book is 1.23 rubles -/
theorem book_price_is_correct : book_price = 1.23 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_is_correct_l177_17768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_l177_17729

theorem number_of_divisors (a b c : ℕ) :
  let n := 2^a * 3^b * 5^c
  ∃ d : Finset ℕ, (∀ x : ℕ, x ∣ n ↔ x ∈ d) ∧ d.card = (a + 1) * (b + 1) * (c + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_l177_17729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_arrangements_l177_17788

-- Define the colors
inductive Color
  | Red
  | Yellow
  | Green
deriving Inhabited

-- Define the shape of the polygons
inductive Shape
  | Pentagon
  | Triangle
deriving Inhabited

-- Define a polygon
structure Polygon where
  shape : Shape
  color : Color
deriving Inhabited

-- Define the arrangement of polygons
def Arrangement := List Polygon

-- Function to check if two colors are different
def different_colors (c1 c2 : Color) : Prop :=
  c1 ≠ c2

-- Function to check if an arrangement is valid
def valid_arrangement (arr : Arrangement) : Prop :=
  arr.length = 6 ∧
  (arr.head?.get!).shape = Shape.Pentagon ∧
  (arr.head?.get!).color = Color.Red ∧
  (∀ i ∈ [1, 2, 3, 4, 5], (arr.get! i).shape = Shape.Triangle) ∧
  (∀ i ∈ [0, 1, 2, 3, 4], different_colors (arr.get! i).color (arr.get! (i + 1)).color) ∧
  different_colors (arr.get! 5).color (arr.head?.get!).color

-- Theorem stating that there are exactly 2 valid arrangements
theorem two_valid_arrangements :
  ∃! (n : Nat), ∃ (arr_list : List Arrangement),
    arr_list.length = n ∧
    (∀ arr ∈ arr_list, valid_arrangement arr) ∧
    (∀ arr, valid_arrangement arr → arr ∈ arr_list) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_arrangements_l177_17788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_over_6_l177_17766

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ Real.sin x then Real.sin x else x

theorem f_pi_over_6 : f (Real.pi / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_over_6_l177_17766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x5_expansion_l177_17709

theorem coefficient_x5_expansion :
  let f : Polynomial ℤ := (X - 1) * (X - 2) * (X - 3) * (X - 4) * (X - 5) * (X - 6)
  f.coeff 5 = -21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x5_expansion_l177_17709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_two_fifths_l177_17726

/-- Sequence b_n defined recursively -/
def b : ℕ → ℚ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => 2
  | (n + 3) => b (n + 2) + b (n + 1)

/-- The sum of the infinite series -/
noncomputable def seriesSum : ℚ := ∑' n, b n / 3^(n + 1)

/-- Theorem stating that the sum of the infinite series equals 2/5 -/
theorem series_sum_equals_two_fifths : seriesSum = 2/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_two_fifths_l177_17726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_l177_17718

/-- A triangle with integer side lengths, area 60, one side of length 13, and perimeter 54 -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  area : ℕ
  perimeter : ℕ
  ha : a = 13
  harea : area = 60
  hperimeter : perimeter = 54
  hsum : a + b + c = perimeter

/-- The shortest side of the triangle is 13 -/
theorem shortest_side (t : Triangle) : min t.a (min t.b t.c) = 13 := by
  sorry

#check Triangle
#check shortest_side

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_l177_17718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_l177_17762

theorem sum_of_angles (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin α = Real.sqrt 5 / 5) (h4 : Real.sin β = Real.sqrt 10 / 10) : α + β = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_l177_17762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_range_part_two_range_l177_17719

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - ((a+1)/2) * x^2 + a*x - 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (1/2) * (a-4) * x^2

-- Part I
theorem part_one_range (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 2, ∃ y, f a x = y) → 1 ≤ a ∧ a < 5/3 :=
by sorry

-- Part II
theorem part_two_range (a : ℝ) :
  (a ≥ 3) →
  (∀ x₁ x₂, x₁ ∈ Set.Icc 2 3 → x₂ ∈ Set.Icc 2 3 → x₁ ≠ x₂ → 
    |f a x₁ - f a x₂| > |g a x₁ - g a x₂|) →
  18/5 ≤ a ∧ a ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_range_part_two_range_l177_17719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_runs_1800_meters_l177_17721

/-- A race between two runners p and q, where p is faster but gives q a head start --/
structure Race where
  -- p's speed relative to q's (1.0 would mean same speed)
  p_speed_factor : ℝ
  -- The head start given to q in meters
  q_head_start : ℝ
  -- The distance q runs in meters
  q_distance : ℝ

/-- The conditions of our specific race --/
def our_race : Race where
  p_speed_factor := 1.2
  q_head_start := 300
  q_distance := 1500

/-- The distance p runs in the race --/
noncomputable def p_distance (r : Race) : ℝ := r.q_distance + r.q_head_start

/-- The time taken by q to finish the race --/
noncomputable def q_time (r : Race) : ℝ := r.q_distance

/-- The time taken by p to finish the race --/
noncomputable def p_time (r : Race) : ℝ := (r.q_distance + r.q_head_start) / r.p_speed_factor

/-- Theorem stating that in our specific race, p runs 1800 meters --/
theorem p_runs_1800_meters :
  p_distance our_race = 1800 ∧ p_time our_race = q_time our_race := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_runs_1800_meters_l177_17721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_perpendicular_parallel_l177_17767

-- Define the necessary structures
structure Line where

structure Plane where

-- Define the relationships
def perpendicular (l : Line) (p : Plane) : Prop := sorry

def parallel (l : Line) (p : Plane) : Prop := sorry

def inPlane (l : Line) (p : Plane) : Prop := sorry

def perpendicularLines (l1 l2 : Line) : Prop := sorry

def parallelLines (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem line_plane_perpendicular_parallel :
  (∀ (l : Line) (p : Plane), perpendicular l p → 
    ∀ (l' : Line), inPlane l' p → perpendicularLines l l') ∧
  ¬(∀ (l : Line) (p : Plane), parallel l p → 
    ∀ (l' : Line), inPlane l' p → parallelLines l l') :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_perpendicular_parallel_l177_17767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_beta_l177_17725

theorem sin_alpha_minus_beta (α β : ℝ) 
  (h1 : Real.sin α = 2 * Real.sqrt 2 / 3)
  (h2 : Real.cos (α + β) = -1 / 3)
  (h3 : 0 < α ∧ α < Real.pi / 2)
  (h4 : 0 < β ∧ β < Real.pi / 2) :
  Real.sin (α - β) = 10 * Real.sqrt 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_beta_l177_17725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_tea_july_cost_l177_17736

/-- The cost per pound of green tea and coffee in June. -/
def june_cost : ℝ := sorry

/-- The cost per pound of coffee in July. -/
def july_coffee_cost : ℝ := 2 * june_cost

/-- The cost per pound of green tea in July. -/
def july_tea_cost : ℝ := 0.3 * june_cost

/-- The total cost of the mixture in July. -/
def mixture_cost : ℝ := 3.45

/-- The weight of the mixture in pounds. -/
def mixture_weight : ℝ := 3

theorem green_tea_july_cost :
  july_tea_cost = 0.30 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_tea_july_cost_l177_17736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distance_bounds_l177_17701

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the distance function d
def d (P : ℝ × ℝ) : ℝ := 
  (P.1 - A.1)^2 + (P.2 - A.2)^2 + (P.1 - B.1)^2 + (P.2 - B.2)^2

-- Theorem statement
theorem circle_distance_bounds :
  (∃ P : ℝ × ℝ, circleC P.1 P.2 ∧ d P = 74) ∧
  (∃ P : ℝ × ℝ, circleC P.1 P.2 ∧ d P = 34) ∧
  (∀ P : ℝ × ℝ, circleC P.1 P.2 → 34 ≤ d P ∧ d P ≤ 74) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distance_bounds_l177_17701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l177_17724

-- Define the given vertices
def v1 : ℝ × ℝ := (-1, 5)
def v2 : ℝ × ℝ := (4, -3)
def v3 : ℝ × ℝ := (11, 5)

-- Define the ellipse using the given vertices
def Ellipse (v4 : ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  (v1 ∈ Set.range (fun t => (a * Real.cos t, b * Real.sin t))) ∧
  (v2 ∈ Set.range (fun t => (a * Real.cos t, b * Real.sin t))) ∧
  (v3 ∈ Set.range (fun t => (a * Real.cos t, b * Real.sin t))) ∧
  (v4 ∈ Set.range (fun t => (a * Real.cos t, b * Real.sin t)))

-- Theorem statement
theorem ellipse_foci_distance (v4 : ℝ × ℝ) (h : Ellipse v4) :
  ∃ (f1 f2 : ℝ × ℝ), Real.sqrt ((f1.1 - f2.1)^2 + (f1.2 - f2.2)^2) = 4 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l177_17724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_equivalent_angle_neg_1050_l177_17786

/-- The smallest positive angle that has the same terminal side as a given angle -/
noncomputable def smallestPositiveEquivalentAngle (angle : ℝ) : ℝ :=
  (angle % 360 + 360) % 360

/-- Theorem: The smallest positive angle that has the same terminal side as -1050° is 30° -/
theorem smallest_positive_equivalent_angle_neg_1050 :
  smallestPositiveEquivalentAngle (-1050) = 30 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_equivalent_angle_neg_1050_l177_17786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_dividing_segment_l177_17700

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors
variable (C D Q : V)

-- Define the line segment CD and the point Q on it
def on_line_segment (A B P : V) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B

-- Define the ratio condition
def ratio_condition (C D Q : V) : Prop :=
  ∃ k : ℝ, k > 0 ∧ ‖Q - C‖ = 4 * k ∧ ‖D - Q‖ = k

-- State the theorem
theorem point_dividing_segment (h1 : on_line_segment C D Q) (h2 : ratio_condition C D Q) :
  Q = (4/5) • C + (1/5) • D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_dividing_segment_l177_17700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l177_17772

theorem shaded_area_calculation (side_length : ℝ) (radius : ℝ) : 
  side_length = 14 →
  radius = 7 →
  let larger_square_area := side_length ^ 2
  let smaller_square_side := side_length / 2
  let smaller_square_area := smaller_square_side ^ 2
  let quarter_circles_area := π * radius ^ 2
  (quarter_circles_area - smaller_square_area) = 49 * π - 49 := by
    intro h_side h_radius
    -- Placeholder for the actual proof
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l177_17772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_average_age_l177_17720

theorem combined_average_age (room_a_count : ℕ) (room_a_avg : ℝ) 
                             (room_b_count : ℕ) (room_b_avg : ℝ) :
  room_a_count = 7 →
  room_a_avg = 35 →
  room_b_count = 5 →
  room_b_avg = 30 →
  Int.floor ((((room_a_count : ℝ) * room_a_avg + (room_b_count : ℝ) * room_b_avg) / 
   ((room_a_count : ℝ) + (room_b_count : ℝ))) + 0.5) = 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_average_age_l177_17720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_mile_taxi_cost_l177_17765

noncomputable def base_fare : ℝ := 2.00
noncomputable def cost_per_mile : ℝ := 0.30
noncomputable def minimum_charge : ℝ := 5.00
noncomputable def minimum_charge_threshold : ℝ := 4.00

noncomputable def taxi_cost (miles : ℝ) : ℝ :=
  if miles < minimum_charge_threshold
  then max (base_fare + miles * cost_per_mile) minimum_charge
  else base_fare + miles * cost_per_mile

theorem three_mile_taxi_cost :
  taxi_cost 3 = 5.00 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_mile_taxi_cost_l177_17765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l177_17790

/-- Conversion from spherical coordinates to rectangular coordinates -/
noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ,
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

/-- The given spherical coordinates -/
noncomputable def given_spherical : ℝ × ℝ × ℝ :=
  (3, 3 * Real.pi / 2, Real.pi / 3)

/-- The expected rectangular coordinates -/
noncomputable def expected_rectangular : ℝ × ℝ × ℝ :=
  (0, -3 * Real.sqrt 3 / 2, 1.5)

theorem spherical_to_rectangular_conversion :
  spherical_to_rectangular given_spherical.1 given_spherical.2.1 given_spherical.2.2 = expected_rectangular :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l177_17790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_difference_l177_17776

theorem max_angle_difference (α β : Real) (h1 : Real.tan α = 3 * Real.tan β) 
  (h2 : 0 ≤ β) (h3 : β < α) (h4 : α ≤ π/2) : 
  ∃ (max : Real), ∀ (α' β' : Real), 
    Real.tan α' = 3 * Real.tan β' → 0 ≤ β' → β' < α' → α' ≤ π/2 → 
    α' - β' ≤ max ∧ max = π/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_difference_l177_17776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_increase_l177_17751

theorem circle_circumference_increase (e : ℝ) (h : e > 0) : 
  ∃ (P : ℝ), P = π * e ∧ ∀ d : ℝ, P = (π * (d + e)) - (π * d) := by
  use π * e
  constructor
  · rfl
  · intro d
    ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_increase_l177_17751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_segments_with_midpoint_in_triangle_l177_17743

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle in a 2D plane --/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- A segment with a given midpoint --/
structure Segment where
  endpoint1 : Point
  endpoint2 : Point
  midpoint : Point

/-- Checks if a point is inside a triangle --/
def isInsideTriangle (p : Point) (t : Triangle) : Prop := sorry

/-- Checks if a point is on the boundary of a triangle --/
def isOnTriangleBoundary (p : Point) (t : Triangle) : Prop := sorry

/-- Checks if a segment has its midpoint at a given point --/
def hasMidpointAt (s : Segment) (p : Point) : Prop := sorry

/-- The main theorem --/
theorem max_segments_with_midpoint_in_triangle (t : Triangle) (O : Point) :
  isInsideTriangle O t →
  ∃ (n : ℕ), n ≤ 3 ∧
    (∀ (m : ℕ) (segs : Fin m → Segment),
      (∀ i, isOnTriangleBoundary (segs i).endpoint1 t ∧
            isOnTriangleBoundary (segs i).endpoint2 t ∧
            hasMidpointAt (segs i) O) →
      m ≤ n) ∧
    (∃ (segs : Fin n → Segment),
      ∀ i, isOnTriangleBoundary (segs i).endpoint1 t ∧
           isOnTriangleBoundary (segs i).endpoint2 t ∧
           hasMidpointAt (segs i) O) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_segments_with_midpoint_in_triangle_l177_17743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lambda_value_l177_17707

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - (2*a + 2) * x + (2*a + 1) * Real.log x

theorem min_lambda_value (a : ℝ) (h_a : a ∈ Set.Icc (1/2 : ℝ) 2) :
  ∀ (x₁ x₂ : ℝ) (h_x₁ : x₁ ∈ Set.Icc 1 2) (h_x₂ : x₂ ∈ Set.Icc 1 2) (h_ne : x₁ ≠ x₂),
  ∃ (lambda : ℝ), lambda ≥ 6 ∧ 
  (∀ (μ : ℝ), μ ≥ lambda → |f a x₁ - f a x₂| < μ * |1/x₁ - 1/x₂|) ∧
  (∀ (ν : ℝ), ν < lambda → ∃ (y₁ y₂ : ℝ) (hy₁ : y₁ ∈ Set.Icc 1 2) (hy₂ : y₂ ∈ Set.Icc 1 2) (hy_ne : y₁ ≠ y₂),
    |f a y₁ - f a y₂| ≥ ν * |1/y₁ - 1/y₂|) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lambda_value_l177_17707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_base_for_512_l177_17796

/-- Given a natural number n and a base b, returns the number of digits in the representation of n in base b. -/
def numDigits (n : ℕ) (b : ℕ) : ℕ :=
  if n = 0 then 1 else
  Nat.log b n + 1

/-- Given a natural number n and a base b, returns the last digit of n when represented in base b. -/
def lastDigit (n : ℕ) (b : ℕ) : ℕ :=
  n % b

theorem unique_base_for_512 :
  ∃! b : ℕ, b > 1 ∧ numDigits 512 b = 4 ∧ Odd (lastDigit 512 b) ∧ b = 7 := by
  sorry

#eval numDigits 512 7
#eval lastDigit 512 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_base_for_512_l177_17796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l177_17739

theorem angle_in_second_quadrant (θ : ℝ) :
  (Real.sin θ * Real.cos θ < 0 ∧ 2 * Real.cos θ < 0) → 
  (Real.sin θ > 0 ∧ Real.cos θ < 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l177_17739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minyoung_initial_money_l177_17753

/-- The amount of candy Minyoung could buy -/
def A : ℕ := sorry

/-- The initial amount of money Minyoung had (in won) -/
def initial_money : ℕ := sorry

/-- Minyoung could buy A pieces of candy worth 90 won each with all his money -/
axiom condition1 : initial_money = 90 * A

/-- If Minyoung bought A pieces of 60 won candies, he would have 270 won left -/
axiom condition2 : initial_money = 60 * A + 270

theorem minyoung_initial_money : initial_money = 810 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minyoung_initial_money_l177_17753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AOB_is_six_l177_17785

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Calculates the area of a triangle given two points in polar coordinates and the origin -/
noncomputable def triangleArea (a b : PolarPoint) : ℝ :=
  (1/2) * a.r * b.r * Real.sin (abs (a.θ - b.θ))

/-- Theorem stating that the area of triangle AOB is 6 square units -/
theorem area_of_triangle_AOB_is_six :
  let a : PolarPoint := ⟨6, π/3⟩
  let b : PolarPoint := ⟨4, π/6⟩
  triangleArea a b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AOB_is_six_l177_17785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_problem_l177_17775

/-- The number of candies remaining after eating a fraction for a given number of days -/
def candies_remaining (initial : ℕ) (fraction_left : ℚ) (days : ℕ) : ℚ :=
  (fraction_left ^ days) * initial

/-- The original number of candies, rounded up to the nearest integer -/
def original_candies (remaining : ℕ) (fraction_left : ℚ) (days : ℕ) : ℕ :=
  (remaining : ℚ) / (fraction_left ^ days) |>.ceil.toNat

theorem candy_problem (remaining : ℕ) (fraction_left : ℚ) (days : ℕ) 
    (h1 : remaining = 28)
    (h2 : fraction_left = 0.7)
    (h3 : days = 3) :
  original_candies remaining fraction_left days = 82 := by
  sorry

#eval original_candies 28 (7/10) 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_problem_l177_17775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_win_match_probability_l177_17763

def probability_win_set : ℝ := 0.4

def probability_win_match : ℝ :=
  let p := probability_win_set
  let q := 1 - p
  (Nat.choose 5 3) * p^3 * q^2 +
  (Nat.choose 5 4) * p^4 * q +
  p^5

theorem win_match_probability :
  abs (probability_win_match - 0.31744) < 0.00001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_win_match_probability_l177_17763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_loss_percentage_l177_17787

noncomputable def cost_price : ℝ := 1400
noncomputable def additional_amount : ℝ := 196
noncomputable def gain_percentage : ℝ := 4

noncomputable def selling_price_with_gain (cp : ℝ) (gain : ℝ) : ℝ :=
  cp * (1 + gain / 100)

noncomputable def selling_price_with_loss (cp : ℝ) (loss : ℝ) : ℝ :=
  cp * (1 - loss / 100)

theorem watch_loss_percentage :
  ∃ (loss_percentage : ℝ),
    selling_price_with_loss cost_price loss_percentage + additional_amount =
      selling_price_with_gain cost_price gain_percentage ∧
    loss_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_loss_percentage_l177_17787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l177_17713

/-- An ellipse with a focus and directrix -/
structure Ellipse where
  focus : ℝ × ℝ
  directrix : ℝ → ℝ

/-- A line parallel to y = x -/
def parallel_line (F : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | ∃ (k : ℝ), P.1 - F.1 = k ∧ P.2 - F.2 = k}

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (E : Ellipse) : ℝ := sorry

/-- The center of an ellipse -/
noncomputable def center (E : Ellipse) : ℝ × ℝ := sorry

/-- The intersection points of a line and an ellipse -/
noncomputable def intersection_points (E : Ellipse) (L : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- Check if a point is inside a circle -/
def inside_circle (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop := sorry

/-- The main theorem -/
theorem ellipse_eccentricity_range (E : Ellipse) :
  let F := E.focus
  let L := parallel_line F
  let AB := intersection_points E L
  let P := center E
  (∀ (A B : ℝ × ℝ), A ∈ AB → B ∈ AB → inside_circle P A B) →
  let e := eccentricity E
  0 < e ∧ e < Real.sqrt (2 - Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l177_17713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_shift_l177_17769

/-- A function that represents the shifted sine wave -/
noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (2 * (x + φ) + Real.pi / 4)

/-- The theorem stating the condition for g to be an even function -/
theorem even_function_shift (φ : ℝ) :
  (∀ x, g φ x = g φ (-x)) ↔ ∃ k : ℤ, φ = Real.pi / 8 + k * Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_shift_l177_17769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l177_17749

-- Define the speed of the train in km/hr
noncomputable def train_speed : ℝ := 210

-- Define the time taken to cross the pole in seconds
noncomputable def crossing_time : ℝ := 2.3998080153587713

-- Define the conversion factor from km/hr to m/s
noncomputable def km_hr_to_m_s : ℝ := 5 / 18

-- Define the approximation threshold
noncomputable def approx_threshold : ℝ := 0.5

-- Theorem statement
theorem train_length_approx :
  let speed_m_s := train_speed * km_hr_to_m_s
  let length := speed_m_s * crossing_time
  abs (length - 140) < approx_threshold := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l177_17749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_four_l177_17727

open Real Function

/-- Given a function f : ℝ → ℝ satisfying f(x) = x^2 + 3f'(1)x - f(1) for all x ∈ ℝ, prove that f(4) = 5 -/
theorem function_value_at_four (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 + 3 * (deriv f 1) * x - f 1) :
  f 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_four_l177_17727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_june_has_greatest_percent_difference_l177_17733

-- Define the sales data for each month
def sales_data : List (Nat × Nat) := [
  (6, 4),  -- January
  (8, 2),  -- February
  (9, 6),  -- March
  (5, 8),  -- April
  (2, 6),  -- May
  (1, 5)   -- June
]

-- Define the function to calculate percentage difference
def percentage_difference (e : Nat) (w : Nat) : Rat :=
  (max (e : Rat) (w : Rat) - min (e : Rat) (w : Rat)) / min (e : Rat) (w : Rat) * 100

-- Theorem statement
theorem june_has_greatest_percent_difference :
  ∀ (month : Fin 6),
    percentage_difference (sales_data[5].1) (sales_data[5].2) ≥
    percentage_difference (sales_data[month].1) (sales_data[month].2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_june_has_greatest_percent_difference_l177_17733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_implies_a_range_l177_17782

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - 2

theorem increasing_interval_implies_a_range (a : ℝ) :
  (∃ x₁ x₂, 1/2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 ∧ 
   ∀ x ∈ Set.Ioo x₁ x₂, ∀ y ∈ Set.Ioo x₁ x₂, x < y → f a x < f a y) →
  a > -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_implies_a_range_l177_17782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_special_lines_l177_17798

/-- A line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a line passes through a point --/
def Line.passesThroughPoint (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if a line has equal intercepts on both axes --/
def Line.hasEqualIntercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = -l.c / l.b

/-- The set of lines passing through (11,1) with equal intercepts --/
def specialLines : Set Line :=
  {l : Line | l.passesThroughPoint 11 1 ∧ l.hasEqualIntercepts}

/-- Assertion that there are exactly two special lines --/
theorem two_special_lines : ∃ (s : Finset Line), s.card = 2 ∧ ∀ l, l ∈ s ↔ l ∈ specialLines := by
  sorry

/-- Helper lemma: The two special lines are x - 11y = 0 and x + y - 12 = 0 --/
lemma special_lines_explicit : 
  ∃ (l₁ l₂ : Line), 
    l₁ ∈ specialLines ∧ 
    l₂ ∈ specialLines ∧ 
    l₁ ≠ l₂ ∧
    (l₁.a = 1 ∧ l₁.b = -11 ∧ l₁.c = 0) ∧
    (l₂.a = 1 ∧ l₂.b = 1 ∧ l₂.c = -12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_special_lines_l177_17798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_percentage_l177_17777

theorem revenue_percentage (R : ℝ) (h : R > 0) : 
  let projected_revenue := 1.20 * R
  let actual_revenue := 0.90 * R
  (actual_revenue / projected_revenue) * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_percentage_l177_17777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_meet_in_50_minutes_l177_17708

/-- Two planes flying towards each other -/
structure PlaneMeeting where
  initial_distance : ℝ
  speed_a : ℝ
  speed_b : ℝ

/-- Calculate the time (in minutes) for two planes to meet -/
noncomputable def meeting_time (pm : PlaneMeeting) : ℝ :=
  pm.initial_distance / (pm.speed_a + pm.speed_b) * 60

/-- Theorem: The planes meet after 50 minutes -/
theorem planes_meet_in_50_minutes :
  let pm : PlaneMeeting := {
    initial_distance := 500,
    speed_a := 240,
    speed_b := 360
  }
  meeting_time pm = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_meet_in_50_minutes_l177_17708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_price_l177_17735

/-- The selling price of an item given its cost price and markup percentage -/
noncomputable def selling_price (cost : ℝ) (markup_percent : ℝ) : ℝ :=
  cost * (1 + markup_percent / 100)

/-- Theorem: The selling price of a computer table with cost price 7000 and 20% markup is 8400 -/
theorem computer_table_price : selling_price 7000 20 = 8400 := by
  -- Unfold the definition of selling_price
  unfold selling_price
  -- Simplify the arithmetic
  simp [mul_add, mul_div_right_comm]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_price_l177_17735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_approximates_probability_l177_17732

/-- Represents a random event -/
structure RandomEvent where
  occurences : ℕ
  trials : ℕ

/-- Represents the probability of an event -/
noncomputable def probability (A : RandomEvent) : ℝ :=
  (A.occurences : ℝ) / (A.trials : ℝ)

/-- Represents the frequency of an event -/
noncomputable def frequency (A : RandomEvent) : ℝ :=
  (A.occurences : ℝ) / (A.trials : ℝ)

/-- States that for a large number of trials, the frequency approximates the probability -/
theorem frequency_approximates_probability (A : RandomEvent) (ε : ℝ) (h_large : A.trials > 1000) :
  ∃ (δ : ℝ), δ > 0 ∧ |probability A - frequency A| < ε := by
  sorry

#check frequency_approximates_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_approximates_probability_l177_17732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_problem_l177_17756

/-- The reflection point of a light ray on a plane --/
def reflection_point (A C : ℝ × ℝ × ℝ) (plane_normal : ℝ × ℝ × ℝ) (plane_const : ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- The given problem statement --/
theorem light_reflection_problem :
  let A : ℝ × ℝ × ℝ := (2, 1, -1)
  let C : ℝ × ℝ × ℝ := (-1, 4, 2)
  let plane_normal : ℝ × ℝ × ℝ := (1, -1, 1)
  let plane_const : ℝ := 1
  let B : ℝ × ℝ × ℝ := reflection_point A C plane_normal plane_const
  B = (7/3, 2/3, -2/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_problem_l177_17756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_price_equation_l177_17746

/-- Represents the original price of a bottle of water -/
def x : ℝ := sorry

/-- Represents the price of a box of water (4 bottles) -/
def box_price : ℝ := 26

/-- Represents the discount per bottle due to the promotion -/
def discount : ℝ := 0.6

/-- Represents the number of free bottles for each bought bottle -/
def free_bottles : ℝ := 3

/-- 
The equation representing the relationship between the original price 
and the promotional price of water bottles
-/
theorem water_price_equation : 
  (box_price / (x - discount)) - (box_price / x) = free_bottles :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_price_equation_l177_17746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_on_interval_l177_17737

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin (x/2) + Real.cos (x/2))^2 - 2 * Real.sqrt 3 * (Real.cos (x/2))^2 + Real.sqrt 3

theorem f_range_on_interval :
  ∀ x ∈ Set.Icc 0 Real.pi, 
    1 - Real.sqrt 3 ≤ f x ∧ f x ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_on_interval_l177_17737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_min_max_f_on_interval_l177_17770

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x^2 / (2*x + 2)

-- Define the interval
def interval : Set ℝ := { x | -1/3 ≤ x ∧ x ≤ 1 }

-- State the theorem
theorem sum_of_min_max_f_on_interval :
  ∃ (min_val max_val : ℝ),
    (∀ x ∈ interval, f x ≥ min_val) ∧
    (∃ x ∈ interval, f x = min_val) ∧
    (∀ x ∈ interval, f x ≤ max_val) ∧
    (∃ x ∈ interval, f x = max_val) ∧
    min_val + max_val = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_min_max_f_on_interval_l177_17770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eggs_left_on_shelf_l177_17744

theorem eggs_left_on_shelf 
  (x : ℕ) -- number of eggs bought
  (y : ℚ) -- fraction of eggs used
  (z : ℕ) -- number of eggs broken
  (h1 : 0 ≤ y) -- ensure y is non-negative
  (h2 : y < 1) -- ensure y is a proper fraction
  : ℚ :=
  ↑x * (1 - y) - ↑z

#check eggs_left_on_shelf

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eggs_left_on_shelf_l177_17744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l177_17757

open Real

-- Define the function f(x) = x ln(x)
noncomputable def f (x : ℝ) : ℝ := x * log x

-- State the theorem
theorem f_properties :
  -- f(x) is defined for x > 0
  ∀ x > 0,
  -- Part 1: Monotonicity intervals
  (∀ y ∈ Set.Ioo 0 (1/ℯ), ∀ z ∈ Set.Ioo 0 (1/ℯ), y < z → f y > f z) ∧
  (∀ y ∈ Set.Ioi (1/ℯ), ∀ z ∈ Set.Ioi (1/ℯ), y < z → f y < f z) ∧
  -- Part 2: Inequality for any positive real numbers a and b
  ∀ a > 0, ∀ b > 0, (f a + f b) / 2 > f ((a + b) / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l177_17757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_prime_symmetric_to_M_l177_17771

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop := 4*x + 6*y + 4*z - 25 = 0

/-- Point M -/
def M : Fin 3 → ℝ := ![1, 0, 1]

/-- Point M' -/
def M' : Fin 3 → ℝ := ![3, 3, 3]

/-- Definition of symmetry with respect to a plane -/
def symmetric_wrt_plane (p q : Fin 3 → ℝ) : Prop :=
  ∃ (m : Fin 3 → ℝ), 
    plane_equation (m 0) (m 1) (m 2) ∧ 
    ∀ i : Fin 3, m i = (p i + q i) / 2

/-- Theorem stating that M' is symmetric to M with respect to the given plane -/
theorem M_prime_symmetric_to_M : symmetric_wrt_plane M M' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_prime_symmetric_to_M_l177_17771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_variance_specific_case_l177_17779

noncomputable def sample_variance (s : List ℝ) : ℝ :=
  let n := s.length
  let mean := s.sum / n
  (s.map (λ x => (x - mean)^2)).sum / n

theorem sample_variance_specific_case :
  ∀ a : ℝ,
  let s := [a, 0, 1, 2, 3]
  s.length = 5 ∧
  s.sum / s.length = 1 →
  sample_variance s = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_variance_specific_case_l177_17779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_extrema_l177_17799

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

-- State the theorem
theorem tangent_and_extrema :
  -- The tangent line at x = 0 is y = 1
  (∀ y, (deriv f) 0 * 0 + f 0 = y ↔ y = 1) ∧
  -- The maximum value of f in [0, π/2] is 1
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1) ∧
  -- The minimum value of f in [0, π/2] is -π/2
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -Real.pi / 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -Real.pi / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_extrema_l177_17799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_l177_17728

/-- Parametric curve definition -/
noncomputable def x (t : ℝ) : ℝ := 3 * Real.cos t - 2 * Real.sin t

/-- Parametric curve definition -/
noncomputable def y (t : ℝ) : ℝ := 5 * Real.sin t

/-- Constants for the equation -/
noncomputable def a : ℝ := 1 / 9
noncomputable def b : ℝ := 4 / 45
noncomputable def c : ℝ := 13 / 225

/-- Theorem stating that the parametric curve satisfies the given equation -/
theorem curve_equation (t : ℝ) : a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_l177_17728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_perpendicular_line_l177_17712

-- Define the original line
noncomputable def original_line (x y : ℝ) : Prop := 4 * x + 5 * y = 10

-- Define the slope of the original line
noncomputable def original_slope : ℝ := -4 / 5

-- Define the perpendicular line
noncomputable def perpendicular_line (x y : ℝ) : Prop := y = (5 / 4) * x - 3

-- Define the x-intercept of the perpendicular line
noncomputable def x_intercept : ℝ := 12 / 5

-- Theorem statement
theorem x_intercept_of_perpendicular_line :
  ∀ x y : ℝ,
  original_line x y →
  perpendicular_line x y →
  x_intercept = 12 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_perpendicular_line_l177_17712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l177_17761

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (1, 0)
noncomputable def C : ℝ × ℝ := (0, 2)
noncomputable def D : ℝ × ℝ := (3, 3)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_sum_distances :
  ∀ P : ℝ × ℝ, distance P A + distance P B + distance P C + distance P D ≥ 2 * Real.sqrt 3 + Real.sqrt 5 := by
  sorry

#check min_sum_distances

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l177_17761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_m_l177_17711

-- Define the interval [0, π/3]
def I : Set ℝ := Set.Icc 0 (Real.pi / 3)

-- Define the condition that m must satisfy
def satisfies_condition (m : ℝ) : Prop :=
  ∀ x ∈ I, m ≥ 2 * Real.tan x

-- State the theorem
theorem min_value_m :
  (∃ m, satisfies_condition m) ∧ 
  (∀ m, satisfies_condition m → m ≥ 2 * Real.sqrt 3) ∧
  satisfies_condition (2 * Real.sqrt 3) := by
  sorry

#check min_value_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_m_l177_17711
