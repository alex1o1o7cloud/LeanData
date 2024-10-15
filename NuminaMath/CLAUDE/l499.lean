import Mathlib

namespace NUMINAMATH_CALUDE_dessert_preference_l499_49952

theorem dessert_preference (total : ℕ) (apple : ℕ) (chocolate : ℕ) (neither : ℕ)
  (h1 : total = 50)
  (h2 : apple = 22)
  (h3 : chocolate = 20)
  (h4 : neither = 15) :
  apple + chocolate - (total - neither) = 7 :=
by sorry

end NUMINAMATH_CALUDE_dessert_preference_l499_49952


namespace NUMINAMATH_CALUDE_number_problem_l499_49971

theorem number_problem (x : ℝ) : 0.65 * x = 0.8 * x - 21 → x = 140 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l499_49971


namespace NUMINAMATH_CALUDE_store_visits_per_week_l499_49939

/-- The number of store visits per week given the fort's completion status and collection period -/
theorem store_visits_per_week 
  (total_sticks : ℕ)
  (completion_percentage : ℚ)
  (collection_weeks : ℕ)
  (h1 : total_sticks = 400)
  (h2 : completion_percentage = 3/5)
  (h3 : collection_weeks = 80) :
  (completion_percentage * total_sticks) / collection_weeks = 3 := by
  sorry

end NUMINAMATH_CALUDE_store_visits_per_week_l499_49939


namespace NUMINAMATH_CALUDE_power_of_power_l499_49949

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l499_49949


namespace NUMINAMATH_CALUDE_equation_describes_parabola_l499_49954

-- Define the equation
def equation (x y : ℝ) : Prop := |y + 5| = Real.sqrt ((x - 2)^2 + y^2)

-- Define what it means for an equation to describe a parabola
def describes_parabola (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ a b c d : ℝ, a ≠ 0 ∧ ∀ x y : ℝ, eq x y ↔ y = a * x^2 + b * x + c ∨ x = a * y^2 + b * y + d

-- Theorem statement
theorem equation_describes_parabola : describes_parabola equation := by sorry

end NUMINAMATH_CALUDE_equation_describes_parabola_l499_49954


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l499_49908

def T (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    -n / 2
  else
    (n + 1) / 2

def U (n : ℕ) : ℤ := 2 * T n

theorem sum_of_specific_terms : U 13 + T 25 + U 40 = -13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l499_49908


namespace NUMINAMATH_CALUDE_exradii_sum_equals_p_squared_l499_49904

/-- Given a triangle with sides a, b, c, exradii ra, rb, rc, and semi-perimeter p,
    if the products of exradii satisfy certain conditions, then the sum of these
    products equals p^2. -/
theorem exradii_sum_equals_p_squared
  (a b c ra rb rc p : ℝ)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_ra : 0 < ra) (h_pos_rb : 0 < rb) (h_pos_rc : 0 < rc)
  (h_semi_perimeter : p = (a + b + c) / 2)
  (h_ra_rb : ra * rb = p * (p - c))
  (h_rb_rc : rb * rc = p * (p - a))
  (h_rc_ra : rc * ra = p * (p - b)) :
  ra * rb + rb * rc + rc * ra = p^2 := by
  sorry

end NUMINAMATH_CALUDE_exradii_sum_equals_p_squared_l499_49904


namespace NUMINAMATH_CALUDE_fourth_sphere_radius_l499_49992

/-- Given four spheres where each touches the other three, and three of them have radius R,
    the radius of the fourth sphere is R/3. -/
theorem fourth_sphere_radius (R : ℝ) (R_pos : R > 0) : ℝ :=
  let fourth_radius := R / 3
  fourth_radius

#check fourth_sphere_radius

end NUMINAMATH_CALUDE_fourth_sphere_radius_l499_49992


namespace NUMINAMATH_CALUDE_polygon_with_five_triangles_l499_49967

/-- A polygon with n sides -/
structure Polygon (n : ℕ) where
  -- We don't need to define the structure, just declare it

/-- The number of triangles formed when drawing diagonals from a single vertex -/
def triangles_from_vertex (n : ℕ) : ℕ := n - 2

/-- Theorem: If the diagonals from the same vertex of an n-sided polygon
    exactly divide the polygon into 5 triangles, then n = 7 -/
theorem polygon_with_five_triangles (n : ℕ) :
  triangles_from_vertex n = 5 → n = 7 := by
  sorry


end NUMINAMATH_CALUDE_polygon_with_five_triangles_l499_49967


namespace NUMINAMATH_CALUDE_normal_level_short_gallons_needed_after_evaporation_l499_49973

/-- Represents a water reservoir with given properties -/
structure Reservoir where
  current_level : ℝ
  normal_level : ℝ
  total_capacity : ℝ
  evaporation_rate : ℝ
  current_is_twice_normal : current_level = 2 * normal_level
  current_is_75_percent : current_level = 0.75 * total_capacity
  h_current_level : current_level = 30
  h_evaporation_rate : evaporation_rate = 0.1

/-- The normal level is 25 million gallons short of total capacity -/
theorem normal_level_short (r : Reservoir) :
  r.total_capacity - r.normal_level = 25 :=
sorry

/-- After evaporation, 13 million gallons are needed to reach total capacity -/
theorem gallons_needed_after_evaporation (r : Reservoir) :
  r.total_capacity - (r.current_level - r.evaporation_rate * r.current_level) = 13 :=
sorry

end NUMINAMATH_CALUDE_normal_level_short_gallons_needed_after_evaporation_l499_49973


namespace NUMINAMATH_CALUDE_total_savings_together_l499_49963

/-- Regular price of a window -/
def regular_price : ℝ := 120

/-- Calculate the number of windows to pay for given the number of windows bought -/
def windows_to_pay_for (n : ℕ) : ℕ :=
  n - n / 6

/-- Calculate the price with the special deal (free 6th window) -/
def price_with_deal (n : ℕ) : ℝ :=
  (windows_to_pay_for n : ℝ) * regular_price

/-- Apply the additional 5% discount for purchases over 10 windows -/
def apply_discount (price : ℝ) (n : ℕ) : ℝ :=
  if n > 10 then price * 0.95 else price

/-- Calculate the final price after all discounts -/
def final_price (n : ℕ) : ℝ :=
  apply_discount (price_with_deal n) n

/-- Nina's number of windows -/
def nina_windows : ℕ := 9

/-- Carl's number of windows -/
def carl_windows : ℕ := 11

/-- Theorem: The total savings when Nina and Carl buy windows together is $348 -/
theorem total_savings_together : 
  (nina_windows + carl_windows) * regular_price - final_price (nina_windows + carl_windows) = 348 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_together_l499_49963


namespace NUMINAMATH_CALUDE_helmet_store_theorem_l499_49914

structure HelmetStore where
  wholesale_price_A : ℕ
  wholesale_price_B : ℕ
  day1_sales_A : ℕ
  day1_sales_B : ℕ
  day1_total : ℕ
  day2_sales_A : ℕ
  day2_sales_B : ℕ
  day2_total : ℕ
  budget : ℕ
  total_helmets : ℕ
  profit_target : ℕ

def selling_prices (store : HelmetStore) : ℕ × ℕ :=
  -- Placeholder for the function to calculate selling prices
  (0, 0)

def can_achieve_profit (store : HelmetStore) (prices : ℕ × ℕ) : Prop :=
  -- Placeholder for the function to check if profit target can be achieved
  false

theorem helmet_store_theorem (store : HelmetStore) 
  (h1 : store.wholesale_price_A = 40)
  (h2 : store.wholesale_price_B = 30)
  (h3 : store.day1_sales_A = 10)
  (h4 : store.day1_sales_B = 15)
  (h5 : store.day1_total = 1150)
  (h6 : store.day2_sales_A = 6)
  (h7 : store.day2_sales_B = 12)
  (h8 : store.day2_total = 810)
  (h9 : store.budget = 3400)
  (h10 : store.total_helmets = 100)
  (h11 : store.profit_target = 1300) :
  let prices := selling_prices store
  prices = (55, 40) ∧ ¬(can_achieve_profit store prices) := by
  sorry

end NUMINAMATH_CALUDE_helmet_store_theorem_l499_49914


namespace NUMINAMATH_CALUDE_integral_of_derivative_scaled_l499_49929

theorem integral_of_derivative_scaled (f : ℝ → ℝ) (a b : ℝ) (hf : Differentiable ℝ f) (hab : a < b) :
  ∫ x in a..b, (deriv f (3 * x)) = (1 / 3) * (f (3 * b) - f (3 * a)) := by
  sorry

end NUMINAMATH_CALUDE_integral_of_derivative_scaled_l499_49929


namespace NUMINAMATH_CALUDE_f_properties_l499_49984

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x

theorem f_properties :
  (∀ x > 1, f x > 0) ∧
  (∀ x, 0 < x → x < 1 → f x < 0) ∧
  (Set.range f = Set.Ici (-1 / (2 * Real.exp 1))) ∧
  (∀ x > 0, f x ≥ x - 1) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l499_49984


namespace NUMINAMATH_CALUDE_max_B_at_125_l499_49940

def B (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.3 ^ k)

theorem max_B_at_125 :
  ∀ k : ℕ, k ≤ 500 → B 125 ≥ B k :=
by sorry

end NUMINAMATH_CALUDE_max_B_at_125_l499_49940


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l499_49938

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x - 3 ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*x - 3 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l499_49938


namespace NUMINAMATH_CALUDE_horse_speed_l499_49991

/-- Given a square field with area 1600 km^2 and a horse that takes 10 hours to run around it,
    the speed of the horse is 16 km/h. -/
theorem horse_speed (field_area : ℝ) (run_time : ℝ) (horse_speed : ℝ) : 
  field_area = 1600 → run_time = 10 → horse_speed = (4 * Real.sqrt field_area) / run_time → 
  horse_speed = 16 := by sorry

end NUMINAMATH_CALUDE_horse_speed_l499_49991


namespace NUMINAMATH_CALUDE_area_perimeter_ratio_l499_49913

/-- The side length of the square -/
def square_side : ℝ := 5

/-- The side length of the equilateral triangle -/
def triangle_side : ℝ := 6

/-- The area of a square given its side length -/
def square_area (side : ℝ) : ℝ := side * side

/-- The perimeter of an equilateral triangle given its side length -/
def equilateral_triangle_perimeter (side : ℝ) : ℝ := 3 * side

/-- Theorem stating the ratio of the square's area to the triangle's perimeter -/
theorem area_perimeter_ratio :
  (square_area square_side) / (equilateral_triangle_perimeter triangle_side) = 25 / 18 := by
  sorry


end NUMINAMATH_CALUDE_area_perimeter_ratio_l499_49913


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_l499_49907

/-- Represents the value of one trillion in scientific notation -/
def trillion : ℝ := 10^12

/-- The gross domestic product in trillion yuan -/
def gdp : ℝ := 114

/-- The gross domestic product expressed in scientific notation -/
def gdp_scientific : ℝ := 1.14 * 10^14

theorem gdp_scientific_notation :
  gdp * trillion = gdp_scientific := by sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_l499_49907


namespace NUMINAMATH_CALUDE_tom_car_lease_cost_l499_49998

/-- Calculates the total yearly cost for Tom's car lease -/
theorem tom_car_lease_cost :
  let miles_per_week : ℕ := 4 * 50 + 3 * 100
  let cost_per_mile : ℚ := 1 / 10
  let weekly_fee : ℕ := 100
  let weeks_per_year : ℕ := 52
  (miles_per_week : ℚ) * cost_per_mile * weeks_per_year + (weekly_fee : ℚ) * weeks_per_year = 7800 := by
  sorry

end NUMINAMATH_CALUDE_tom_car_lease_cost_l499_49998


namespace NUMINAMATH_CALUDE_cornelias_current_age_l499_49962

/-- Proves Cornelia's current age given the conditions of the problem -/
theorem cornelias_current_age (kilee_current_age : ℕ) (cornelia_future_age kilee_future_age : ℕ) :
  kilee_current_age = 20 →
  kilee_future_age = kilee_current_age + 10 →
  cornelia_future_age = 3 * kilee_future_age →
  cornelia_future_age - 10 = 80 :=
by sorry

end NUMINAMATH_CALUDE_cornelias_current_age_l499_49962


namespace NUMINAMATH_CALUDE_MON_is_right_angle_l499_49970

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define point E
def E : ℝ × ℝ := (2, 2)

-- Define the line l
def line_l (x y : ℝ) : Prop := ∃ k, y = k*(x - 2)

-- Define that l passes through (2,0)
axiom l_through_2_0 : line_l 2 0

-- Define points A and B on the parabola and line l
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
axiom A_on_parabola : parabola A.1 A.2
axiom B_on_parabola : parabola B.1 B.2
axiom A_on_l : line_l A.1 A.2
axiom B_on_l : line_l B.1 B.2
axiom A_not_E : A ≠ E
axiom B_not_E : B ≠ E

-- Define points M and N
def M : ℝ × ℝ := sorry
def N : ℝ × ℝ := sorry
axiom M_on_EA : ∃ t, M = (1 - t) • E + t • A
axiom N_on_EB : ∃ t, N = (1 - t) • E + t • B
axiom M_on_x_neg2 : M.1 = -2
axiom N_on_x_neg2 : N.1 = -2

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem to prove
theorem MON_is_right_angle : 
  let OM := M - O
  let ON := N - O
  OM.1 * ON.1 + OM.2 * ON.2 = 0 := by sorry

end NUMINAMATH_CALUDE_MON_is_right_angle_l499_49970


namespace NUMINAMATH_CALUDE_range_of_x_for_proposition_l499_49947

theorem range_of_x_for_proposition (x : ℝ) : 
  (∃ a : ℝ, a ∈ Set.Icc 1 3 ∧ a * x^2 + (a - 2) * x - 2 > 0) ↔ 
  x < -1 ∨ x > 2/3 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_for_proposition_l499_49947


namespace NUMINAMATH_CALUDE_rectangle_area_l499_49955

theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 2 * w
  let square_side := 1
  let rectangle_perimeter := 2 * l + 2 * w
  let square_perimeter := 4 * square_side
  rectangle_perimeter = square_perimeter → l * w = 8 / 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l499_49955


namespace NUMINAMATH_CALUDE_calculation_proof_l499_49910

theorem calculation_proof : 2⁻¹ + Real.sqrt 16 - (3 - Real.sqrt 3)^0 + |Real.sqrt 2 - 1/2| = 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l499_49910


namespace NUMINAMATH_CALUDE_win_sector_area_l499_49997

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 3/8) :
  p * π * r^2 = 24 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l499_49997


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_4_pow_17_minus_2_pow_29_l499_49975

theorem greatest_prime_factor_of_4_pow_17_minus_2_pow_29 : 
  ∃ (p : ℕ), p.Prime ∧ p = 31 ∧ 
  (∀ q : ℕ, q.Prime → q ∣ (4^17 - 2^29) → q ≤ p) :=
sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_4_pow_17_minus_2_pow_29_l499_49975


namespace NUMINAMATH_CALUDE_root_sum_eighth_power_l499_49937

theorem root_sum_eighth_power (r s : ℝ) : 
  (r^2 - 2*r*Real.sqrt 6 + 3 = 0) →
  (s^2 - 2*s*Real.sqrt 6 + 3 = 0) →
  r^8 + s^8 = 93474 := by
sorry

end NUMINAMATH_CALUDE_root_sum_eighth_power_l499_49937


namespace NUMINAMATH_CALUDE_boys_average_age_l499_49987

/-- Proves that the average age of boys is 12 years given the school statistics -/
theorem boys_average_age (total_students : ℕ) (girls : ℕ) (girls_avg_age : ℝ) (school_avg_age : ℝ) :
  total_students = 652 →
  girls = 163 →
  girls_avg_age = 11 →
  school_avg_age = 11.75 →
  let boys := total_students - girls
  let boys_total_age := school_avg_age * total_students - girls_avg_age * girls
  boys_total_age / boys = 12 := by
sorry


end NUMINAMATH_CALUDE_boys_average_age_l499_49987


namespace NUMINAMATH_CALUDE_multiply_add_theorem_l499_49972

theorem multiply_add_theorem : 15 * 30 + 45 * 15 + 90 = 1215 := by
  sorry

end NUMINAMATH_CALUDE_multiply_add_theorem_l499_49972


namespace NUMINAMATH_CALUDE_group_trip_cost_l499_49988

/-- The total cost for a group trip, given the number of people and cost per person. -/
def total_cost (num_people : ℕ) (cost_per_person : ℕ) : ℕ :=
  num_people * cost_per_person

/-- Proof that the total cost for 15 people at $900 each is $13,500. -/
theorem group_trip_cost :
  total_cost 15 900 = 13500 := by
  sorry

end NUMINAMATH_CALUDE_group_trip_cost_l499_49988


namespace NUMINAMATH_CALUDE_parallelepiped_properties_l499_49999

/-- A rectangular parallelepiped with an inscribed sphere -/
structure Parallelepiped :=
  (k : ℝ)  -- Ratio of parallelepiped volume to sphere volume
  (h : k > 0)  -- k is positive

/-- Theorem about the angles and permissible values of k for a parallelepiped with an inscribed sphere -/
theorem parallelepiped_properties (p : Parallelepiped) :
  let α := Real.arcsin (6 / (Real.pi * p.k))
  ∃ (angle1 angle2 : ℝ),
    (angle1 = α ∧ angle2 = Real.pi - α) ∧  -- Angles at the base
    p.k ≥ 6 / Real.pi :=  -- Permissible values of k
by sorry

end NUMINAMATH_CALUDE_parallelepiped_properties_l499_49999


namespace NUMINAMATH_CALUDE_beth_crayon_packs_l499_49923

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 10

/-- The number of extra crayons not in packs -/
def extra_crayons : ℕ := 6

/-- The total number of crayons Beth has -/
def total_crayons : ℕ := 46

/-- The number of packs of crayons Beth has -/
def num_packs : ℕ := (total_crayons - extra_crayons) / crayons_per_pack

theorem beth_crayon_packs :
  num_packs = 4 :=
sorry

end NUMINAMATH_CALUDE_beth_crayon_packs_l499_49923


namespace NUMINAMATH_CALUDE_moon_speed_conversion_l499_49936

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_per_second : ℝ := 1.02

/-- Converts speed from kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * seconds_per_hour

theorem moon_speed_conversion :
  km_per_second_to_km_per_hour moon_speed_km_per_second = 3672 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_conversion_l499_49936


namespace NUMINAMATH_CALUDE_log_equation_solution_l499_49985

theorem log_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ 2 * Real.log x = Real.log (x + 12) :=
by
  -- The unique solution is x = 4
  use 4
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l499_49985


namespace NUMINAMATH_CALUDE_sum_parity_eq_parity_of_M_l499_49982

/-- Represents the parity of a number -/
inductive Parity
  | Even
  | Odd

/-- The sum of N even numbers and M odd numbers -/
def sum_parity (N M : ℕ) : Parity :=
  match M % 2 with
  | 0 => Parity.Even
  | _ => Parity.Odd

/-- The parity of a natural number -/
def parity (n : ℕ) : Parity :=
  match n % 2 with
  | 0 => Parity.Even
  | _ => Parity.Odd

/-- Theorem: The parity of the sum of N even numbers and M odd numbers
    is equal to the parity of M -/
theorem sum_parity_eq_parity_of_M (N M : ℕ) :
  sum_parity N M = parity M := by sorry

end NUMINAMATH_CALUDE_sum_parity_eq_parity_of_M_l499_49982


namespace NUMINAMATH_CALUDE_inverse_proportion_ordering_l499_49995

/-- Proves that for points on an inverse proportion function, 
    if x₁ < 0 < x₂, then y₁ < y₂ -/
theorem inverse_proportion_ordering (x₁ x₂ y₁ y₂ : ℝ) : 
  x₁ < 0 → 0 < x₂ → y₁ = 6 / x₁ → y₂ = 6 / x₂ → y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ordering_l499_49995


namespace NUMINAMATH_CALUDE_range_of_2cos_squared_l499_49961

theorem range_of_2cos_squared (x : ℝ) : 0 ≤ 2 * (Real.cos x)^2 ∧ 2 * (Real.cos x)^2 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_2cos_squared_l499_49961


namespace NUMINAMATH_CALUDE_mrs_wonderful_class_size_l499_49901

theorem mrs_wonderful_class_size :
  ∀ (girls boys jelly_beans_given : ℕ),
  girls + boys = 28 →
  boys = girls + 2 →
  jelly_beans_given = girls * girls + boys * boys →
  jelly_beans_given = 400 - 6 :=
by
  sorry

end NUMINAMATH_CALUDE_mrs_wonderful_class_size_l499_49901


namespace NUMINAMATH_CALUDE_investment_result_unique_initial_investment_l499_49934

/-- Represents the growth of an investment over time with compound interest and additional investments. -/
def investment_growth (initial_investment : ℝ) : ℝ :=
  let after_compound := initial_investment * (1 + 0.20)^3
  let after_triple := after_compound * 3
  after_triple * (1 + 0.15)

/-- Theorem stating that an initial investment of $10,000 results in $59,616 after the given growth pattern. -/
theorem investment_result : investment_growth 10000 = 59616 := by
  sorry

/-- Theorem proving the uniqueness of the initial investment that results in $59,616. -/
theorem unique_initial_investment (x : ℝ) :
  investment_growth x = 59616 → x = 10000 := by
  sorry

end NUMINAMATH_CALUDE_investment_result_unique_initial_investment_l499_49934


namespace NUMINAMATH_CALUDE_bottom_level_legos_l499_49996

/-- Represents a 3-level pyramid with decreasing lego sides -/
structure LegoPyramid where
  bottom : ℕ  -- Number of legos per side on the bottom level
  mid : ℕ     -- Number of legos per side on the middle level
  top : ℕ     -- Number of legos per side on the top level

/-- Calculates the total number of legos in the pyramid -/
def totalLegos (p : LegoPyramid) : ℕ :=
  p.bottom ^ 2 + p.mid ^ 2 + p.top ^ 2

/-- Theorem: The bottom level of a 3-level pyramid with 110 total legos has 7 legos per side -/
theorem bottom_level_legos :
  ∃ (p : LegoPyramid),
    p.mid = p.bottom - 1 ∧
    p.top = p.bottom - 2 ∧
    totalLegos p = 110 ∧
    p.bottom = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_bottom_level_legos_l499_49996


namespace NUMINAMATH_CALUDE_john_computer_cost_l499_49930

/-- The total cost of a computer after upgrades -/
def total_cost (initial_cost old_video_card old_memory old_processor new_video_card new_memory new_processor : ℕ) : ℕ :=
  initial_cost + new_video_card + new_memory + new_processor - old_video_card - old_memory - old_processor

/-- Theorem: The total cost of John's computer after upgrades is $2500 -/
theorem john_computer_cost : 
  total_cost 2000 300 100 150 500 200 350 = 2500 := by
  sorry

end NUMINAMATH_CALUDE_john_computer_cost_l499_49930


namespace NUMINAMATH_CALUDE_no_divisibility_by_1955_l499_49974

theorem no_divisibility_by_1955 : ∀ n : ℤ, ¬(1955 ∣ (n^2 + n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_no_divisibility_by_1955_l499_49974


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_eight_sqrt_two_l499_49912

theorem sqrt_sum_equals_eight_sqrt_two : 
  Real.sqrt ((5 - 4 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 4 * Real.sqrt 2) ^ 2) = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_eight_sqrt_two_l499_49912


namespace NUMINAMATH_CALUDE_seating_theorem_l499_49922

/-- Number of seats in a row -/
def total_seats : ℕ := 7

/-- Number of people to be seated -/
def people_to_seat : ℕ := 4

/-- Number of adjacent empty seats required -/
def adjacent_empty_seats : ℕ := 2

/-- Function to calculate the number of seating arrangements -/
def seating_arrangements (total_seats : ℕ) (people_to_seat : ℕ) (adjacent_empty_seats : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of seating arrangements -/
theorem seating_theorem :
  seating_arrangements total_seats people_to_seat adjacent_empty_seats = 336 :=
sorry

end NUMINAMATH_CALUDE_seating_theorem_l499_49922


namespace NUMINAMATH_CALUDE_expected_distinct_colors_value_l499_49900

/-- The number of balls in the bag -/
def n : ℕ := 10

/-- The number of times a ball is picked -/
def k : ℕ := 4

/-- The probability of not picking a specific color in one draw -/
def p : ℚ := 9/10

/-- The expected number of distinct colors -/
def expected_distinct_colors : ℚ := n * (1 - p^k)

theorem expected_distinct_colors_value :
  expected_distinct_colors = 3439/1000 := by sorry

end NUMINAMATH_CALUDE_expected_distinct_colors_value_l499_49900


namespace NUMINAMATH_CALUDE_tim_balloon_count_l499_49909

/-- Calculates the number of Tim's balloons given Dan's balloons and the multiplier -/
def tims_balloons (dans_balloons : ℕ) (multiplier : ℕ) : ℕ :=
  dans_balloons * multiplier

/-- Theorem: Given Dan has 59 violet balloons and Tim has 11 times more,
    Tim has 649 violet balloons -/
theorem tim_balloon_count :
  tims_balloons 59 11 = 649 := by
  sorry

end NUMINAMATH_CALUDE_tim_balloon_count_l499_49909


namespace NUMINAMATH_CALUDE_fishbowl_count_l499_49981

theorem fishbowl_count (fish_per_bowl : ℕ) (total_fish : ℕ) (h1 : fish_per_bowl = 23) (h2 : total_fish = 6003) :
  total_fish / fish_per_bowl = 261 := by
  sorry

end NUMINAMATH_CALUDE_fishbowl_count_l499_49981


namespace NUMINAMATH_CALUDE_parallelogram_opposite_sides_l499_49989

/-- A parallelogram in a 2D Cartesian plane -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def diagonalIntersection (p : Parallelogram) : ℝ × ℝ := (0, 1)

def lineAB : LineEquation := { a := 1, b := -2, c := -2 }

theorem parallelogram_opposite_sides (p : Parallelogram) 
  (h1 : diagonalIntersection p = (0, 1))
  (h2 : lineAB = { a := 1, b := -2, c := -2 }) :
  ∃ (lineCD : LineEquation), lineCD = { a := 1, b := -2, c := 6 } := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_opposite_sides_l499_49989


namespace NUMINAMATH_CALUDE_tangent_and_perpendicular_l499_49920

-- Define the curve f(x)
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 1

-- Define the line perpendicular to the given line
def perp_line (x y : ℝ) : Prop := 2*x - 6*y + 1 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 3*x + y + 2 = 0

-- Define the theorem
theorem tangent_and_perpendicular :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the curve
    f x₀ = y₀ ∧
    -- The tangent line passes through (x₀, y₀)
    tangent_line x₀ y₀ ∧
    -- The tangent line is perpendicular to the given line
    (∀ (x y : ℝ), perp_line x y → 
      (y - y₀) = -(1/3) * (x - x₀)) ∧
    -- The slope of the tangent line at (x₀, y₀) is the derivative of f at x₀
    (3*x₀^2 + 6*x₀ = -3) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_perpendicular_l499_49920


namespace NUMINAMATH_CALUDE_max_min_product_l499_49957

theorem max_min_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (sum_eq : a + b + c = 10) (prod_sum_eq : a * b + b * c + c * a = 25) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 25 / 9 ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    a' + b' + c' = 10 ∧ a' * b' + b' * c' + c' * a' = 25 ∧
    min (a' * b') (min (b' * c') (c' * a')) = 25 / 9 :=
by sorry

end NUMINAMATH_CALUDE_max_min_product_l499_49957


namespace NUMINAMATH_CALUDE_large_planks_count_l499_49942

theorem large_planks_count (nails_per_plank : ℕ) (additional_nails : ℕ) (total_nails : ℕ) :
  nails_per_plank = 17 →
  additional_nails = 8 →
  total_nails = 229 →
  ∃ (x : ℕ), x * nails_per_plank + additional_nails = total_nails ∧ x = 13 :=
by sorry

end NUMINAMATH_CALUDE_large_planks_count_l499_49942


namespace NUMINAMATH_CALUDE_brick_weight_l499_49916

theorem brick_weight :
  ∀ x : ℝ, x = 2 + x / 2 → x = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_brick_weight_l499_49916


namespace NUMINAMATH_CALUDE_chad_video_games_earnings_l499_49921

/-- Chad's earnings and savings problem -/
theorem chad_video_games_earnings
  (savings_rate : ℚ)
  (mowing_earnings : ℚ)
  (birthday_earnings : ℚ)
  (odd_jobs_earnings : ℚ)
  (total_savings : ℚ)
  (h1 : savings_rate = 40 / 100)
  (h2 : mowing_earnings = 600)
  (h3 : birthday_earnings = 250)
  (h4 : odd_jobs_earnings = 150)
  (h5 : total_savings = 460) :
  let total_earnings := total_savings / savings_rate
  let known_earnings := mowing_earnings + birthday_earnings + odd_jobs_earnings
  total_earnings - known_earnings = 150 := by
sorry

end NUMINAMATH_CALUDE_chad_video_games_earnings_l499_49921


namespace NUMINAMATH_CALUDE_evenPerfectSquareFactorsCount_l499_49941

/-- The number of even perfect square factors of 2^6 * 7^12 * 3^2 -/
def evenPerfectSquareFactors : ℕ :=
  let n : ℕ := 2^6 * 7^12 * 3^2
  -- Count of valid combinations for exponents a, b, c
  let aCount : ℕ := 3  -- a can be 2, 4, or 6
  let bCount : ℕ := 7  -- b can be 0, 2, 4, 6, 8, 10, 12
  let cCount : ℕ := 2  -- c can be 0 or 2
  aCount * bCount * cCount

/-- Theorem: The number of even perfect square factors of 2^6 * 7^12 * 3^2 is 42 -/
theorem evenPerfectSquareFactorsCount : evenPerfectSquareFactors = 42 := by
  sorry

end NUMINAMATH_CALUDE_evenPerfectSquareFactorsCount_l499_49941


namespace NUMINAMATH_CALUDE_work_completion_time_proportional_aarti_triple_work_time_l499_49964

/-- If a person can complete a piece of work in a given number of days,
    then the time required to complete a multiple of that work is proportional. -/
theorem work_completion_time_proportional
  (days_for_single_work : ℕ) (work_multiple : ℕ) :
  let days_for_multiple_work := days_for_single_work * work_multiple
  days_for_multiple_work = days_for_single_work * work_multiple :=
by sorry

/-- Aarti's work completion time for triple work -/
theorem aarti_triple_work_time :
  let days_for_single_work := 6
  let work_multiple := 3
  let days_for_triple_work := days_for_single_work * work_multiple
  days_for_triple_work = 18 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_proportional_aarti_triple_work_time_l499_49964


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l499_49943

theorem perpendicular_vectors_x_value (x : ℝ) : 
  let a : Fin 3 → ℝ := ![2, -1, x]
  let b : Fin 3 → ℝ := ![3, 2, -1]
  (∀ i : Fin 3, (a i) * (b i) = 0) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l499_49943


namespace NUMINAMATH_CALUDE_at_most_one_obtuse_angle_l499_49968

-- Define a triangle
def Triangle : Type := Unit

-- Define an angle in a triangle
def Angle (t : Triangle) : Type := Unit

-- Define if an angle is obtuse
def IsObtuse (t : Triangle) (a : Angle t) : Prop := sorry

-- State the theorem
theorem at_most_one_obtuse_angle (t : Triangle) :
  ¬∃ (a b : Angle t), a ≠ b ∧ IsObtuse t a ∧ IsObtuse t b :=
sorry

end NUMINAMATH_CALUDE_at_most_one_obtuse_angle_l499_49968


namespace NUMINAMATH_CALUDE_rose_painting_time_l499_49979

/-- Time to paint a lily in minutes -/
def lily_time : ℕ := 5

/-- Time to paint an orchid in minutes -/
def orchid_time : ℕ := 3

/-- Time to paint a vine in minutes -/
def vine_time : ℕ := 2

/-- Total time spent painting in minutes -/
def total_time : ℕ := 213

/-- Number of lilies painted -/
def lily_count : ℕ := 17

/-- Number of roses painted -/
def rose_count : ℕ := 10

/-- Number of orchids painted -/
def orchid_count : ℕ := 6

/-- Number of vines painted -/
def vine_count : ℕ := 20

/-- Time to paint a rose in minutes -/
def rose_time : ℕ := 7

theorem rose_painting_time : 
  lily_count * lily_time + rose_count * rose_time + orchid_count * orchid_time + vine_count * vine_time = total_time := by
  sorry

end NUMINAMATH_CALUDE_rose_painting_time_l499_49979


namespace NUMINAMATH_CALUDE_inequality_implication_l499_49928

theorem inequality_implication (a b : ℝ) : -2 * a + 1 < -2 * b + 1 → a > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l499_49928


namespace NUMINAMATH_CALUDE_fraction_equality_l499_49948

theorem fraction_equality : (10^9 + 10^6) / (3 * 10^4) = 100100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l499_49948


namespace NUMINAMATH_CALUDE_cylinder_heights_sum_l499_49978

theorem cylinder_heights_sum (p₁ p₂ p₃ : ℝ) 
  (h₁ : p₁ = 6) 
  (h₂ : p₂ = 9) 
  (h₃ : p₃ = 11) : 
  p₁ + p₂ + p₃ = 26 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_heights_sum_l499_49978


namespace NUMINAMATH_CALUDE_socks_thrown_away_l499_49924

theorem socks_thrown_away (initial_socks : ℕ) (new_socks : ℕ) (final_socks : ℕ) : 
  initial_socks = 33 → new_socks = 13 → final_socks = 27 → 
  initial_socks - (final_socks - new_socks) = 19 := by
sorry

end NUMINAMATH_CALUDE_socks_thrown_away_l499_49924


namespace NUMINAMATH_CALUDE_complex_magnitude_l499_49917

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = 1 - Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l499_49917


namespace NUMINAMATH_CALUDE_lateral_angle_cosine_l499_49918

/-- A regular triangular pyramid with an inscribed sphere -/
structure RegularPyramid where
  -- The ratio of the intersection point on an edge
  intersectionRatio : ℝ
  -- Assumption that the pyramid is regular and has an inscribed sphere
  regular : Bool
  hasInscribedSphere : Bool

/-- The angle between a lateral face and the base plane of the pyramid -/
def lateralAngle (p : RegularPyramid) : ℝ := sorry

/-- Main theorem: The cosine of the lateral angle is 7/10 -/
theorem lateral_angle_cosine (p : RegularPyramid) 
  (h1 : p.intersectionRatio = 1.55)
  (h2 : p.regular = true)
  (h3 : p.hasInscribedSphere = true) : 
  Real.cos (lateralAngle p) = 7/10 := by sorry

end NUMINAMATH_CALUDE_lateral_angle_cosine_l499_49918


namespace NUMINAMATH_CALUDE_problem_solution_l499_49946

theorem problem_solution (x y : ℝ) : 
  (0.5 * x = 0.05 * 500 - 20) ∧ 
  (0.3 * y = 0.25 * x + 10) → 
  (x = 10 ∧ y = 125/3) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l499_49946


namespace NUMINAMATH_CALUDE_draw_with_replacement_l499_49911

-- Define the number of balls in the bin
def num_balls : ℕ := 15

-- Define the number of draws
def num_draws : ℕ := 4

-- Define the function to calculate the number of ways to draw balls
def ways_to_draw (n : ℕ) (k : ℕ) : ℕ := n ^ k

-- Theorem statement
theorem draw_with_replacement :
  ways_to_draw num_balls num_draws = 50625 := by
  sorry

end NUMINAMATH_CALUDE_draw_with_replacement_l499_49911


namespace NUMINAMATH_CALUDE_fraction_ordering_l499_49903

theorem fraction_ordering : 6/29 < 8/25 ∧ 8/25 < 10/31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l499_49903


namespace NUMINAMATH_CALUDE_upper_side_length_l499_49986

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  lower_side : ℝ
  upper_side : ℝ
  height : ℝ
  area : ℝ
  upper_shorter : upper_side = lower_side - 6
  height_value : height = 8
  area_value : area = 72
  area_formula : area = (lower_side + upper_side) / 2 * height

/-- Theorem: The length of the upper side of the trapezoid is 6 cm -/
theorem upper_side_length (t : Trapezoid) : t.upper_side = 6 := by
  sorry

end NUMINAMATH_CALUDE_upper_side_length_l499_49986


namespace NUMINAMATH_CALUDE_product_minus_third_lower_bound_l499_49905

theorem product_minus_third_lower_bound 
  (x y z : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0) 
  (a : ℝ) 
  (h1 : x * y - z = a) 
  (h2 : y * z - x = a) 
  (h3 : z * x - y = a) : 
  a ≥ -1/4 := by
sorry

end NUMINAMATH_CALUDE_product_minus_third_lower_bound_l499_49905


namespace NUMINAMATH_CALUDE_mapping_not_necessarily_injective_l499_49976

-- Define sets A and B
variable (A B : Type)

-- Define a mapping from A to B
variable (f : A → B)

-- Theorem stating that it's possible for two different elements in A to have the same image in B
theorem mapping_not_necessarily_injective :
  ∃ (x y : A), x ≠ y ∧ f x = f y :=
sorry

end NUMINAMATH_CALUDE_mapping_not_necessarily_injective_l499_49976


namespace NUMINAMATH_CALUDE_platform_length_l499_49926

/-- Given a train of length 300 meters that takes 36 seconds to cross a platform
    and 18 seconds to cross a signal pole, the length of the platform is 300 meters. -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 36)
  (h3 : pole_crossing_time = 18) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 300 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l499_49926


namespace NUMINAMATH_CALUDE_triangle_area_l499_49966

/-- The area of the triangle formed by the intersection of two lines and the y-axis --/
theorem triangle_area (line1 line2 : ℝ → ℝ) : 
  line1 = (λ x => 3 * x - 6) →
  line2 = (λ x => -4 * x + 24) →
  let x_intersect := (30 : ℝ) / 7
  let y_intersect := (48 : ℝ) / 7
  let base := 30
  let height := x_intersect
  (1 / 2 : ℝ) * base * height = 450 / 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l499_49966


namespace NUMINAMATH_CALUDE_regression_line_change_l499_49983

/-- Represents a linear regression equation of the form y = a + bx -/
structure RegressionLine where
  a : ℝ
  b : ℝ

/-- Calculates the change in y when x increases by 1 unit -/
def change_in_y (line : RegressionLine) : ℝ := -line.b

/-- Theorem: For the given regression line, when x increases by 1 unit, y decreases by 1.5 units -/
theorem regression_line_change (line : RegressionLine) 
  (h1 : line.a = 2) 
  (h2 : line.b = -1.5) : 
  change_in_y line = -1.5 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_change_l499_49983


namespace NUMINAMATH_CALUDE_reflection_of_P_across_x_axis_l499_49906

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflection of a point across the x-axis -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- The original point P -/
def P : Point :=
  { x := -2, y := 3 }

theorem reflection_of_P_across_x_axis :
  reflect_x P = { x := -2, y := -3 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_P_across_x_axis_l499_49906


namespace NUMINAMATH_CALUDE_ones_digit_8_pow_32_l499_49953

/-- The ones digit of a natural number -/
def ones_digit (n : ℕ) : ℕ := n % 10

/-- The ones digit of 8^n for any natural number n -/
def ones_digit_8_pow (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 8
  | 2 => 4
  | 3 => 2
  | _ => 0  -- This case is unreachable, but needed for exhaustiveness

theorem ones_digit_8_pow_32 :
  ones_digit (8^32) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_8_pow_32_l499_49953


namespace NUMINAMATH_CALUDE_two_color_draw_count_l499_49935

def total_balls : ℕ := 6
def red_balls : ℕ := 2
def white_balls : ℕ := 3
def blue_balls : ℕ := 1
def draw_count : ℕ := 3

def ways_two_colors : ℕ := 13

theorem two_color_draw_count :
  ways_two_colors = (total_balls.choose draw_count) - 
    (red_balls * white_balls * blue_balls) - 
    (if white_balls ≥ draw_count then 1 else 0) :=
by sorry

end NUMINAMATH_CALUDE_two_color_draw_count_l499_49935


namespace NUMINAMATH_CALUDE_sum_of_fractions_zero_l499_49951

theorem sum_of_fractions_zero (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h_sum : a + b + c = d) : 
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_zero_l499_49951


namespace NUMINAMATH_CALUDE_number_problem_l499_49959

theorem number_problem (N : ℝ) : 
  (1/8 : ℝ) * (3/5 : ℝ) * (4/7 : ℝ) * (5/11 : ℝ) * N - (1/9 : ℝ) * (2/3 : ℝ) * (3/4 : ℝ) * (5/8 : ℝ) * N = 30 → 
  (75/100 : ℝ) * N = -1476 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l499_49959


namespace NUMINAMATH_CALUDE_batsman_average_after_31st_inning_l499_49932

/-- Represents a batsman's cricket statistics -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : ℕ) : ℚ :=
  (b.totalRuns + runsScored : ℚ) / (b.innings + 1 : ℚ)

theorem batsman_average_after_31st_inning 
  (b : Batsman)
  (h1 : b.innings = 30)
  (h2 : newAverage b 105 = b.average + 3) :
  newAverage b 105 = 15 := by
  sorry

#check batsman_average_after_31st_inning

end NUMINAMATH_CALUDE_batsman_average_after_31st_inning_l499_49932


namespace NUMINAMATH_CALUDE_work_completion_time_l499_49990

-- Define the work completion time for Person A
def person_a_time : ℝ := 24

-- Define the combined work completion time for Person A and Person B
def combined_time : ℝ := 15

-- Define the work completion time for Person B
def person_b_time : ℝ := 40

-- Theorem statement
theorem work_completion_time :
  (1 / person_a_time + 1 / person_b_time = 1 / combined_time) := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l499_49990


namespace NUMINAMATH_CALUDE_dishonest_dealer_profit_l499_49950

/-- The profit percentage of a dishonest dealer who uses 800 grams instead of 1000 grams per kg -/
theorem dishonest_dealer_profit (actual_weight : ℕ) (claimed_weight : ℕ) : 
  actual_weight = 800 ∧ claimed_weight = 1000 → 
  (claimed_weight - actual_weight : ℚ) / claimed_weight * 100 = 20 := by
  sorry

#check dishonest_dealer_profit

end NUMINAMATH_CALUDE_dishonest_dealer_profit_l499_49950


namespace NUMINAMATH_CALUDE_tobias_driveways_shoveled_tobias_driveways_shoveled_proof_l499_49927

theorem tobias_driveways_shoveled : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun shoe_cost saving_months allowance lawn_charge shovel_charge wage change hours_worked lawns_mowed driveways_shoveled =>
    shoe_cost = 95 ∧
    saving_months = 3 ∧
    allowance = 5 ∧
    lawn_charge = 15 ∧
    shovel_charge = 7 ∧
    wage = 8 ∧
    change = 15 ∧
    hours_worked = 10 ∧
    lawns_mowed = 4 →
    driveways_shoveled = 6

theorem tobias_driveways_shoveled_proof : tobias_driveways_shoveled 95 3 5 15 7 8 15 10 4 6 := by
  sorry

end NUMINAMATH_CALUDE_tobias_driveways_shoveled_tobias_driveways_shoveled_proof_l499_49927


namespace NUMINAMATH_CALUDE_solve_for_c_l499_49944

theorem solve_for_c (m a b c : ℝ) (h : m = (c * b * a) / (a - c)) :
  c = (m * a) / (m + b * a) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_c_l499_49944


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l499_49960

theorem sufficient_condition_for_inequality (m : ℝ) (h1 : m ≠ 0) :
  (m > 2 → m + 4 / m > 4) ∧ ¬(m + 4 / m > 4 → m > 2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l499_49960


namespace NUMINAMATH_CALUDE_max_shoe_pairs_l499_49919

theorem max_shoe_pairs (initial_pairs : ℕ) (lost_shoes : ℕ) (max_remaining_pairs : ℕ) : 
  initial_pairs = 23 → lost_shoes = 9 → max_remaining_pairs = 14 →
  max_remaining_pairs = initial_pairs - lost_shoes / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_shoe_pairs_l499_49919


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l499_49945

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1,
    given that one of its asymptotes passes through the point (2, √21) -/
theorem hyperbola_eccentricity (a b : ℝ) (h : a ≠ 0) (k : b ≠ 0) :
  (∃ (x y : ℝ), x = 2 ∧ y = Real.sqrt 21 ∧ y = (b / a) * x) →
  Real.sqrt (1 + (b / a)^2) = 5/2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l499_49945


namespace NUMINAMATH_CALUDE_a_less_than_b_l499_49977

theorem a_less_than_b (x a b : ℝ) (h1 : x > 0) (h2 : a * b ≠ 0) (h3 : a * x < b * x + 1) : a < b := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_b_l499_49977


namespace NUMINAMATH_CALUDE_yulgi_allowance_l499_49980

theorem yulgi_allowance (Y G : ℕ) 
  (sum : Y + G = 6000)
  (sum_minus_diff : Y + G - (Y - G) = 4800)
  (Y_greater : Y > G) : Y = 3600 := by
sorry

end NUMINAMATH_CALUDE_yulgi_allowance_l499_49980


namespace NUMINAMATH_CALUDE_inverse_sum_property_l499_49956

-- Define a function f with domain ℝ
variable (f : ℝ → ℝ)

-- Define the inverse function g of f
variable (g : ℝ → ℝ)

-- Define the symmetry condition for f
def symmetric_about_neg_one_zero (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f ((-1) - x) = f ((-1) + x)

-- Define the inverse relationship between f and g
def inverse_functions (f g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (f x) = x ∧ f (g x) = x

-- Theorem statement
theorem inverse_sum_property
  (h_sym : symmetric_about_neg_one_zero f)
  (h_inv : inverse_functions f g)
  (x₁ x₂ : ℝ)
  (h_sum : x₁ + x₂ = 0) :
  g x₁ + g x₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_property_l499_49956


namespace NUMINAMATH_CALUDE_exponential_inequality_l499_49969

theorem exponential_inequality (x : ℝ) : (2 : ℝ) ^ (2 * x - 7) > (2 : ℝ) ^ (4 * x - 1) ↔ x < -3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l499_49969


namespace NUMINAMATH_CALUDE_reflection_in_fourth_quadrant_l499_49993

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the second quadrant -/
def secondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Defines the fourth quadrant -/
def fourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Reflects a point across the y-axis -/
def reflectYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- Theorem stating that if P is in the second quadrant, 
    then the reflection of Q across the y-axis is in the fourth quadrant -/
theorem reflection_in_fourth_quadrant (a b : ℝ) :
  let p : Point := { x := a, y := b }
  let q : Point := { x := a - 1, y := -b }
  secondQuadrant p → fourthQuadrant (reflectYAxis q) := by
  sorry


end NUMINAMATH_CALUDE_reflection_in_fourth_quadrant_l499_49993


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l499_49931

theorem quadratic_inequality_always_negative : ∀ x : ℝ, -6 * x^2 + 2 * x - 8 < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l499_49931


namespace NUMINAMATH_CALUDE_circle_m_equation_l499_49925

/-- A circle M with center on the negative x-axis and radius 4, tangent to the line 3x + 4y + 4 = 0 -/
structure CircleM where
  /-- The x-coordinate of the center of the circle -/
  a : ℝ
  /-- The center is on the negative x-axis -/
  h_negative : a < 0
  /-- The radius of the circle is 4 -/
  radius : ℝ := 4
  /-- The line 3x + 4y + 4 = 0 is tangent to the circle -/
  h_tangent : |3 * a + 4| / Real.sqrt (3^2 + 4^2) = radius

/-- The equation of circle M is (x+8)² + y² = 16 -/
theorem circle_m_equation (m : CircleM) : 
  ∀ x y : ℝ, (x - m.a)^2 + y^2 = m.radius^2 ↔ (x + 8)^2 + y^2 = 16 :=
sorry

end NUMINAMATH_CALUDE_circle_m_equation_l499_49925


namespace NUMINAMATH_CALUDE_washing_machine_capacity_l499_49994

/-- Given a total amount of clothes and a number of washing machines, 
    calculate the amount of clothes one washing machine can wash per day. -/
def clothes_per_machine (total_clothes : ℕ) (num_machines : ℕ) : ℕ :=
  total_clothes / num_machines

/-- Theorem stating that for 200 pounds of clothes and 8 machines, 
    each machine can wash 25 pounds per day. -/
theorem washing_machine_capacity : clothes_per_machine 200 8 = 25 := by
  sorry

end NUMINAMATH_CALUDE_washing_machine_capacity_l499_49994


namespace NUMINAMATH_CALUDE_notebook_discount_rate_l499_49915

/-- The maximum discount rate that can be applied to a notebook while maintaining a minimum profit margin. -/
theorem notebook_discount_rate (cost : ℝ) (original_price : ℝ) (min_profit_margin : ℝ) :
  cost = 6 →
  original_price = 9 →
  min_profit_margin = 0.05 →
  ∃ (max_discount : ℝ), 
    max_discount = 0.7 ∧ 
    ∀ (discount : ℝ), 
      discount ≤ max_discount →
      (original_price * (1 - discount) - cost) / cost ≥ min_profit_margin :=
by sorry

end NUMINAMATH_CALUDE_notebook_discount_rate_l499_49915


namespace NUMINAMATH_CALUDE_product_of_integers_l499_49958

theorem product_of_integers (w x y z : ℤ) : 
  0 < w → w < x → x < y → y < z → w + z = 5 → w * x * y * z = 36 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l499_49958


namespace NUMINAMATH_CALUDE_log_x_125_l499_49902

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_x_125 (x : ℝ) (h : log 8 (5 * x) = 3) : 
  log x 125 = 3 / log 8 5 := by sorry

end NUMINAMATH_CALUDE_log_x_125_l499_49902


namespace NUMINAMATH_CALUDE_rectangular_prism_to_cube_l499_49965

theorem rectangular_prism_to_cube (a b c : ℝ) (h1 : a = 8) (h2 : b = 8) (h3 : c = 27) :
  ∃ s : ℝ, s^3 = a * b * c ∧ s = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_to_cube_l499_49965


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l499_49933

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- Sum function

/-- Theorem: For an arithmetic sequence, if S_30 = S_60, then S_90 = 0 -/
theorem arithmetic_sequence_sum_property (seq : ArithmeticSequence) 
  (h : seq.S 30 = seq.S 60) : seq.S 90 = 0 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l499_49933
