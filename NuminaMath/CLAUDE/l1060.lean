import Mathlib

namespace NUMINAMATH_CALUDE_science_fiction_readers_l1060_106018

theorem science_fiction_readers
  (total : ℕ)
  (literary : ℕ)
  (both : ℕ)
  (h1 : total = 150)
  (h2 : literary = 90)
  (h3 : both = 60) :
  total = literary + (total - literary - both) - both :=
by sorry

end NUMINAMATH_CALUDE_science_fiction_readers_l1060_106018


namespace NUMINAMATH_CALUDE_triangle_property_l1060_106071

-- Define the binary operation ★
noncomputable def star (A B : ℂ) : ℂ := 
  let ζ : ℂ := Complex.exp (Complex.I * Real.pi / 3)
  ζ * (B - A) + A

-- Define the theorem
theorem triangle_property (I M O : ℂ) :
  star I (star M O) = star (star O I) M →
  -- Triangle IMO is positively oriented
  (Complex.arg ((I - O) / (M - O)) > 0) ∧
  -- Triangle IMO is isosceles with OI = OM
  Complex.abs (I - O) = Complex.abs (M - O) ∧
  -- ∠IOM = 2π/3
  Complex.arg ((I - O) / (M - O)) = 2 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l1060_106071


namespace NUMINAMATH_CALUDE_wife_account_percentage_l1060_106022

def income : ℝ := 800000

def children_percentage : ℝ := 0.2
def num_children : ℕ := 3
def orphan_donation_percentage : ℝ := 0.05
def final_amount : ℝ := 40000

theorem wife_account_percentage :
  let children_total := children_percentage * num_children * income
  let after_children := income - children_total
  let orphan_donation := orphan_donation_percentage * after_children
  let after_donation := after_children - orphan_donation
  let wife_deposit := after_donation - final_amount
  (wife_deposit / income) * 100 = 33 := by sorry

end NUMINAMATH_CALUDE_wife_account_percentage_l1060_106022


namespace NUMINAMATH_CALUDE_total_ipods_l1060_106000

-- Define the initial number of iPods Emmy has
def emmy_initial : ℕ := 14

-- Define the number of iPods Emmy loses
def emmy_lost : ℕ := 6

-- Define Emmy's remaining iPods
def emmy_remaining : ℕ := emmy_initial - emmy_lost

-- Define Rosa's iPods in terms of Emmy's remaining
def rosa : ℕ := emmy_remaining / 2

-- Theorem to prove
theorem total_ipods : emmy_remaining + rosa = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_ipods_l1060_106000


namespace NUMINAMATH_CALUDE_haunted_mansion_entry_exit_l1060_106091

/-- The number of windows in the haunted mansion -/
def num_windows : ℕ := 8

/-- The number of ways to enter and exit the haunted mansion through different windows -/
def num_ways : ℕ := num_windows * (num_windows - 1)

/-- Theorem: The number of ways to enter and exit the haunted mansion through different windows is 56 -/
theorem haunted_mansion_entry_exit : num_ways = 56 := by
  sorry

end NUMINAMATH_CALUDE_haunted_mansion_entry_exit_l1060_106091


namespace NUMINAMATH_CALUDE_simplify_radical_l1060_106095

theorem simplify_radical (a b : ℝ) (h : b > 0) :
  Real.sqrt (20 * a^3 * b^2) = 2 * a * b * Real.sqrt (5 * a) :=
by sorry

end NUMINAMATH_CALUDE_simplify_radical_l1060_106095


namespace NUMINAMATH_CALUDE_two_pairs_satisfying_equation_l1060_106015

theorem two_pairs_satisfying_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℕ), 
    (2 * x₁^3 = y₁^4) ∧ 
    (2 * x₂^3 = y₂^4) ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) :=
by sorry

end NUMINAMATH_CALUDE_two_pairs_satisfying_equation_l1060_106015


namespace NUMINAMATH_CALUDE_sin_2alpha_values_l1060_106089

theorem sin_2alpha_values (α : Real) 
  (h1 : 2 * (Real.tan α)^2 - 7 * Real.tan α + 3 = 0) :
  (π < α ∧ α < 5*π/4 → Real.sin (2*α) = 4/5) ∧
  (5*π/4 < α ∧ α < 3*π/2 → Real.sin (2*α) = 3/5) := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_values_l1060_106089


namespace NUMINAMATH_CALUDE_largest_integer_problem_l1060_106038

theorem largest_integer_problem (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧  -- Four different integers
  (a + b + c + d) / 4 = 72 ∧  -- Average is 72
  a = 21  -- Smallest integer is 21
  → d = 222 := by  -- Largest integer is 222
  sorry

end NUMINAMATH_CALUDE_largest_integer_problem_l1060_106038


namespace NUMINAMATH_CALUDE_vertex_x_coordinate_l1060_106020

def f (x : ℝ) := 3 * x^2 + 9 * x + 5

theorem vertex_x_coordinate (x : ℝ) :
  x = -1.5 ↔ ∀ y : ℝ, f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_vertex_x_coordinate_l1060_106020


namespace NUMINAMATH_CALUDE_range_of_a_l1060_106070

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a * x + 1 else Real.log x

-- Define symmetry about the origin
def symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∃ x > 0, f (-x) = -f x

-- Theorem statement
theorem range_of_a (a : ℝ) :
  symmetric_about_origin (f a) → a ∈ Set.Iic 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1060_106070


namespace NUMINAMATH_CALUDE_ellipse_equation_hyperbola_equation_l1060_106027

-- Ellipse problem
theorem ellipse_equation (f c a b : ℝ) (h1 : f = 8) (h2 : c = 4) (h3 : a = 5) (h4 : b = 3) (h5 : c / a = 0.8) :
  (∀ x y : ℝ, x^2 / 25 + y^2 / 9 = 1) ∨ (∀ x y : ℝ, y^2 / 25 + x^2 / 9 = 1) :=
sorry

-- Hyperbola problem
theorem hyperbola_equation (a b m : ℝ) 
  (h1 : ∀ x y : ℝ, y^2 / 4 - x^2 / 3 = 1 → y^2 / (4*m) - x^2 / (3*m) = 1) 
  (h2 : 3^2 / (6*m) - 2^2 / (8*m) = 1) :
  (∀ x y : ℝ, x^2 / 6 - y^2 / 8 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_hyperbola_equation_l1060_106027


namespace NUMINAMATH_CALUDE_school_travel_time_l1060_106068

/-- Given a boy who walks at 7/6 of his usual rate and reaches school 6 minutes early,
    his usual time to reach the school is 42 minutes. -/
theorem school_travel_time (usual_rate : ℝ) (usual_time : ℝ) 
    (h1 : usual_rate > 0) 
    (h2 : usual_time > 0)
    (h3 : usual_rate * usual_time = (7/6 * usual_rate) * (usual_time - 6)) : 
  usual_time = 42 := by
sorry

end NUMINAMATH_CALUDE_school_travel_time_l1060_106068


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1060_106077

theorem solution_set_of_inequality (x : ℝ) :
  (6 * x^2 + 5 * x < 4) ↔ (-4/3 < x ∧ x < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1060_106077


namespace NUMINAMATH_CALUDE_parabola_line_intersection_dot_product_l1060_106012

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := y = (2/3)*(x + 2)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the intersection points
def M : ℝ × ℝ := (1, 2)
def N : ℝ × ℝ := (4, 4)

-- Define vectors FM and FN
def FM : ℝ × ℝ := (M.1 - focus.1, M.2 - focus.2)
def FN : ℝ × ℝ := (N.1 - focus.1, N.2 - focus.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem parabola_line_intersection_dot_product :
  parabola M.1 M.2 ∧ parabola N.1 N.2 ∧
  line M.1 M.2 ∧ line N.1 N.2 →
  dot_product FM FN = 8 :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_dot_product_l1060_106012


namespace NUMINAMATH_CALUDE_first_day_duration_l1060_106014

def total_distance : ℝ := 115

def day2_distance : ℝ := 6 * 6 + 3 * 3

def day3_distance : ℝ := 7 * 5

def day1_speed : ℝ := 5

theorem first_day_duration : ∃ (hours : ℝ), 
  hours * day1_speed + day2_distance + day3_distance = total_distance ∧ hours = 7 := by
  sorry

end NUMINAMATH_CALUDE_first_day_duration_l1060_106014


namespace NUMINAMATH_CALUDE_curve_not_hyperbola_l1060_106024

/-- The curve equation -/
def curve_equation (m : ℝ) (x y : ℝ) : Prop :=
  (m - 1) * x^2 + (3 - m) * y^2 = (m - 1) * (3 - m)

/-- Definition of a non-hyperbola based on the coefficient condition -/
def is_not_hyperbola (m : ℝ) : Prop :=
  (m - 1) * (3 - m) ≥ 0

/-- Theorem stating that for m in [1,3], the curve is not a hyperbola -/
theorem curve_not_hyperbola (m : ℝ) (h : 1 ≤ m ∧ m ≤ 3) : is_not_hyperbola m := by
  sorry

end NUMINAMATH_CALUDE_curve_not_hyperbola_l1060_106024


namespace NUMINAMATH_CALUDE_points_on_opposite_sides_l1060_106037

-- Define the line
def line (x y : ℝ) : ℝ := 2*y - 6*x + 1

-- Define the points
def origin : ℝ × ℝ := (0, 0)
def point : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem points_on_opposite_sides :
  line origin.1 origin.2 * line point.1 point.2 < 0 := by sorry

end NUMINAMATH_CALUDE_points_on_opposite_sides_l1060_106037


namespace NUMINAMATH_CALUDE_equation_equivalence_l1060_106066

theorem equation_equivalence (x : ℝ) : 
  (x + 1) / 0.3 - (2 * x - 1) / 0.7 = 1 ↔ (10 * x + 10) / 3 - (20 * x - 10) / 7 = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1060_106066


namespace NUMINAMATH_CALUDE_distance_between_cities_l1060_106036

/-- The distance between two cities given specific conditions of bus and car travel --/
theorem distance_between_cities (bus_speed car_speed : ℝ) 
  (h1 : bus_speed = 40)
  (h2 : car_speed = 50)
  (h3 : 0 < bus_speed ∧ 0 < car_speed)
  : ∃ (s : ℝ), s = 160 ∧ 
    (s - 10) / car_speed + 1/4 = (s - 30) / bus_speed := by
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l1060_106036


namespace NUMINAMATH_CALUDE_parabola_tangent_lines_l1060_106061

/-- The parabola defined by x^2 = 4y with focus (0, 1) -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 4 * p.2}

/-- The focus of the parabola -/
def Focus : ℝ × ℝ := (0, 1)

/-- The line perpendicular to the y-axis passing through the focus -/
def PerpendicularLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0}

/-- The intersection points of the parabola and the perpendicular line -/
def IntersectionPoints : Set (ℝ × ℝ) :=
  Parabola ∩ PerpendicularLine

/-- The tangent line at a point on the parabola -/
def TangentLine (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | q.1 + p.2 * q.2 + (p.1^2 / 4 + 1) = 0}

theorem parabola_tangent_lines :
  ∀ p ∈ IntersectionPoints,
    TangentLine p = {q : ℝ × ℝ | q.1 + q.2 + 1 = 0} ∨
    TangentLine p = {q : ℝ × ℝ | q.1 - q.2 - 1 = 0} :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_lines_l1060_106061


namespace NUMINAMATH_CALUDE_revenue_is_432_l1060_106006

/-- Represents the rental business scenario -/
structure RentalBusiness where
  canoe_price : ℕ
  kayak_price : ℕ
  canoe_kayak_ratio : ℚ
  canoe_kayak_difference : ℕ

/-- Calculates the total revenue for the day -/
def total_revenue (rb : RentalBusiness) : ℕ :=
  let kayaks := rb.canoe_kayak_difference * 3
  let canoes := kayaks + rb.canoe_kayak_difference
  kayaks * rb.kayak_price + canoes * rb.canoe_price

/-- Theorem stating that the total revenue for the given scenario is $432 -/
theorem revenue_is_432 (rb : RentalBusiness) 
  (h1 : rb.canoe_price = 9)
  (h2 : rb.kayak_price = 12)
  (h3 : rb.canoe_kayak_ratio = 4/3)
  (h4 : rb.canoe_kayak_difference = 6) :
  total_revenue rb = 432 := by
  sorry

#eval total_revenue { canoe_price := 9, kayak_price := 12, canoe_kayak_ratio := 4/3, canoe_kayak_difference := 6 }

end NUMINAMATH_CALUDE_revenue_is_432_l1060_106006


namespace NUMINAMATH_CALUDE_count_valid_pairs_l1060_106049

def satisfies_equation (x y : ℤ) : Prop :=
  2 * x^2 - 2 * x * y + y^2 = 289

def valid_pair (p : ℤ × ℤ) : Prop :=
  satisfies_equation p.1 p.2 ∧ p.1 ≥ 0

theorem count_valid_pairs :
  ∃ (S : Finset (ℤ × ℤ)), (∀ p ∈ S, valid_pair p) ∧ S.card = 7 ∧
  ∀ p : ℤ × ℤ, valid_pair p → p ∈ S :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l1060_106049


namespace NUMINAMATH_CALUDE_range_of_a_l1060_106029

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 9*x + a^2/x + 7
  else if x > 0 then 9*x + a^2/x - 7
  else 0

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x = -f a (-x)) ∧  -- f is odd
  (∀ x : ℝ, x ≥ 0 → f a x ≥ a + 1) →  -- condition for x ≥ 0
  a ≤ -8/7 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1060_106029


namespace NUMINAMATH_CALUDE_coin_and_die_prob_l1060_106040

/-- A fair coin -/
def FairCoin : Type := Bool

/-- A regular eight-sided die -/
def EightSidedDie : Type := Fin 8

/-- The event of getting heads on a fair coin -/
def headsEvent (c : FairCoin) : Prop := c = true

/-- The event of getting an even number on an eight-sided die -/
def evenDieEvent (d : EightSidedDie) : Prop := d.val % 2 = 0

/-- The probability of an event on a fair coin -/
axiom probCoin (event : FairCoin → Prop) : ℚ

/-- The probability of an event on an eight-sided die -/
axiom probDie (event : EightSidedDie → Prop) : ℚ

/-- The probability of getting heads on a fair coin -/
axiom prob_heads : probCoin headsEvent = 1/2

/-- The probability of getting an even number on an eight-sided die -/
axiom prob_even_die : probDie evenDieEvent = 1/2

/-- The main theorem: The probability of getting heads on a fair coin and an even number
    on a regular eight-sided die when flipped and rolled once is 1/4 -/
theorem coin_and_die_prob :
  probCoin headsEvent * probDie evenDieEvent = 1/4 := by sorry

end NUMINAMATH_CALUDE_coin_and_die_prob_l1060_106040


namespace NUMINAMATH_CALUDE_range_of_k_l1060_106048

def p (k : ℝ) : Prop := k^2 - 8*k - 20 ≤ 0

def q (k : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ 
  (∀ (x y : ℝ), x^2 / (4 - k) + y^2 / (1 - k) = 1 ↔ 
    x^2 / a^2 - y^2 / b^2 = 1) ∧ 
  (4 - k > 0) ∧ (1 - k < 0)

theorem range_of_k (k : ℝ) : 
  ((p k ∨ q k) ∧ ¬(p k ∧ q k)) → 
  ((-2 ≤ k ∧ k ≤ 1) ∨ (4 ≤ k ∧ k ≤ 10)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_l1060_106048


namespace NUMINAMATH_CALUDE_cube_cutting_problem_l1060_106069

theorem cube_cutting_problem :
  ∃! (n : ℕ), ∃ (s : ℕ), s < n ∧ n^3 - s^3 = 152 := by
  sorry

end NUMINAMATH_CALUDE_cube_cutting_problem_l1060_106069


namespace NUMINAMATH_CALUDE_find_a_and_b_l1060_106084

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x^2 + 7 * x - 15 < 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b ≤ 0}

-- State the theorem
theorem find_a_and_b :
  ∃ (a b : ℝ),
    (A ∩ B a b = ∅) ∧
    (A ∪ B a b = {x | -5 < x ∧ x ≤ 2}) ∧
    (a = -7/2) ∧
    (b = 3) := by
  sorry

end NUMINAMATH_CALUDE_find_a_and_b_l1060_106084


namespace NUMINAMATH_CALUDE_third_tea_price_is_175_5_l1060_106041

/-- The price of the third variety of tea -/
def third_tea_price (price1 price2 mixture_price : ℚ) : ℚ :=
  2 * mixture_price - (price1 + price2) / 2

/-- Theorem stating that the price of the third variety of tea is 175.5 given the conditions -/
theorem third_tea_price_is_175_5 :
  third_tea_price 126 135 153 = 175.5 := by
  sorry

end NUMINAMATH_CALUDE_third_tea_price_is_175_5_l1060_106041


namespace NUMINAMATH_CALUDE_sum_of_roots_l1060_106054

theorem sum_of_roots (a b : ℝ) : 
  (a^2 - 4*a - 2023 = 0) → (b^2 - 4*b - 2023 = 0) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1060_106054


namespace NUMINAMATH_CALUDE_product_bounds_l1060_106007

theorem product_bounds (x y z : Real) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π/12) 
  (h4 : x + y + z = π/2) : 
  1/8 ≤ Real.cos x * Real.sin y * Real.cos z ∧ 
  Real.cos x * Real.sin y * Real.cos z ≤ 1/4 + Real.sqrt 3/8 := by
  sorry

end NUMINAMATH_CALUDE_product_bounds_l1060_106007


namespace NUMINAMATH_CALUDE_number_division_problem_l1060_106088

theorem number_division_problem : ∃ N : ℕ, N = (555 + 445) * (2 * (555 - 445)) + 30 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1060_106088


namespace NUMINAMATH_CALUDE_marble_probability_correct_l1060_106081

def marble_probability (initial_red : ℕ) (initial_blue : ℕ) (initial_green : ℕ) (initial_white : ℕ)
                       (removed_red : ℕ) (removed_blue : ℕ) (added_green : ℕ) :
  (ℚ × ℚ × ℚ) :=
  let final_red : ℕ := initial_red - removed_red
  let final_blue : ℕ := initial_blue - removed_blue
  let final_green : ℕ := initial_green + added_green
  let total : ℕ := final_red + final_blue + final_green + initial_white
  ((final_red : ℚ) / total, (final_blue : ℚ) / total, (final_green : ℚ) / total)

theorem marble_probability_correct :
  marble_probability 12 10 8 5 5 4 3 = (7/29, 6/29, 11/29) := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_correct_l1060_106081


namespace NUMINAMATH_CALUDE_mushroom_soup_total_l1060_106072

theorem mushroom_soup_total (team1 team2 team3 : ℕ) 
  (h1 : team1 = 90) 
  (h2 : team2 = 120) 
  (h3 : team3 = 70) : 
  team1 + team2 + team3 = 280 := by
sorry

end NUMINAMATH_CALUDE_mushroom_soup_total_l1060_106072


namespace NUMINAMATH_CALUDE_exists_n_good_not_n_plus_one_good_l1060_106057

/-- Sum of digits of a natural number -/
def S (k : ℕ) : ℕ := sorry

/-- A natural number is n-good if there exists a sequence satisfying the given condition -/
def is_n_good (a n : ℕ) : Prop :=
  ∃ (seq : Fin (n + 1) → ℕ), seq ⟨n, sorry⟩ = a ∧
    ∀ (i : Fin n), seq ⟨i.val + 1, sorry⟩ = seq i - S (seq i)

/-- For any n, there exists a number that is n-good but not (n+1)-good -/
theorem exists_n_good_not_n_plus_one_good :
  ∀ n : ℕ, ∃ a : ℕ, is_n_good a n ∧ ¬is_n_good a (n + 1) := by sorry

end NUMINAMATH_CALUDE_exists_n_good_not_n_plus_one_good_l1060_106057


namespace NUMINAMATH_CALUDE_rose_garden_delivery_l1060_106060

theorem rose_garden_delivery (red yellow white : ℕ) : 
  red + yellow = 120 →
  red + white = 105 →
  yellow + white = 45 →
  red + yellow + white = 135 →
  (red = 90 ∧ white = 15 ∧ yellow = 30) := by
  sorry

end NUMINAMATH_CALUDE_rose_garden_delivery_l1060_106060


namespace NUMINAMATH_CALUDE_no_solution_implies_m_equals_negative_five_l1060_106032

theorem no_solution_implies_m_equals_negative_five (m : ℝ) : 
  (∀ x : ℝ, x ≠ -1 → (3*x - 2)/(x + 1) ≠ 2 + m/(x + 1)) → m = -5 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_equals_negative_five_l1060_106032


namespace NUMINAMATH_CALUDE_left_of_kolya_l1060_106050

/-- The number of people in a physical education class line-up -/
def ClassSize : ℕ := 29

/-- The number of people to the right of Kolya -/
def RightOfKolya : ℕ := 12

/-- The number of people to the left of Sasha -/
def LeftOfSasha : ℕ := 20

/-- The number of people to the right of Sasha -/
def RightOfSasha : ℕ := 8

/-- Theorem: The number of people to the left of Kolya is 16 -/
theorem left_of_kolya : ClassSize - RightOfKolya - 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_left_of_kolya_l1060_106050


namespace NUMINAMATH_CALUDE_projectile_max_height_l1060_106003

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -9 * t^2 + 36 * t + 24

/-- Theorem stating that the maximum height of the projectile is 60 meters -/
theorem projectile_max_height : 
  ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 60 := by
  sorry

end NUMINAMATH_CALUDE_projectile_max_height_l1060_106003


namespace NUMINAMATH_CALUDE_specific_eighth_term_l1060_106035

/-- An arithmetic sequence is defined by its second and fourteenth terms -/
structure ArithmeticSequence where
  second_term : ℚ
  fourteenth_term : ℚ

/-- The eighth term of an arithmetic sequence -/
def eighth_term (seq : ArithmeticSequence) : ℚ :=
  (seq.second_term + seq.fourteenth_term) / 2

/-- Theorem stating the eighth term of the specific arithmetic sequence -/
theorem specific_eighth_term :
  let seq := ArithmeticSequence.mk (8/11) (9/13)
  eighth_term seq = 203/286 := by sorry

end NUMINAMATH_CALUDE_specific_eighth_term_l1060_106035


namespace NUMINAMATH_CALUDE_white_dandelions_on_saturday_l1060_106026

/-- Represents the state of dandelions on a given day -/
structure DandelionState :=
  (yellow : ℕ)
  (white : ℕ)

/-- Represents the lifecycle of a dandelion -/
def dandelionLifecycle : ℕ := 5

/-- The number of days a dandelion is yellow -/
def yellowDays : ℕ := 3

/-- The number of days a dandelion is white -/
def whiteDays : ℕ := 2

/-- Calculates the number of white dandelions on Saturday given the states on Monday and Wednesday -/
def whiteDandelionsOnSaturday (monday : DandelionState) (wednesday : DandelionState) : ℕ :=
  (wednesday.yellow + wednesday.white) - monday.yellow

theorem white_dandelions_on_saturday 
  (monday : DandelionState) 
  (wednesday : DandelionState) 
  (h1 : monday.yellow = 20)
  (h2 : monday.white = 14)
  (h3 : wednesday.yellow = 15)
  (h4 : wednesday.white = 11) :
  whiteDandelionsOnSaturday monday wednesday = 6 := by
  sorry

#check white_dandelions_on_saturday

end NUMINAMATH_CALUDE_white_dandelions_on_saturday_l1060_106026


namespace NUMINAMATH_CALUDE_power_equation_solution_l1060_106098

theorem power_equation_solution : 
  ∃! x : ℤ, (10 : ℝ) ^ x * (10 : ℝ) ^ 652 = 1000 ∧ x = -649 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1060_106098


namespace NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_l1060_106045

-- Define the sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 2}
def B : Set ℝ := {x | (x - 1) / (x + 5) > 0}

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | x < -5 ∨ x > -3} := by sorry

-- Theorem for the intersection of A and complement of B
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x | -3 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_l1060_106045


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1060_106090

theorem expand_and_simplify (x y : ℝ) : (x + 6) * (x + 8 + y) = x^2 + 14*x + x*y + 48 + 6*y := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1060_106090


namespace NUMINAMATH_CALUDE_three_algorithms_among_four_l1060_106075

/-- A statement describing a process or task -/
structure Statement where
  description : String

/-- Predicate to determine if a statement is an algorithm -/
def is_algorithm (s : Statement) : Prop :=
  -- This definition would typically include formal criteria for what constitutes an algorithm
  sorry

/-- The set of given statements -/
def given_statements : Finset Statement := sorry

theorem three_algorithms_among_four :
  ∃ (alg_statements : Finset Statement),
    alg_statements ⊆ given_statements ∧
    (∀ s ∈ alg_statements, is_algorithm s) ∧
    Finset.card alg_statements = 3 ∧
    Finset.card given_statements = 4 := by
  sorry

end NUMINAMATH_CALUDE_three_algorithms_among_four_l1060_106075


namespace NUMINAMATH_CALUDE_school_pet_ownership_l1060_106087

theorem school_pet_ownership (total_students : ℕ) (cat_owners : ℕ) (bird_owners : ℕ)
  (h_total : total_students = 500)
  (h_cats : cat_owners = 80)
  (h_birds : bird_owners = 120) :
  (cat_owners : ℚ) / total_students * 100 = 16 ∧
  (bird_owners : ℚ) / total_students * 100 = 24 :=
by sorry

end NUMINAMATH_CALUDE_school_pet_ownership_l1060_106087


namespace NUMINAMATH_CALUDE_zoo_animal_ratio_l1060_106092

/-- Given a zoo with penguins and polar bears, prove the ratio of polar bears to penguins -/
theorem zoo_animal_ratio (num_penguins num_total : ℕ) 
  (h1 : num_penguins = 21)
  (h2 : num_total = 63) :
  (num_total - num_penguins) / num_penguins = 2 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_ratio_l1060_106092


namespace NUMINAMATH_CALUDE_spatial_relations_l1060_106073

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallelPlaneLine : Plane → Line → Prop)
variable (perpendicularPlaneLine : Plane → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- State the theorem
theorem spatial_relations 
  (m n l : Line) (α β : Plane) 
  (h_distinct_lines : m ≠ n ∧ m ≠ l ∧ n ≠ l) 
  (h_distinct_planes : α ≠ β) :
  -- Define the propositions
  let p1 := ∀ m n α, parallel m n → contains α n → parallelPlaneLine α m
  let p2 := ∀ l m α β, perpendicularPlaneLine α l → perpendicularPlaneLine β m → perpendicular l m → perpendicularPlanes α β
  let p3 := ∀ l m n, perpendicular l n → perpendicular m n → parallel l m
  let p4 := ∀ m n α β, perpendicularPlanes α β → intersect α β m → contains β n → perpendicular n m → perpendicularPlaneLine α n
  -- The theorem statement
  (¬p1 ∧ p2 ∧ ¬p3 ∧ p4) :=
by
  sorry

end NUMINAMATH_CALUDE_spatial_relations_l1060_106073


namespace NUMINAMATH_CALUDE_factory_machines_capping_l1060_106093

/-- Represents a machine in the factory -/
structure Machine where
  capping_rate : ℕ  -- bottles capped per minute
  working_time : ℕ  -- working time in minutes

/-- Calculates the total number of bottles capped by a machine -/
def total_capped (m : Machine) : ℕ := m.capping_rate * m.working_time

theorem factory_machines_capping (machine_a machine_b machine_c machine_d machine_e : Machine) :
  machine_a.capping_rate = 24 ∧
  machine_a.working_time = 10 ∧
  machine_b.capping_rate = machine_a.capping_rate - 3 ∧
  machine_b.working_time = 12 ∧
  machine_c.capping_rate = machine_b.capping_rate + 6 ∧
  machine_c.working_time = 15 ∧
  machine_d.capping_rate = machine_c.capping_rate - 4 ∧
  machine_d.working_time = 8 ∧
  machine_e.capping_rate = machine_d.capping_rate + 5 ∧
  machine_e.working_time = 5 →
  total_capped machine_a = 240 ∧
  total_capped machine_b = 252 ∧
  total_capped machine_c = 405 ∧
  total_capped machine_d = 184 ∧
  total_capped machine_e = 140 := by
  sorry

#check factory_machines_capping

end NUMINAMATH_CALUDE_factory_machines_capping_l1060_106093


namespace NUMINAMATH_CALUDE_subcommittee_count_l1060_106086

theorem subcommittee_count : 
  let total_members : ℕ := 12
  let coach_count : ℕ := 5
  let subcommittee_size : ℕ := 5
  let total_subcommittees := Nat.choose total_members subcommittee_size
  let non_coach_count := total_members - coach_count
  let all_non_coach_subcommittees := Nat.choose non_coach_count subcommittee_size
  total_subcommittees - all_non_coach_subcommittees = 771 := by
sorry

end NUMINAMATH_CALUDE_subcommittee_count_l1060_106086


namespace NUMINAMATH_CALUDE_zoo_count_l1060_106001

/-- Counts the total number of animals observed during a zoo trip --/
def count_animals (snakes : ℕ) (arctic_foxes : ℕ) (leopards : ℕ) : ℕ :=
  let bee_eaters := 10 * (snakes / 2 + 2 * leopards)
  let cheetahs := 4 * (arctic_foxes - leopards)
  let alligators := 3 * (snakes * arctic_foxes * leopards)
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators

/-- Theorem stating the total number of animals counted during the zoo trip --/
theorem zoo_count : count_animals 100 80 20 = 481340 := by
  sorry

end NUMINAMATH_CALUDE_zoo_count_l1060_106001


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_l1060_106065

theorem min_value_sum_fractions (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) / z + (x + z) / y + (y + z) / x + (x + y + z) / (x + y) ≥ 7 ∧
  ((x + y) / z + (x + z) / y + (y + z) / x + (x + y + z) / (x + y) = 7 ↔ x = y ∧ y = z) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_l1060_106065


namespace NUMINAMATH_CALUDE_max_difference_second_largest_smallest_l1060_106052

theorem max_difference_second_largest_smallest (a b c d e f g h : ℕ) :
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 →
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧ f < g ∧ g < h →
  (a + b + c) / 3 = 9 →
  (a + b + c + d + e + f + g + h) / 8 = 19 →
  (f + g + h) / 3 = 29 →
  ∃ (a' b' c' d' e' f' g' h' : ℕ),
    a' ≠ 0 ∧ b' ≠ 0 ∧ c' ≠ 0 ∧ d' ≠ 0 ∧ e' ≠ 0 ∧ f' ≠ 0 ∧ g' ≠ 0 ∧ h' ≠ 0 ∧
    a' < b' ∧ b' < c' ∧ c' < d' ∧ d' < e' ∧ e' < f' ∧ f' < g' ∧ g' < h' ∧
    (a' + b' + c') / 3 = 9 ∧
    (a' + b' + c' + d' + e' + f' + g' + h') / 8 = 19 ∧
    (f' + g' + h') / 3 = 29 ∧
    g' - b' = 26 ∧
    ∀ (a'' b'' c'' d'' e'' f'' g'' h'' : ℕ),
      a'' ≠ 0 ∧ b'' ≠ 0 ∧ c'' ≠ 0 ∧ d'' ≠ 0 ∧ e'' ≠ 0 ∧ f'' ≠ 0 ∧ g'' ≠ 0 ∧ h'' ≠ 0 →
      a'' < b'' ∧ b'' < c'' ∧ c'' < d'' ∧ d'' < e'' ∧ e'' < f'' ∧ f'' < g'' ∧ g'' < h'' →
      (a'' + b'' + c'') / 3 = 9 →
      (a'' + b'' + c'' + d'' + e'' + f'' + g'' + h'') / 8 = 19 →
      (f'' + g'' + h'') / 3 = 29 →
      g'' - b'' ≤ 26 :=
by
  sorry

end NUMINAMATH_CALUDE_max_difference_second_largest_smallest_l1060_106052


namespace NUMINAMATH_CALUDE_lcm_of_three_numbers_l1060_106078

theorem lcm_of_three_numbers (A B C : ℕ+) 
  (h_product : A * B * C = 185771616)
  (h_hcf_abc : Nat.gcd A (Nat.gcd B C) = 121)
  (h_hcf_ab : Nat.gcd A B = 363) :
  Nat.lcm A (Nat.lcm B C) = 61919307 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_three_numbers_l1060_106078


namespace NUMINAMATH_CALUDE_number_problem_l1060_106063

theorem number_problem : ∃ x : ℝ, 1.3333 * x = 4.82 ∧ abs (x - 3.615) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1060_106063


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l1060_106062

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal [true, false, true, true, true, true, false, false]) = [2, 3, 3, 0] := by
  sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l1060_106062


namespace NUMINAMATH_CALUDE_valid_sequence_probability_l1060_106033

/-- Recursive function to calculate the number of valid sequences of length n -/
def b : ℕ → ℕ
| 0 => 0
| 1 => 0
| 2 => 0
| 3 => 0
| 4 => 1
| 5 => 1
| 6 => 2
| 7 => 3
| n + 8 => b (n + 5) + b (n + 4)

/-- The probability of generating a valid sequence of length 12 -/
def prob : ℚ := 5 / 1024

theorem valid_sequence_probability :
  b 12 = 5 ∧ 2^10 = 1024 ∧ prob = (b 12 : ℚ) / 2^10 := by sorry

end NUMINAMATH_CALUDE_valid_sequence_probability_l1060_106033


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1060_106002

theorem inequality_and_equality_condition (a b c : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) 
  (h_not_all_equal : ¬(a = b ∧ b = c)) : 
  (((a - b*c)^2 + (b - a*c)^2 + (c - a*b)^2) ≥ 
   (1/2) * ((a - b)^2 + (b - c)^2 + (c - a)^2)) ∧
  ((a - b*c)^2 + (b - a*c)^2 + (c - a*b)^2 = 
   (1/2) * ((a - b)^2 + (b - c)^2 + (c - a)^2) ↔ 
   ((a > 0 ∧ b = 0 ∧ c = 0) ∨ (a = 0 ∧ b > 0 ∧ c = 0) ∨ (a = 0 ∧ b = 0 ∧ c > 0))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1060_106002


namespace NUMINAMATH_CALUDE_extra_bananas_l1060_106013

theorem extra_bananas (total_children absent_children original_bananas : ℕ) 
  (h1 : total_children = 840)
  (h2 : absent_children = 420)
  (h3 : original_bananas = 2) : 
  let present_children := total_children - absent_children
  let total_bananas := total_children * original_bananas
  let actual_bananas := total_bananas / present_children
  actual_bananas - original_bananas = 2 := by sorry

end NUMINAMATH_CALUDE_extra_bananas_l1060_106013


namespace NUMINAMATH_CALUDE_alberto_clara_distance_difference_l1060_106053

/-- The difference in distance traveled between two bikers over a given time -/
def distance_difference (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed1 * time) - (speed2 * time)

/-- Theorem stating the difference in distance traveled between Alberto and Clara -/
theorem alberto_clara_distance_difference :
  distance_difference 16 12 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_alberto_clara_distance_difference_l1060_106053


namespace NUMINAMATH_CALUDE_rectangle_with_equal_sides_is_square_inverse_proposition_inverse_proposition_is_true_l1060_106042

-- Define what a rectangle is
def is_rectangle (shape : Type) : Prop := sorry

-- Define what a square is
def is_square (shape : Type) : Prop := sorry

-- Define what it means for a shape to have equal adjacent sides
def has_equal_adjacent_sides (shape : Type) : Prop := sorry

-- The original proposition
theorem rectangle_with_equal_sides_is_square (shape : Type) :
  is_rectangle shape → has_equal_adjacent_sides shape → is_square shape := sorry

-- The inverse proposition
theorem inverse_proposition (shape : Type) :
  is_square shape → is_rectangle shape ∧ has_equal_adjacent_sides shape := sorry

-- The main theorem: proving that the inverse proposition is true
theorem inverse_proposition_is_true :
  (∀ shape, is_square shape → is_rectangle shape ∧ has_equal_adjacent_sides shape) := sorry

end NUMINAMATH_CALUDE_rectangle_with_equal_sides_is_square_inverse_proposition_inverse_proposition_is_true_l1060_106042


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1060_106043

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1060_106043


namespace NUMINAMATH_CALUDE_new_assistant_drawing_time_main_theorem_l1060_106004

/-- Represents a beer barrel with two taps -/
structure BeerBarrel where
  capacity : ℕ
  midwayTapRate : ℚ  -- litres per minute
  lowerTapRate : ℚ   -- litres per minute

/-- Calculates the time taken to empty half the barrel using the midway tap -/
def timeToHalfEmpty (barrel : BeerBarrel) : ℚ :=
  (barrel.capacity / 2) / barrel.midwayTapRate

/-- Calculates the additional time the lower tap was used -/
def additionalLowerTapTime : ℕ := 24

/-- Theorem: The new assistant drew beer for 150 minutes -/
theorem new_assistant_drawing_time (barrel : BeerBarrel)
    (h1 : barrel.capacity = 36)
    (h2 : barrel.midwayTapRate = 1 / 6)
    (h3 : barrel.lowerTapRate = 1 / 4)
    : ℚ :=
  150

/-- Main theorem to prove -/
theorem main_theorem (barrel : BeerBarrel)
    (h1 : barrel.capacity = 36)
    (h2 : barrel.midwayTapRate = 1 / 6)
    (h3 : barrel.lowerTapRate = 1 / 4)
    : new_assistant_drawing_time barrel h1 h2 h3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_new_assistant_drawing_time_main_theorem_l1060_106004


namespace NUMINAMATH_CALUDE_nineteenth_replacement_in_july_l1060_106085

/-- Represents the months of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Calculates the number of months between two replacements -/
def monthsBetweenReplacements : ℕ := 7

/-- Calculates the total number of months after a given number of replacements -/
def totalMonthsAfter (replacements : ℕ) : ℕ :=
  monthsBetweenReplacements * (replacements - 1)

/-- Determines the month after a given number of months from January -/
def monthAfter (months : ℕ) : Month :=
  match months % 12 with
  | 0 => Month.January
  | 1 => Month.February
  | 2 => Month.March
  | 3 => Month.April
  | 4 => Month.May
  | 5 => Month.June
  | 6 => Month.July
  | 7 => Month.August
  | 8 => Month.September
  | 9 => Month.October
  | 10 => Month.November
  | _ => Month.December

/-- Theorem: The 19th replacement occurs in July -/
theorem nineteenth_replacement_in_july :
  monthAfter (totalMonthsAfter 19) = Month.July := by
  sorry

end NUMINAMATH_CALUDE_nineteenth_replacement_in_july_l1060_106085


namespace NUMINAMATH_CALUDE_circle_radius_l1060_106005

/-- Given a circle and a line passing through its center, prove the radius is 3 -/
theorem circle_radius (m : ℝ) : 
  (∀ x y, x^2 + y^2 - 2*x + m*y - 4 = 0 → (x - 1)^2 + (y + m/2)^2 = 9) ∧ 
  (2 * 1 + (-m/2) = 0) →
  ∃ r, r = 3 ∧ ∀ x y, (x - 1)^2 + (y + m/2)^2 = r^2 → x^2 + y^2 - 2*x + m*y - 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l1060_106005


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1060_106044

theorem binomial_coefficient_ratio (n k : ℕ) (h1 : n > 0) (h2 : k > 0) :
  (Nat.choose n k : ℚ) / (Nat.choose n (k + 1) : ℚ) = 1 / 3 ∧
  (Nat.choose n (k + 1) : ℚ) / (Nat.choose n (k + 2) : ℚ) = 1 / 2 →
  n + k = 6 :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1060_106044


namespace NUMINAMATH_CALUDE_tina_sold_26_more_than_katya_l1060_106030

/-- The number of glasses of lemonade sold by Katya -/
def katya_sales : ℕ := 8

/-- The number of glasses of lemonade sold by Ricky -/
def ricky_sales : ℕ := 9

/-- The number of glasses of lemonade sold by Tina -/
def tina_sales : ℕ := 2 * (katya_sales + ricky_sales)

/-- Theorem: Tina sold 26 more glasses of lemonade than Katya -/
theorem tina_sold_26_more_than_katya : tina_sales - katya_sales = 26 := by
  sorry

end NUMINAMATH_CALUDE_tina_sold_26_more_than_katya_l1060_106030


namespace NUMINAMATH_CALUDE_negative_power_equality_l1060_106011

theorem negative_power_equality : -2010^2011 = (-2010)^2011 := by sorry

end NUMINAMATH_CALUDE_negative_power_equality_l1060_106011


namespace NUMINAMATH_CALUDE_cistern_fill_time_l1060_106080

-- Define the time to fill without leak
def T : ℝ := 12

-- Define the time to fill with leak
def time_with_leak : ℝ := T + 2

-- Define the time to empty when full
def time_to_empty : ℝ := 84

-- State the theorem
theorem cistern_fill_time :
  (1 / T - 1 / time_to_empty = 1 / time_with_leak) ∧
  (T > 0) :=
sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l1060_106080


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l1060_106039

/-- The distance between an endpoint of the major axis and an endpoint of the minor axis of the ellipse 4(x+2)^2 + 16y^2 = 64 is 2√5. -/
theorem ellipse_axis_endpoint_distance : 
  ∃ (C D : ℝ × ℝ),
    (∀ (x y : ℝ), 4 * (x + 2)^2 + 16 * y^2 = 64 → 
      ((x = C.1 ∧ y = C.2) ∨ (x = -C.1 ∧ y = -C.2)) ∨ 
      ((x = D.1 ∧ y = D.2) ∨ (x = -D.1 ∧ y = -D.2))) →
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l1060_106039


namespace NUMINAMATH_CALUDE_craig_total_commissions_l1060_106076

/-- Represents the commission structure for an appliance brand -/
structure CommissionStructure where
  refrigerator_base : ℝ
  refrigerator_rate : ℝ
  washing_machine_base : ℝ
  washing_machine_rate : ℝ
  oven_base : ℝ
  oven_rate : ℝ

/-- Represents the sales data for an appliance brand -/
structure SalesData where
  refrigerators : ℕ
  refrigerators_price : ℝ
  washing_machines : ℕ
  washing_machines_price : ℝ
  ovens : ℕ
  ovens_price : ℝ

/-- Calculates the commission for a single appliance type -/
def calculate_commission (base : ℝ) (rate : ℝ) (quantity : ℕ) (total_price : ℝ) : ℝ :=
  (base + rate * total_price) * quantity

/-- Calculates the total commission for a brand -/
def total_brand_commission (cs : CommissionStructure) (sd : SalesData) : ℝ :=
  calculate_commission cs.refrigerator_base cs.refrigerator_rate sd.refrigerators sd.refrigerators_price +
  calculate_commission cs.washing_machine_base cs.washing_machine_rate sd.washing_machines sd.washing_machines_price +
  calculate_commission cs.oven_base cs.oven_rate sd.ovens sd.ovens_price

/-- Main theorem: Craig's total commissions for the week -/
theorem craig_total_commissions :
  let brand_a_cs : CommissionStructure := {
    refrigerator_base := 75,
    refrigerator_rate := 0.08,
    washing_machine_base := 50,
    washing_machine_rate := 0.10,
    oven_base := 60,
    oven_rate := 0.12
  }
  let brand_b_cs : CommissionStructure := {
    refrigerator_base := 90,
    refrigerator_rate := 0.06,
    washing_machine_base := 40,
    washing_machine_rate := 0.14,
    oven_base := 70,
    oven_rate := 0.10
  }
  let brand_a_sales : SalesData := {
    refrigerators := 3,
    refrigerators_price := 5280,
    washing_machines := 4,
    washing_machines_price := 2140,
    ovens := 5,
    ovens_price := 4620
  }
  let brand_b_sales : SalesData := {
    refrigerators := 2,
    refrigerators_price := 3780,
    washing_machines := 3,
    washing_machines_price := 2490,
    ovens := 4,
    ovens_price := 3880
  }
  total_brand_commission brand_a_cs brand_a_sales + total_brand_commission brand_b_cs brand_b_sales = 9252.60 := by
  sorry

end NUMINAMATH_CALUDE_craig_total_commissions_l1060_106076


namespace NUMINAMATH_CALUDE_max_value_h_exists_m_for_inequality_l1060_106034

open Real

/-- The natural logarithm function -/
noncomputable def f (x : ℝ) : ℝ := log x

/-- The square function -/
def g (x : ℝ) : ℝ := x^2

/-- The function h(x) = ln x - x + 1 -/
noncomputable def h (x : ℝ) : ℝ := f x - x + 1

theorem max_value_h :
  ∀ x > 0, h x ≤ 0 ∧ ∃ x₀ > 0, h x₀ = 0 :=
sorry

theorem exists_m_for_inequality (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hlt : x₁ < x₂) :
  ∃ m ≤ (-1/2), m * (g x₂ - g x₁) - x₂ * f x₂ + x₁ * f x₁ > 0 :=
sorry

end NUMINAMATH_CALUDE_max_value_h_exists_m_for_inequality_l1060_106034


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1060_106019

/-- Proves that a boat traveling 54 km downstream in 3 hours and 54 km upstream in 9 hours has a speed of 12 km/hr in still water. -/
theorem boat_speed_in_still_water : 
  ∀ (v_b v_r : ℝ), 
    v_b > 0 → 
    v_r > 0 → 
    v_b + v_r = 54 / 3 → 
    v_b - v_r = 54 / 9 → 
    v_b = 12 := by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1060_106019


namespace NUMINAMATH_CALUDE_group_size_problem_l1060_106082

theorem group_size_problem (total_collection : ℕ) 
  (h1 : total_collection = 3249) 
  (h2 : ∃ n : ℕ, n * n = total_collection) : 
  ∃ n : ℕ, n = 57 ∧ n * n = total_collection :=
sorry

end NUMINAMATH_CALUDE_group_size_problem_l1060_106082


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1060_106031

theorem sqrt_equation_solution (z : ℝ) : 
  (Real.sqrt 1.21) / (Real.sqrt 0.81) + (Real.sqrt z) / (Real.sqrt 0.49) = 2.9365079365079367 → 
  z = 1.44 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1060_106031


namespace NUMINAMATH_CALUDE_probability_correct_l1060_106055

/-- Represents a runner on a circular track -/
structure Runner where
  direction : Bool  -- true for counterclockwise, false for clockwise
  lap_time : ℝ      -- time to complete one lap in seconds

/-- Represents the track and race setup -/
structure RaceSetup where
  track_length : ℝ           -- length of the track in meters
  focus_start : ℝ            -- start of focus area in meters from start line
  focus_length : ℝ           -- length of focus area in meters
  alice : Runner
  bob : Runner
  race_start_time : ℝ        -- start time of the race in seconds
  photo_start_time : ℝ       -- start time of photo opportunity in seconds
  photo_end_time : ℝ         -- end time of photo opportunity in seconds

def setup : RaceSetup := {
  track_length := 500
  focus_start := 50
  focus_length := 150
  alice := { direction := true, lap_time := 120 }
  bob := { direction := false, lap_time := 75 }
  race_start_time := 0
  photo_start_time := 15 * 60
  photo_end_time := 16 * 60
}

/-- Calculates the probability of both runners being in the focus area -/
def probability_both_in_focus (s : RaceSetup) : ℚ :=
  11/60

theorem probability_correct (s : RaceSetup) :
  s = setup → probability_both_in_focus s = 11/60 := by sorry

end NUMINAMATH_CALUDE_probability_correct_l1060_106055


namespace NUMINAMATH_CALUDE_complement_intersection_M_N_l1060_106067

-- Define the sets M and N
def M : Set ℝ := {x | x < 3}
def N : Set ℝ := {x | x > -1}

-- Define the universal set U
def U : Type := ℝ

-- State the theorem
theorem complement_intersection_M_N :
  (M ∩ N)ᶜ = {x : ℝ | x ≤ -1 ∨ x ≥ 3} :=
by sorry

end NUMINAMATH_CALUDE_complement_intersection_M_N_l1060_106067


namespace NUMINAMATH_CALUDE_inverse_of_A_l1060_106021

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -1; 4, 3]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![3/10, 1/10; -2/5, 1/5]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l1060_106021


namespace NUMINAMATH_CALUDE_max_intersection_points_l1060_106096

/-- Given 20 points on the positive x-axis and 10 points on the positive y-axis,
    the maximum number of intersection points in the first quadrant formed by
    the segments connecting these points is equal to the product of
    combinations C(20,2) and C(10,2). -/
theorem max_intersection_points (x_points y_points : ℕ) 
  (hx : x_points = 20) (hy : y_points = 10) :
  (x_points.choose 2) * (y_points.choose 2) = 8550 := by
  sorry

end NUMINAMATH_CALUDE_max_intersection_points_l1060_106096


namespace NUMINAMATH_CALUDE_max_value_x_plus_y_l1060_106079

/-- Given plane vectors OA, OB, OC satisfying certain conditions, 
    the maximum value of x + y is √2. -/
theorem max_value_x_plus_y (OA OB OC : ℝ × ℝ) (x y : ℝ) : 
  (norm OA = 1) → 
  (norm OB = 1) → 
  (norm OC = 1) → 
  (OA.1 * OB.1 + OA.2 * OB.2 = 0) → 
  (OC = (x * OA.1 + y * OB.1, x * OA.2 + y * OB.2)) → 
  (∃ (x y : ℝ), x + y ≤ Real.sqrt 2 ∧ 
    ∀ (x' y' : ℝ), (OC = (x' * OA.1 + y' * OB.1, x' * OA.2 + y' * OB.2)) → 
      x' + y' ≤ x + y) :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_y_l1060_106079


namespace NUMINAMATH_CALUDE_pure_imaginary_equation_l1060_106025

-- Define the complex number i
def i : ℂ := Complex.I

-- Define a pure imaginary number
def isPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem pure_imaginary_equation (z : ℂ) (b : ℝ) 
  (h1 : isPureImaginary z) 
  (h2 : (2 - i) * z = 4 - b * i) : 
  b = -8 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_equation_l1060_106025


namespace NUMINAMATH_CALUDE_mark_and_carolyn_money_l1060_106010

theorem mark_and_carolyn_money : 
  3/4 + 3/10 = 21/20 := by sorry

end NUMINAMATH_CALUDE_mark_and_carolyn_money_l1060_106010


namespace NUMINAMATH_CALUDE_absolute_value_equality_l1060_106099

theorem absolute_value_equality (x : ℝ) : |x - 3| = |x + 1| → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l1060_106099


namespace NUMINAMATH_CALUDE_nursery_seedling_price_l1060_106016

theorem nursery_seedling_price :
  ∀ (price_day2 : ℝ),
    (price_day2 > 0) →
    (2 * (8000 / (price_day2 - 5)) = 17000 / price_day2) →
    price_day2 = 85 := by
  sorry

end NUMINAMATH_CALUDE_nursery_seedling_price_l1060_106016


namespace NUMINAMATH_CALUDE_solution_set_l1060_106023

/-- A decreasing function f: ℝ → ℝ that passes through (0, 3) and (3, -1) -/
def f : ℝ → ℝ :=
  sorry

/-- f is a decreasing function -/
axiom f_decreasing : ∀ x y, x < y → f y < f x

/-- f(0) = 3 -/
axiom f_at_zero : f 0 = 3

/-- f(3) = -1 -/
axiom f_at_three : f 3 = -1

/-- The solution set of |f(x+1) - 1| < 2 is (-1, 2) -/
theorem solution_set : 
  {x : ℝ | |f (x + 1) - 1| < 2} = Set.Ioo (-1) 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_l1060_106023


namespace NUMINAMATH_CALUDE_valid_sequence_power_of_two_l1060_106009

/-- A sequence of pairwise distinct reals satisfying the given condition -/
def ValidSequence (N : ℕ) (a : ℕ → ℝ) : Prop :=
  N ≥ 3 ∧
  (∀ i j, i < N → j < N → i ≠ j → a i ≠ a j) ∧
  (∀ i, i < N → a i ≥ a ((2 * i) % N))

/-- The theorem stating that N must be a power of 2 -/
theorem valid_sequence_power_of_two (N : ℕ) (a : ℕ → ℝ) :
  ValidSequence N a → ∃ k : ℕ, N = 2^k :=
sorry

end NUMINAMATH_CALUDE_valid_sequence_power_of_two_l1060_106009


namespace NUMINAMATH_CALUDE_price_adjustment_percentage_l1060_106028

theorem price_adjustment_percentage (P : ℝ) (x : ℝ) (h : P > 0) :
  P * (1 + x / 100) * (1 - x / 100) = 0.75 * P →
  x = 50 := by
sorry

end NUMINAMATH_CALUDE_price_adjustment_percentage_l1060_106028


namespace NUMINAMATH_CALUDE_max_value_is_six_range_of_m_l1060_106059

-- Define the problem setup
def problem_setup (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 6

-- Define the maximum value function
def max_value (a b c : ℝ) : ℝ := a + 2*b + c

-- Theorem for the maximum value
theorem max_value_is_six (a b c : ℝ) (h : problem_setup a b c) :
  ∃ (M : ℝ), (∀ (a' b' c' : ℝ), problem_setup a' b' c' → max_value a' b' c' ≤ M) ∧
             M = 6 :=
sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∀ (x : ℝ), |x + 1| + |x + m| ≥ 6) ↔ (m ≥ 7 ∨ m ≤ -5) :=
sorry

end NUMINAMATH_CALUDE_max_value_is_six_range_of_m_l1060_106059


namespace NUMINAMATH_CALUDE_cubic_roots_inequality_l1060_106046

theorem cubic_roots_inequality (a b : ℝ) 
  (h : ∀ x : ℝ, a * x^3 - x^2 + b * x - 1 = 0 → x > 0) : 
  0 < 3 * a * b ∧ 3 * a * b ≤ 1 ∧ b ≥ Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_inequality_l1060_106046


namespace NUMINAMATH_CALUDE_xiao_ming_reading_inequality_l1060_106056

/-- Represents Xiao Ming's reading situation -/
def reading_situation (total_pages : ℕ) (total_days : ℕ) (initial_pages_per_day : ℕ) (initial_days : ℕ) (remaining_pages_per_day : ℝ) : Prop :=
  (initial_pages_per_day * initial_days : ℝ) + (remaining_pages_per_day * (total_days - initial_days)) ≥ total_pages

/-- The inequality correctly represents Xiao Ming's reading situation -/
theorem xiao_ming_reading_inequality :
  reading_situation 72 10 5 2 x ↔ 10 + 8 * x ≥ 72 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_reading_inequality_l1060_106056


namespace NUMINAMATH_CALUDE_factor_x4_minus_81_l1060_106083

theorem factor_x4_minus_81 (x : ℝ) : x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_x4_minus_81_l1060_106083


namespace NUMINAMATH_CALUDE_correct_statement_l1060_106008

-- Define propositions P and Q
def P : Prop := Real.pi < 2
def Q : Prop := Real.pi > 3

-- Theorem statement
theorem correct_statement :
  (P ∨ Q) ∧ (¬P) := by sorry

end NUMINAMATH_CALUDE_correct_statement_l1060_106008


namespace NUMINAMATH_CALUDE_parabola_coefficients_l1060_106047

/-- A parabola with a vertical axis of symmetry -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℚ) : ℚ :=
  p.a * x^2 + p.b * x + p.c

/-- The vertex of a parabola -/
def Parabola.vertex (p : Parabola) : ℚ × ℚ :=
  (- p.b / (2 * p.a), p.c - p.b^2 / (4 * p.a))

theorem parabola_coefficients :
  ∃ (p : Parabola),
    p.vertex = (5, -3) ∧
    p.y_coord 3 = 7 ∧
    p.a = 5/2 ∧
    p.b = -25 ∧
    p.c = 119/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l1060_106047


namespace NUMINAMATH_CALUDE_linear_function_passes_through_points_l1060_106017

/-- A linear function passing through (-1, 4) also passes through (1, 0) -/
theorem linear_function_passes_through_points :
  ∀ k : ℝ, (4 = k * (-1) - k) → (0 = k * 1 - k) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_passes_through_points_l1060_106017


namespace NUMINAMATH_CALUDE_cube_equation_solution_l1060_106064

theorem cube_equation_solution :
  ∃ x : ℝ, (2*x - 8)^3 = 64 ∧ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l1060_106064


namespace NUMINAMATH_CALUDE_sphere_plane_distance_l1060_106097

/-- The distance between the center of a sphere and a plane intersecting it -/
theorem sphere_plane_distance (r : ℝ) (A : ℝ) (h1 : r = 2) (h2 : A = Real.pi) :
  Real.sqrt (r^2 - (A / Real.pi)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_plane_distance_l1060_106097


namespace NUMINAMATH_CALUDE_hexagon_area_from_triangle_l1060_106094

/-- Given a triangle XYZ with circumcircle radius R and perimeter P, 
    the area of the hexagon formed by the intersection points of 
    the perpendicular bisectors with the circumcircle is (P * R) / 4 -/
theorem hexagon_area_from_triangle (R P : ℝ) (hR : R = 10) (hP : P = 45) :
  let hexagon_area := (P * R) / 4
  hexagon_area = 112.5 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_from_triangle_l1060_106094


namespace NUMINAMATH_CALUDE_flour_per_new_crust_l1060_106074

/-- Amount of flour per pie crust in cups -/
def flour_per_crust : ℚ := 1 / 8

/-- Number of pie crusts made daily -/
def daily_crusts : ℕ := 40

/-- Total flour used daily in cups -/
def total_flour : ℚ := daily_crusts * flour_per_crust

/-- Number of new pie crusts -/
def new_crusts : ℕ := 50

/-- Number of cakes -/
def cakes : ℕ := 10

/-- Flour used for cakes in cups -/
def cake_flour : ℚ := 1

/-- Theorem stating the amount of flour per new pie crust -/
theorem flour_per_new_crust : 
  (total_flour - cake_flour) / new_crusts = 2 / 25 := by sorry

end NUMINAMATH_CALUDE_flour_per_new_crust_l1060_106074


namespace NUMINAMATH_CALUDE_max_writers_is_fifty_l1060_106051

/-- Represents the number of people at a newspaper conference --/
structure ConferenceAttendees where
  total : Nat
  editors : Nat
  both : Nat
  neither : Nat
  hTotal : total = 90
  hEditors : editors > 38
  hNeither : neither = 2 * both
  hBothMax : both ≤ 6

/-- The maximum number of writers at the conference --/
def maxWriters (c : ConferenceAttendees) : Nat :=
  c.total - c.editors - c.both

/-- Theorem stating that the maximum number of writers is 50 --/
theorem max_writers_is_fifty (c : ConferenceAttendees) : maxWriters c ≤ 50 ∧ ∃ c', maxWriters c' = 50 := by
  sorry

#eval maxWriters { total := 90, editors := 39, both := 1, neither := 2, hTotal := rfl, hEditors := by norm_num, hNeither := rfl, hBothMax := by norm_num }

end NUMINAMATH_CALUDE_max_writers_is_fifty_l1060_106051


namespace NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l1060_106058

theorem factorial_of_factorial_divided_by_factorial :
  (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l1060_106058
