import Mathlib

namespace NUMINAMATH_CALUDE_convention_handshakes_l758_75891

/-- The number of handshakes in a convention with multiple companies --/
def number_of_handshakes (num_companies : ℕ) (representatives_per_company : ℕ) : ℕ :=
  let total_people := num_companies * representatives_per_company
  let handshakes_per_person := total_people - representatives_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem stating that the number of handshakes in the specific convention scenario is 160 --/
theorem convention_handshakes :
  number_of_handshakes 5 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_l758_75891


namespace NUMINAMATH_CALUDE_lcm_of_4_8_9_10_l758_75841

theorem lcm_of_4_8_9_10 : Nat.lcm 4 (Nat.lcm 8 (Nat.lcm 9 10)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_4_8_9_10_l758_75841


namespace NUMINAMATH_CALUDE_inequality_preservation_l758_75867

theorem inequality_preservation (x y : ℝ) (h : x > y) : x + 5 > y + 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l758_75867


namespace NUMINAMATH_CALUDE_sharons_harvest_l758_75869

theorem sharons_harvest (greg_harvest : ℝ) (difference : ℝ) (sharon_harvest : ℝ) 
  (h1 : greg_harvest = 0.4)
  (h2 : greg_harvest = sharon_harvest + difference)
  (h3 : difference = 0.3) :
  sharon_harvest = 0.1 := by
sorry

end NUMINAMATH_CALUDE_sharons_harvest_l758_75869


namespace NUMINAMATH_CALUDE_expression_evaluation_l758_75897

theorem expression_evaluation : 
  (2020^3 - 3 * 2020^2 * 2021 + 5 * 2020 * 2021^2 - 2021^3 + 4) / (2020 * 2021) = 
  4042 + 3 / (4080420 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l758_75897


namespace NUMINAMATH_CALUDE_inequality_proof_l758_75871

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^2 * (b + c)) + 1 / (b^2 * (c + a)) + 1 / (c^2 * (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l758_75871


namespace NUMINAMATH_CALUDE_savings_calculation_l758_75878

/-- Calculates the total savings for a year given monthly expenses and average monthly income -/
def yearly_savings (expense1 expense2 expense3 : ℕ) (months1 months2 months3 : ℕ) (avg_income : ℕ) : ℕ :=
  let total_expense := expense1 * months1 + expense2 * months2 + expense3 * months3
  let total_income := avg_income * 12
  total_income - total_expense

/-- Proves that the yearly savings is 5200 given the specific expenses and income -/
theorem savings_calculation : yearly_savings 1700 1550 1800 3 4 5 2125 = 5200 := by
  sorry

#eval yearly_savings 1700 1550 1800 3 4 5 2125

end NUMINAMATH_CALUDE_savings_calculation_l758_75878


namespace NUMINAMATH_CALUDE_seats_needed_l758_75851

theorem seats_needed (total_children : ℕ) (children_per_seat : ℕ) 
  (h1 : total_children = 58) 
  (h2 : children_per_seat = 2) : 
  total_children / children_per_seat = 29 := by
sorry

end NUMINAMATH_CALUDE_seats_needed_l758_75851


namespace NUMINAMATH_CALUDE_tan_alpha_value_l758_75839

theorem tan_alpha_value (α : Real) (h : Real.tan (π/4 - α) = 1/5) : Real.tan α = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l758_75839


namespace NUMINAMATH_CALUDE_binomial_coefficient_7_4_l758_75827

theorem binomial_coefficient_7_4 : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_7_4_l758_75827


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l758_75870

/-- The area of the region between two concentric circles, where the diameter of the larger circle
    is twice the diameter of the smaller circle, and the smaller circle has a diameter of 4 units,
    is equal to 12π square units. -/
theorem area_between_concentric_circles (π : ℝ) : 
  let d_small : ℝ := 4
  let r_small : ℝ := d_small / 2
  let r_large : ℝ := 2 * r_small
  let area_small : ℝ := π * r_small^2
  let area_large : ℝ := π * r_large^2
  area_large - area_small = 12 * π :=
by sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l758_75870


namespace NUMINAMATH_CALUDE_quadratic_properties_l758_75837

/-- The quadratic function f(x) = 2x^2 + 4x - 3 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 3

theorem quadratic_properties :
  (∀ x y : ℝ, y = f x → y > f (-1) → x ≠ -1) ∧ 
  (f (-1) = -5) ∧
  (∀ x : ℝ, -2 ≤ x → x ≤ 1 → -5 ≤ f x ∧ f x ≤ 3) ∧
  (∀ x y : ℝ, y = 2 * (x - 1)^2 - 4 ↔ y = f (x - 2) + 1) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_properties_l758_75837


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l758_75898

theorem binomial_coefficient_problem (h1 : Nat.choose 20 12 = 125970)
                                     (h2 : Nat.choose 18 12 = 18564)
                                     (h3 : Nat.choose 19 12 = 50388) :
  Nat.choose 20 13 = 125970 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l758_75898


namespace NUMINAMATH_CALUDE_no_integer_root_pairs_l758_75859

theorem no_integer_root_pairs (n : ℕ) : ¬ ∃ (a b : Fin 5 → ℤ),
  (∀ k : Fin 5, ∃ (x y : ℤ), x^2 + a k * x + b k = 0 ∧ y^2 + a k * y + b k = 0) ∧
  (∀ k : Fin 5, ∃ m : ℤ, a k = 2 * n + 2 * k + 2 ∨ a k = 2 * n + 2 * k + 4) ∧
  (∀ k : Fin 5, ∃ m : ℤ, b k = 2 * n + 2 * k + 2 ∨ b k = 2 * n + 2 * k + 4) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_root_pairs_l758_75859


namespace NUMINAMATH_CALUDE_arithmetic_sequence_minimum_value_l758_75852

/-- Given an arithmetic sequence {a_n} with common difference d ≠ 0,
    where a₁ = 1 and a₁, a₃, a₁₃ form a geometric sequence,
    prove that the minimum value of (2S_n + 16) / (a_n + 3) is 4,
    where S_n is the sum of the first n terms of {a_n}. -/
theorem arithmetic_sequence_minimum_value (d : ℝ) (n : ℕ) :
  d ≠ 0 →
  let a : ℕ → ℝ := λ k => 1 + (k - 1) * d
  let S : ℕ → ℝ := λ k => k * (a 1 + a k) / 2
  (a 3)^2 = (a 1) * (a 13) →
  (∀ k : ℕ, (2 * S k + 16) / (a k + 3) ≥ 4) ∧
  (∃ k : ℕ, (2 * S k + 16) / (a k + 3) = 4) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_minimum_value_l758_75852


namespace NUMINAMATH_CALUDE_min_omega_value_l758_75813

theorem min_omega_value (f : ℝ → ℝ) (ω φ T : ℝ) :
  (∀ x, f x = Real.cos (ω * x + φ)) →
  ω > 0 →
  0 < φ ∧ φ < π →
  (∀ t > 0, f (t + T) = f t) →
  (∀ t > T, ∃ s ∈ Set.Ioo 0 T, f t = f s) →
  f T = Real.sqrt 3 / 2 →
  f (π / 9) = 0 →
  3 ≤ ω ∧ ∀ ω' > 0, (∀ x, Real.cos (ω' * x + φ) = f x) → ω ≤ ω' :=
by sorry

end NUMINAMATH_CALUDE_min_omega_value_l758_75813


namespace NUMINAMATH_CALUDE_coat_drive_total_l758_75812

theorem coat_drive_total (high_school_coats : ℕ) (elementary_school_coats : ℕ) 
  (h1 : high_school_coats = 6922)
  (h2 : elementary_school_coats = 2515) :
  high_school_coats + elementary_school_coats = 9437 := by
  sorry

end NUMINAMATH_CALUDE_coat_drive_total_l758_75812


namespace NUMINAMATH_CALUDE_total_carriages_l758_75850

/-- The number of carriages in each town -/
structure TownCarriages where
  euston : ℕ
  norfolk : ℕ
  norwich : ℕ
  flyingScotsman : ℕ

/-- The conditions given in the problem -/
def problemConditions (t : TownCarriages) : Prop :=
  t.euston = t.norfolk + 20 ∧
  t.norwich = 100 ∧
  t.flyingScotsman = t.norwich + 20 ∧
  t.euston = 130

/-- The theorem to prove -/
theorem total_carriages (t : TownCarriages) 
  (h : problemConditions t) : 
  t.euston + t.norfolk + t.norwich + t.flyingScotsman = 460 :=
by
  sorry

end NUMINAMATH_CALUDE_total_carriages_l758_75850


namespace NUMINAMATH_CALUDE_triangle_count_difference_l758_75866

/-- The number of distinct, incongruent, integer-sided triangles with perimeter n -/
def t (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem triangle_count_difference (n : ℕ) (h : n ≥ 3) :
  (t (2 * n - 1) - t (2 * n) = ⌊(6 : ℚ) / n⌋) ∨
  (t (2 * n - 1) - t (2 * n) = ⌊(6 : ℚ) / n⌋ + 1) :=
sorry

end NUMINAMATH_CALUDE_triangle_count_difference_l758_75866


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l758_75845

theorem roof_dimension_difference (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 4 * width →
  width * length = 900 →
  length - width = 45 := by
sorry

end NUMINAMATH_CALUDE_roof_dimension_difference_l758_75845


namespace NUMINAMATH_CALUDE_scooter_selling_price_l758_75804

/-- Calculates the selling price of a scooter given initial costs and gain percent -/
theorem scooter_selling_price
  (purchase_price : ℝ)
  (repair_cost : ℝ)
  (gain_percent : ℝ)
  (h1 : purchase_price = 4700)
  (h2 : repair_cost = 600)
  (h3 : gain_percent = 9.433962264150944)
  : ∃ (selling_price : ℝ), selling_price = 5800 := by
  sorry

end NUMINAMATH_CALUDE_scooter_selling_price_l758_75804


namespace NUMINAMATH_CALUDE_exercise_book_distribution_l758_75824

theorem exercise_book_distribution (total_books : ℕ) (num_classes : ℕ) 
  (h1 : total_books = 338) (h2 : num_classes = 3) :
  ∃ (books_per_class : ℕ) (books_left : ℕ),
    books_per_class = 112 ∧ 
    books_left = 2 ∧
    total_books = books_per_class * num_classes + books_left :=
by sorry

end NUMINAMATH_CALUDE_exercise_book_distribution_l758_75824


namespace NUMINAMATH_CALUDE_probability_at_least_one_head_three_coins_l758_75814

theorem probability_at_least_one_head_three_coins :
  let p_head : ℝ := 1 / 2
  let p_tail : ℝ := 1 - p_head
  let p_three_tails : ℝ := p_tail ^ 3
  let p_at_least_one_head : ℝ := 1 - p_three_tails
  p_at_least_one_head = 7 / 8 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_head_three_coins_l758_75814


namespace NUMINAMATH_CALUDE_parallelogram_area_l758_75800

/-- The area of a parallelogram with given properties -/
theorem parallelogram_area (s2 : ℝ) (a : ℝ) (h_s2_pos : s2 > 0) (h_a_pos : a > 0) (h_a_lt_180 : a < 180) :
  let s1 := 2 * s2
  let θ := a * π / 180
  2 * s2^2 * Real.sin θ = s1 * s2 * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l758_75800


namespace NUMINAMATH_CALUDE_role_assignment_theorem_l758_75881

def number_of_ways_to_assign_roles (men : Nat) (women : Nat) (male_roles : Nat) (female_roles : Nat) (either_roles : Nat) : Nat :=
  -- Number of ways to assign male roles
  (men.choose male_roles) * (male_roles.factorial) *
  -- Number of ways to assign female roles
  (women.choose female_roles) * (female_roles.factorial) *
  -- Number of ways to assign either-gender roles
  ((men + women - male_roles - female_roles).choose either_roles) * (either_roles.factorial)

theorem role_assignment_theorem :
  number_of_ways_to_assign_roles 6 7 3 3 2 = 1058400 := by
  sorry

end NUMINAMATH_CALUDE_role_assignment_theorem_l758_75881


namespace NUMINAMATH_CALUDE_range_of_m_max_min_distance_exists_line_l_l758_75805

-- Define the circle C
def circle_C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - m = 0

-- Define point A
def point_A (m : ℝ) : ℝ × ℝ := (m, -2)

-- Define the condition for a point to be inside the circle
def inside_circle (x y m : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 < 5 + m

-- Theorem 1
theorem range_of_m (m : ℝ) : 
  (∃ x y, circle_C x y m ∧ inside_circle m (-2) m) → -1 < m ∧ m < 4 :=
sorry

-- Theorem 2
theorem max_min_distance (x y : ℝ) :
  circle_C x y 4 → 4 ≤ (x - 4)^2 + (y - 2)^2 ∧ (x - 4)^2 + (y - 2)^2 ≤ 64 :=
sorry

-- Define the line l
def line_l (k b : ℝ) (x y : ℝ) : Prop := y = x + b

-- Theorem 3
theorem exists_line_l :
  ∃ b, (b = -4 ∨ b = 1) ∧
    ∃ x₁ y₁ x₂ y₂, 
      circle_C x₁ y₁ 4 ∧ circle_C x₂ y₂ 4 ∧
      line_l 1 b x₁ y₁ ∧ line_l 1 b x₂ y₂ ∧
      (x₁ + x₂ = 0) ∧ (y₁ + y₂ = 0) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_max_min_distance_exists_line_l_l758_75805


namespace NUMINAMATH_CALUDE_function_identity_l758_75817

theorem function_identity (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1) : 
    ∀ x : ℝ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l758_75817


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_one_l758_75889

theorem factorization_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_one_l758_75889


namespace NUMINAMATH_CALUDE_pepperoni_coverage_fraction_l758_75825

-- Define the pizza and pepperoni characteristics
def pizza_diameter : ℝ := 16
def pepperoni_count : ℕ := 32
def pepperoni_across_diameter : ℕ := 8
def pepperoni_overlap_fraction : ℝ := 0.25

-- Theorem statement
theorem pepperoni_coverage_fraction :
  let pepperoni_diameter : ℝ := pizza_diameter / pepperoni_across_diameter
  let pepperoni_radius : ℝ := pepperoni_diameter / 2
  let pepperoni_area : ℝ := π * pepperoni_radius^2
  let effective_pepperoni_area : ℝ := pepperoni_area * (1 - pepperoni_overlap_fraction)
  let total_pepperoni_area : ℝ := pepperoni_count * effective_pepperoni_area
  let pizza_area : ℝ := π * (pizza_diameter / 2)^2
  total_pepperoni_area / pizza_area = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_pepperoni_coverage_fraction_l758_75825


namespace NUMINAMATH_CALUDE_q_investment_time_l758_75883

/-- Represents the investment and profit data for two partners -/
structure PartnershipData where
  investment_ratio_p : ℚ
  investment_ratio_q : ℚ
  profit_ratio_p : ℚ
  profit_ratio_q : ℚ
  time_p : ℚ

/-- Calculates the investment time for partner Q given the partnership data -/
def calculate_time_q (data : PartnershipData) : ℚ :=
  (data.profit_ratio_q * data.investment_ratio_p * data.time_p) / (data.profit_ratio_p * data.investment_ratio_q)

/-- Theorem stating that given the problem conditions, Q's investment time is 20 months -/
theorem q_investment_time (data : PartnershipData)
  (h1 : data.investment_ratio_p = 7)
  (h2 : data.investment_ratio_q = 5)
  (h3 : data.profit_ratio_p = 7)
  (h4 : data.profit_ratio_q = 10)
  (h5 : data.time_p = 10) :
  calculate_time_q data = 20 := by
  sorry

end NUMINAMATH_CALUDE_q_investment_time_l758_75883


namespace NUMINAMATH_CALUDE_odd_divisors_iff_perfect_square_l758_75829

/-- A number is a perfect square if and only if it has an odd number of divisors -/
theorem odd_divisors_iff_perfect_square (n : ℕ) : 
  Odd (Nat.card {d : ℕ | d ∣ n}) ↔ ∃ k : ℕ, n = k^2 := by
  sorry


end NUMINAMATH_CALUDE_odd_divisors_iff_perfect_square_l758_75829


namespace NUMINAMATH_CALUDE_binary_of_89_l758_75803

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Theorem: The binary representation of 89 is [true, false, true, true, false, false, true] -/
theorem binary_of_89 :
  toBinary 89 = [true, false, true, true, false, false, true] := by
  sorry

#eval toBinary 89

end NUMINAMATH_CALUDE_binary_of_89_l758_75803


namespace NUMINAMATH_CALUDE_triplets_shirts_l758_75884

/-- The number of shirts Hazel, Razel, and Gazel have in total -/
def total_shirts (hazel razel gazel : ℕ) : ℕ := hazel + razel + gazel

/-- Theorem stating the total number of shirts given the conditions -/
theorem triplets_shirts : 
  ∀ (hazel razel gazel : ℕ),
  hazel = 6 →
  razel = 2 * hazel →
  gazel = razel / 2 - 1 →
  total_shirts hazel razel gazel = 23 := by
sorry

end NUMINAMATH_CALUDE_triplets_shirts_l758_75884


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l758_75873

theorem triangle_abc_properties (a b c A B C : ℝ) (h1 : 0 < A ∧ A < π) 
  (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π) 
  (h5 : a * Real.cos C + Real.sqrt 3 * a * Real.sin C - b - c = 0) 
  (h6 : a = Real.sqrt 13) (h7 : 1/2 * b * c * Real.sin A = 3 * Real.sqrt 3) : 
  A = π/3 ∧ a + b + c = 7 + Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l758_75873


namespace NUMINAMATH_CALUDE_smallest_steps_l758_75888

theorem smallest_steps (n : ℕ) : 
  n > 20 ∧ 
  n % 6 = 5 ∧ 
  n % 7 = 3 →
  n ≥ 59 :=
by sorry

end NUMINAMATH_CALUDE_smallest_steps_l758_75888


namespace NUMINAMATH_CALUDE_minutes_after_midnight_theorem_l758_75835

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

/-- The starting DateTime -/
def startTime : DateTime :=
  { year := 2021, month := 1, day := 1, hour := 0, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : ℕ := 1453

/-- The expected result DateTime -/
def expectedResult : DateTime :=
  { year := 2021, month := 1, day := 2, hour := 0, minute := 13 }

theorem minutes_after_midnight_theorem :
  addMinutes startTime minutesToAdd = expectedResult :=
sorry

end NUMINAMATH_CALUDE_minutes_after_midnight_theorem_l758_75835


namespace NUMINAMATH_CALUDE_f_at_2_l758_75862

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_at_2 (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8) 
  (h2 : f (-2) = 10) : f 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_at_2_l758_75862


namespace NUMINAMATH_CALUDE_steven_owes_jeremy_l758_75853

theorem steven_owes_jeremy (rate : ℚ) (rooms : ℚ) (amount_owed : ℚ) : 
  rate = 9/4 → rooms = 8/5 → amount_owed = rate * rooms → amount_owed = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_steven_owes_jeremy_l758_75853


namespace NUMINAMATH_CALUDE_parabola_triangle_area_l758_75802

/-- Parabola with equation y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → Prop

/-- Line passing through a point with a given slope -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- Point of intersection between a line and a parabola -/
def intersection (p : Parabola) (l : Line) : ℝ × ℝ := sorry

/-- Foot of the perpendicular from a point to a line -/
def perpendicularFoot (point : ℝ × ℝ) (line : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

/-- Area of a triangle given three points -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem parabola_triangle_area 
  (p : Parabola) 
  (l : Line) 
  (h1 : p.equation = fun x y => y^2 = 4*x)
  (h2 : p.focus = (1, 0))
  (h3 : p.directrix = fun x y => x = -1)
  (h4 : l.point = (1, 0))
  (h5 : l.slope = Real.sqrt 3)
  (h6 : (intersection p l).2 > 0) :
  let A := intersection p l
  let K := perpendicularFoot A p.directrix
  triangleArea A K p.focus = 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_parabola_triangle_area_l758_75802


namespace NUMINAMATH_CALUDE_sum_of_roots_arithmetic_sequence_l758_75892

theorem sum_of_roots_arithmetic_sequence (a b c d : ℝ) : 
  0 < c ∧ 0 < b ∧ 0 < a ∧ 
  a > b ∧ b > c ∧ 
  b = a - d ∧ c = a - 2*d ∧ 
  0 < d ∧
  (b^2 - 4*a*c > 0) →
  -(b / a) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_arithmetic_sequence_l758_75892


namespace NUMINAMATH_CALUDE_sphere_radius_with_inscribed_box_l758_75876

theorem sphere_radius_with_inscribed_box (x y z r : ℝ) : 
  x > 0 → y > 0 → z > 0 → r > 0 →
  2 * (x * y + y * z + x * z) = 384 →
  4 * (x + y + z) = 112 →
  (2 * r) ^ 2 = x ^ 2 + y ^ 2 + z ^ 2 →
  r = 10 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_with_inscribed_box_l758_75876


namespace NUMINAMATH_CALUDE_rocky_training_totals_l758_75861

/-- Rocky's training schedule over three days -/
structure TrainingSchedule where
  initial_distance : ℝ
  initial_elevation : ℝ
  day2_distance_multiplier : ℝ
  day2_elevation_multiplier : ℝ
  day3_distance_multiplier : ℝ
  day3_elevation_multiplier : ℝ

/-- Calculate total distance and elevation gain over three days -/
def calculate_totals (schedule : TrainingSchedule) : ℝ × ℝ :=
  let day1_distance := schedule.initial_distance
  let day1_elevation := schedule.initial_elevation
  let day2_distance := day1_distance * schedule.day2_distance_multiplier
  let day2_elevation := day1_elevation * schedule.day2_elevation_multiplier
  let day3_distance := day2_distance * schedule.day3_distance_multiplier
  let day3_elevation := day2_elevation * schedule.day3_elevation_multiplier
  (day1_distance + day2_distance + day3_distance,
   day1_elevation + day2_elevation + day3_elevation)

/-- Theorem stating the total distance and elevation gain for Rocky's training -/
theorem rocky_training_totals :
  let schedule := TrainingSchedule.mk 4 100 2 1.5 4 2
  calculate_totals schedule = (44, 550) := by
  sorry

#eval calculate_totals (TrainingSchedule.mk 4 100 2 1.5 4 2)

end NUMINAMATH_CALUDE_rocky_training_totals_l758_75861


namespace NUMINAMATH_CALUDE_sum_of_three_squares_l758_75868

theorem sum_of_three_squares (s t : ℚ) 
  (h1 : 3 * t + 2 * s = 27)
  (h2 : 2 * t + 3 * s = 25) :
  3 * s = 63 / 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_l758_75868


namespace NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l758_75834

def Q (n : ℕ) : ℚ := (2^(n-1) : ℚ) / (n.factorial * (2*n + 1))

theorem smallest_n_for_Q_less_than_threshold : 
  ∀ k : ℕ, k > 0 → k < 10 → Q k ≥ 1/5000 ∧ Q 10 < 1/5000 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l758_75834


namespace NUMINAMATH_CALUDE_triangle_identity_l758_75856

/-- The triangle operation on pairs of real numbers -/
def triangle (a b c d : ℝ) : ℝ × ℝ := (a*c + b*d, a*d + b*c)

/-- Theorem: If (u,v) △ (x,y) = (u,v) for all real u and v, then (x,y) = (1,0) -/
theorem triangle_identity (x y : ℝ) : 
  (∀ u v : ℝ, triangle u v x y = (u, v)) → (x, y) = (1, 0) := by sorry

end NUMINAMATH_CALUDE_triangle_identity_l758_75856


namespace NUMINAMATH_CALUDE_modified_chessboard_cannot_be_tiled_l758_75854

/-- Represents a chessboard with two opposite corners removed -/
structure ModifiedChessboard :=
  (size : Nat)
  (total_squares : Nat)
  (white_squares : Nat)
  (black_squares : Nat)

/-- Represents a domino tile -/
structure Domino :=
  (length : Nat)
  (width : Nat)

/-- Defines the properties of a standard 8x8 chessboard with opposite corners removed -/
def standard_modified_chessboard : ModifiedChessboard :=
  { size := 8,
    total_squares := 62,
    white_squares := 32,
    black_squares := 30 }

/-- Defines the properties of a 1x2 domino -/
def standard_domino : Domino :=
  { length := 1,
    width := 2 }

/-- Checks if a chessboard can be tiled with dominoes -/
def can_be_tiled (board : ModifiedChessboard) (tile : Domino) : Prop :=
  board.white_squares = board.black_squares

/-- Theorem stating that the modified 8x8 chessboard cannot be tiled with 1x2 dominoes -/
theorem modified_chessboard_cannot_be_tiled :
  ¬(can_be_tiled standard_modified_chessboard standard_domino) :=
by
  sorry


end NUMINAMATH_CALUDE_modified_chessboard_cannot_be_tiled_l758_75854


namespace NUMINAMATH_CALUDE_sports_club_members_l758_75885

theorem sports_club_members (B T Both Neither : ℕ) 
  (hB : B = 48)
  (hT : T = 46)
  (hBoth : Both = 21)
  (hNeither : Neither = 7) :
  (B + T) - Both + Neither = 80 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_members_l758_75885


namespace NUMINAMATH_CALUDE_sum_of_cubes_l758_75893

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) :
  x^3 + y^3 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l758_75893


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l758_75830

/-- Given vectors a and b in ℝ², if |a + 2b| = |a - 2b|, then |b| = 2√5 -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  a = (-1, -2) →
  b.1 = m →
  b.2 = 2 →
  ‖a + 2 • b‖ = ‖a - 2 • b‖ →
  ‖b‖ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l758_75830


namespace NUMINAMATH_CALUDE_f_explicit_formula_b_value_l758_75807

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x^2 + 5

-- Define the function g
def g (b : ℝ) : ℝ → ℝ := fun x ↦ f x - b * x

-- Theorem for the first part
theorem f_explicit_formula : ∀ x : ℝ, f (x - 2) = x^2 - 4*x + 9 := by sorry

-- Theorem for the second part
theorem b_value : 
  ∃ b : ℝ, b = 1/2 ∧ 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 1, g b x ≤ 11/2) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 1, g b x = 11/2) := by sorry

end NUMINAMATH_CALUDE_f_explicit_formula_b_value_l758_75807


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l758_75822

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l758_75822


namespace NUMINAMATH_CALUDE_unique_losses_l758_75821

/-- Represents a participant in the badminton tournament -/
structure Participant where
  id : Fin 16
  gamesWon : Nat
  gamesLost : Nat

/-- The set of all participants in the tournament -/
def Tournament := Fin 16 → Participant

theorem unique_losses (t : Tournament) : 
  (∀ i j : Fin 16, i ≠ j → (t i).gamesWon ≠ (t j).gamesWon) →
  (∀ i : Fin 16, (t i).gamesWon + (t i).gamesLost = 15) →
  (∀ i : Fin 16, (t i).gamesWon < 16) →
  (∀ i j : Fin 16, i ≠ j → (t i).gamesLost ≠ (t j).gamesLost) :=
by sorry

end NUMINAMATH_CALUDE_unique_losses_l758_75821


namespace NUMINAMATH_CALUDE_dan_buys_five_dozens_l758_75847

/-- The number of golf balls in one dozen -/
def balls_per_dozen : ℕ := 12

/-- The total number of golf balls purchased -/
def total_balls : ℕ := 132

/-- The number of dozens Gus buys -/
def gus_dozens : ℕ := 2

/-- The number of golf balls Chris buys -/
def chris_balls : ℕ := 48

/-- Theorem stating that Dan buys 5 dozens of golf balls -/
theorem dan_buys_five_dozens :
  (total_balls - gus_dozens * balls_per_dozen - chris_balls) / balls_per_dozen = 5 :=
sorry

end NUMINAMATH_CALUDE_dan_buys_five_dozens_l758_75847


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l758_75896

theorem necessary_but_not_sufficient :
  let p : ℝ → Prop := λ x ↦ |x + 1| > 2
  let q : ℝ → Prop := λ x ↦ x > 2
  (∀ x, ¬(q x) → ¬(p x)) ∧ (∃ x, ¬(p x) ∧ q x) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l758_75896


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l758_75831

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 + 9 * p - 21 = 0) →
  (3 * q^2 + 9 * q - 21 = 0) →
  (3 * p - 4) * (6 * q - 8) = 122 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l758_75831


namespace NUMINAMATH_CALUDE_negation_of_all_squares_nonnegative_l758_75858

theorem negation_of_all_squares_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_squares_nonnegative_l758_75858


namespace NUMINAMATH_CALUDE_inequality_and_equality_l758_75880

theorem inequality_and_equality (x : ℝ) (h : x > 0) :
  Real.sqrt (1 / (3 * x + 1)) + Real.sqrt (x / (x + 3)) ≥ 1 ∧
  (Real.sqrt (1 / (3 * x + 1)) + Real.sqrt (x / (x + 3)) = 1 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_l758_75880


namespace NUMINAMATH_CALUDE_pipe_speed_ratio_l758_75828

-- Define the rates of pipes A, B, and C
def rate_A : ℚ := 1 / 28
def rate_B : ℚ := 1 / 14
def rate_C : ℚ := 1 / 7

-- Theorem statement
theorem pipe_speed_ratio :
  -- Given conditions
  (rate_A + rate_B + rate_C = 1 / 4) →  -- All pipes fill the tank in 4 hours
  (rate_C = 2 * rate_B) →               -- Pipe C is twice as fast as B
  (rate_A = 1 / 28) →                   -- Pipe A alone takes 28 hours
  -- Conclusion
  (rate_B / rate_A = 2) :=
by sorry


end NUMINAMATH_CALUDE_pipe_speed_ratio_l758_75828


namespace NUMINAMATH_CALUDE_power_equality_l758_75864

theorem power_equality (p : ℕ) : 16^5 = 4^p → p = 10 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l758_75864


namespace NUMINAMATH_CALUDE_catchup_time_correct_l758_75832

/-- Represents a person walking on the triangle -/
structure Walker where
  speed : ℝ  -- speed in meters per minute
  startVertex : ℕ  -- starting vertex (0, 1, or 2)

/-- Represents the triangle and walking scenario -/
structure TriangleWalk where
  sideLength : ℝ
  walkerA : Walker
  walkerB : Walker
  vertexDelay : ℝ  -- delay at each vertex in seconds

/-- Calculates the time when walker A catches up with walker B -/
def catchUpTime (tw : TriangleWalk) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem catchup_time_correct (tw : TriangleWalk) : 
  tw.sideLength = 200 ∧ 
  tw.walkerA = ⟨100, 0⟩ ∧ 
  tw.walkerB = ⟨80, 1⟩ ∧ 
  tw.vertexDelay = 15 → 
  catchUpTime tw = 1470 :=
sorry

end NUMINAMATH_CALUDE_catchup_time_correct_l758_75832


namespace NUMINAMATH_CALUDE_mary_books_checked_out_l758_75833

/-- Calculates the number of books Mary has checked out after a series of transactions. -/
def books_checked_out (initial : ℕ) (first_return : ℕ) (first_checkout : ℕ) (second_return : ℕ) (second_checkout : ℕ) : ℕ :=
  initial - first_return + first_checkout - second_return + second_checkout

/-- Proves that Mary has 12 books checked out after the given transactions. -/
theorem mary_books_checked_out : 
  books_checked_out 5 3 5 2 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mary_books_checked_out_l758_75833


namespace NUMINAMATH_CALUDE_travis_cereal_consumption_l758_75882

/-- Represents the number of boxes of cereal Travis eats per week -/
def boxes_per_week : ℕ := sorry

/-- The cost of one box of cereal in dollars -/
def cost_per_box : ℚ := 3

/-- The number of weeks in a year -/
def weeks_in_year : ℕ := 52

/-- The total amount Travis spends on cereal in a year in dollars -/
def total_spent : ℚ := 312

theorem travis_cereal_consumption :
  boxes_per_week = 2 ∧
  cost_per_box * boxes_per_week * weeks_in_year = total_spent :=
by sorry

end NUMINAMATH_CALUDE_travis_cereal_consumption_l758_75882


namespace NUMINAMATH_CALUDE_box_volume_increase_l758_75816

/-- Given a rectangular box with dimensions l, w, h satisfying certain conditions,
    prove that increasing each dimension by 2 results in a specific new volume -/
theorem box_volume_increase (l w h : ℝ) 
  (hv : l * w * h = 5184)
  (hs : 2 * (l * w + w * h + h * l) = 1944)
  (he : 4 * (l + w + h) = 216) :
  (l + 2) * (w + 2) * (h + 2) = 7352 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l758_75816


namespace NUMINAMATH_CALUDE_unique_solution_cube_difference_l758_75820

theorem unique_solution_cube_difference (n m : ℤ) : 
  (n + 2)^4 - n^4 = m^3 ↔ n = -1 ∧ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_difference_l758_75820


namespace NUMINAMATH_CALUDE_fraction_unchanged_l758_75848

theorem fraction_unchanged (x y : ℝ) (h : y ≠ 0) :
  (3 * (2 * x)) / (2 * (2 * y)) = (3 * x) / (2 * y) := by sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l758_75848


namespace NUMINAMATH_CALUDE_fifth_term_value_l758_75840

theorem fifth_term_value (n : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (h1 : ∀ n, S n = 2 * n^2 + 3 * n - 1)
  (h2 : a 5 = S 5 - S 4) : 
  a 5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l758_75840


namespace NUMINAMATH_CALUDE_tan_five_pi_fourth_l758_75872

theorem tan_five_pi_fourth : Real.tan (5 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_five_pi_fourth_l758_75872


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l758_75818

theorem simplify_and_evaluate (a : ℚ) : 
  let b : ℚ := -1/3
  (a + b)^2 - a * (2*b + a) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l758_75818


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l758_75842

-- Define the sample space
def SampleSpace := Finset (Fin 6 × Fin 6)

-- Define the events
def event_W (s : SampleSpace) : Prop := sorry
def event_1 (s : SampleSpace) : Prop := sorry
def event_2 (s : SampleSpace) : Prop := sorry

-- Define mutually exclusive
def mutually_exclusive (A B : SampleSpace → Prop) : Prop :=
  ∀ s : SampleSpace, ¬(A s ∧ B s)

-- Define complementary
def complementary (A B : SampleSpace → Prop) : Prop :=
  ∀ s : SampleSpace, (A s ∨ B s) ∧ ¬(A s ∧ B s)

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  mutually_exclusive event_W event_1 ∧
  mutually_exclusive event_W event_2 ∧
  ¬complementary event_W event_1 ∧
  ¬complementary event_W event_2 :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l758_75842


namespace NUMINAMATH_CALUDE_planes_not_parallel_l758_75819

/-- Represents a 3D vector --/
structure Vec3 where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space --/
structure Plane where
  normal : Vec3

/-- Check if two planes are parallel --/
def are_parallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ p1.normal = Vec3.mk (k * p2.normal.x) (k * p2.normal.y) (k * p2.normal.z)

theorem planes_not_parallel : ¬ (are_parallel 
  (Plane.mk (Vec3.mk 0 1 3)) 
  (Plane.mk (Vec3.mk 1 0 3))) := by
  sorry

#check planes_not_parallel

end NUMINAMATH_CALUDE_planes_not_parallel_l758_75819


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l758_75843

theorem fractional_equation_solution_range (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ x ≠ 3 ∧ (2 / (x - 3) + (x + m) / (3 - x) = 2)) →
  m ≤ 8 ∧ m ≠ -1 := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l758_75843


namespace NUMINAMATH_CALUDE_watermelon_banana_weights_l758_75836

theorem watermelon_banana_weights :
  ∀ (watermelon_weight banana_weight : ℕ),
    2 * watermelon_weight + banana_weight = 8100 →
    2 * watermelon_weight + 3 * banana_weight = 8300 →
    watermelon_weight = 4000 ∧ banana_weight = 100 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_banana_weights_l758_75836


namespace NUMINAMATH_CALUDE_greatest_4digit_base9_divisible_by_7_l758_75809

/-- Converts a base 9 number to base 10 --/
def base9_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 9 --/
def base10_to_base9 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 4-digit base 9 number --/
def is_4digit_base9 (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 8888

theorem greatest_4digit_base9_divisible_by_7 :
  ∀ n : ℕ, is_4digit_base9 n →
    (base9_to_base10 n) % 7 = 0 →
    n ≤ 8050 :=
by sorry

end NUMINAMATH_CALUDE_greatest_4digit_base9_divisible_by_7_l758_75809


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l758_75857

theorem rectangle_circle_area_ratio 
  (l w r : ℝ) 
  (h1 : 2 * l + 2 * w = 2 * Real.pi * r) 
  (h2 : l = 2 * w) : 
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l758_75857


namespace NUMINAMATH_CALUDE_bus_count_l758_75874

theorem bus_count (total_students : ℕ) (students_per_bus : ℕ) (h1 : total_students = 360) (h2 : students_per_bus = 45) :
  total_students / students_per_bus = 8 :=
by sorry

end NUMINAMATH_CALUDE_bus_count_l758_75874


namespace NUMINAMATH_CALUDE_tony_ken_ratio_l758_75855

def total_amount : ℚ := 5250
def ken_amount : ℚ := 1750

theorem tony_ken_ratio :
  let tony_amount := total_amount - ken_amount
  (tony_amount : ℚ) / ken_amount = 2 := by sorry

end NUMINAMATH_CALUDE_tony_ken_ratio_l758_75855


namespace NUMINAMATH_CALUDE_abc_equals_314_l758_75886

/-- Represents a base-5 number with two digits -/
def BaseFiveNumber (tens : Nat) (ones : Nat) : Nat :=
  5 * tens + ones

/-- Proposition: Given the conditions, ABC = 314 -/
theorem abc_equals_314 
  (A B C : Nat) 
  (h1 : A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0)
  (h2 : A < 5 ∧ B < 5 ∧ C < 5)
  (h3 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h4 : BaseFiveNumber A B + C = BaseFiveNumber C 0)
  (h5 : BaseFiveNumber A B + BaseFiveNumber B A = BaseFiveNumber C C) :
  100 * A + 10 * B + C = 314 :=
sorry

end NUMINAMATH_CALUDE_abc_equals_314_l758_75886


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l758_75811

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  q ≠ 1 →                          -- q ≠ 1 condition
  a 2 = 9 →                        -- a_2 = 9 condition
  a 3 + a 4 = 18 →                 -- a_3 + a_4 = 18 condition
  q = -2 :=                        -- conclusion: q = -2
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l758_75811


namespace NUMINAMATH_CALUDE_equation_implication_l758_75846

theorem equation_implication (x y : ℝ) 
  (h1 : x^2 - 3*x*y + 2*y^2 + x - y = 0)
  (h2 : x^2 - 2*x*y + y^2 - 5*x + 2*y = 0) :
  x*y - 12*x + 15*y = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_implication_l758_75846


namespace NUMINAMATH_CALUDE_direct_proportionality_from_equation_l758_75899

/-- Two real numbers are directly proportional if their ratio is constant -/
def DirectlyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ y = k * x

/-- Given A and B are non-zero real numbers satisfying 3A = 4B, 
    prove that A and B are directly proportional -/
theorem direct_proportionality_from_equation (A B : ℝ) 
    (h1 : 3 * A = 4 * B) (h2 : A ≠ 0) (h3 : B ≠ 0) : 
    DirectlyProportional A B := by
  sorry

end NUMINAMATH_CALUDE_direct_proportionality_from_equation_l758_75899


namespace NUMINAMATH_CALUDE_set_intersection_problem_l758_75863

theorem set_intersection_problem :
  let A : Set ℤ := {-2, -1, 0, 1}
  let B : Set ℤ := {-1, 0, 1, 2}
  A ∩ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l758_75863


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l758_75823

/-- The speed of a boat in still water, given its speed with and against a stream. -/
theorem boat_speed_in_still_water (along_stream speed_against_stream : ℝ) 
  (h1 : along_stream = 15)
  (h2 : speed_against_stream = 5) :
  (along_stream + speed_against_stream) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l758_75823


namespace NUMINAMATH_CALUDE_weightlifting_time_l758_75808

def practice_duration : ℕ := 120  -- 2 hours in minutes

theorem weightlifting_time (shooting_time running_time weightlifting_time : ℕ) :
  shooting_time = practice_duration / 2 →
  running_time + weightlifting_time = practice_duration - shooting_time →
  running_time = 2 * weightlifting_time →
  weightlifting_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_weightlifting_time_l758_75808


namespace NUMINAMATH_CALUDE_parabola_c_value_l758_75826

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on the parabola -/
def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  x = p.a * y^2 + p.b * y + p.c

/-- Checks if (h, k) is the vertex of the parabola -/
def Parabola.hasVertex (p : Parabola) (h k : ℝ) : Prop :=
  h = p.a * k^2 + p.b * k + p.c ∧
  ∀ y, p.a * y^2 + p.b * y + p.c ≤ h

/-- States that the parabola opens downwards -/
def Parabola.opensDownwards (p : Parabola) : Prop :=
  p.a < 0

theorem parabola_c_value
  (p : Parabola)
  (vertex : p.hasVertex 5 3)
  (point : p.contains 7 6)
  (down : p.opensDownwards) :
  p.c = 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l758_75826


namespace NUMINAMATH_CALUDE_homework_problem_distribution_l758_75890

theorem homework_problem_distribution (total : ℕ) 
  (multiple_choice free_response true_false : ℕ) : 
  total = 45 → 
  multiple_choice = 2 * free_response → 
  free_response = true_false + 7 → 
  total = multiple_choice + free_response + true_false → 
  true_false = 6 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_distribution_l758_75890


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l758_75879

/-- Given a circle C defined by the equation x^2 + y^2 - 2x + 6y = 0,
    prove that its center is (1, -3) and its radius is √10 -/
theorem circle_center_and_radius :
  ∀ (x y : ℝ), x^2 + y^2 - 2*x + 6*y = 0 →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -3) ∧
    radius = Real.sqrt 10 ∧
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry


end NUMINAMATH_CALUDE_circle_center_and_radius_l758_75879


namespace NUMINAMATH_CALUDE_digits_of_product_l758_75810

theorem digits_of_product (n : ℕ) : n = 2^10 * 5^7 * 3^2 → (Nat.digits 10 n).length = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_digits_of_product_l758_75810


namespace NUMINAMATH_CALUDE_problem1_simplification_l758_75806

theorem problem1_simplification (x y : ℝ) : 
  y * (4 * x - 3 * y) + (x - 2 * y)^2 = x^2 + y^2 := by sorry

end NUMINAMATH_CALUDE_problem1_simplification_l758_75806


namespace NUMINAMATH_CALUDE_quadratic_no_roots_l758_75895

/-- A quadratic function with no real roots has a coefficient greater than 1 -/
theorem quadratic_no_roots (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + a ≠ 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_roots_l758_75895


namespace NUMINAMATH_CALUDE_social_media_weekly_time_l758_75887

/-- Calculates the weekly time spent on social media given daily phone usage and social media ratio -/
def weekly_social_media_time (daily_phone_time : ℝ) (social_media_ratio : ℝ) : ℝ :=
  daily_phone_time * social_media_ratio * 7

/-- Theorem: Given 8 hours daily phone usage with half on social media, weekly social media time is 28 hours -/
theorem social_media_weekly_time : 
  weekly_social_media_time 8 0.5 = 28 := by
  sorry


end NUMINAMATH_CALUDE_social_media_weekly_time_l758_75887


namespace NUMINAMATH_CALUDE_largest_m_value_l758_75860

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem largest_m_value :
  ∀ m x y : ℕ,
    m ≥ 1000 →
    m < 10000 →
    is_prime x →
    is_prime y →
    is_prime (10 * x + y) →
    x < 10 →
    y < 10 →
    x > y →
    m = x * y * (10 * x + y) →
    m ≤ 1533 :=
sorry

end NUMINAMATH_CALUDE_largest_m_value_l758_75860


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l758_75877

theorem single_elimination_tournament_games (initial_teams : ℕ) (preliminary_games : ℕ) 
  (eliminated_teams : ℕ) (h1 : initial_teams = 24) (h2 : preliminary_games = 4) 
  (h3 : eliminated_teams = 4) :
  preliminary_games + (initial_teams - eliminated_teams - 1) = 23 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l758_75877


namespace NUMINAMATH_CALUDE_square_ending_theorem_l758_75894

theorem square_ending_theorem (n : ℤ) :
  (∀ d : ℕ, d ∈ Finset.range 9 → (n^2 : ℤ) % 10000 ≠ d * 1111) ∧
  ((∃ d : ℕ, d ∈ Finset.range 9 ∧ (n^2 : ℤ) % 1000 = d * 111) → (n^2 : ℤ) % 1000 = 444) :=
by sorry

end NUMINAMATH_CALUDE_square_ending_theorem_l758_75894


namespace NUMINAMATH_CALUDE_total_bones_equals_twelve_l758_75875

/-- The number of bones carried by each dog in a pack of 5 dogs. -/
def DogBones : Fin 5 → ℕ
  | 0 => 3  -- First dog
  | 1 => DogBones 0 - 1  -- Second dog
  | 2 => 2 * DogBones 1  -- Third dog
  | 3 => 1  -- Fourth dog
  | 4 => 2 * DogBones 3  -- Fifth dog

/-- The theorem states that the sum of bones carried by all 5 dogs equals 12. -/
theorem total_bones_equals_twelve :
  (Finset.sum Finset.univ DogBones) = 12 := by
  sorry


end NUMINAMATH_CALUDE_total_bones_equals_twelve_l758_75875


namespace NUMINAMATH_CALUDE_expression_1_equality_expression_2_equality_expression_3_equality_l758_75849

-- Expression 1
theorem expression_1_equality : (-4)^2 - 6 * (4/3) + 2 * (-1)^3 / (-1/2) = 12 := by sorry

-- Expression 2
theorem expression_2_equality : -1^4 - 1/6 * |2 - (-3)^2| = -13/6 := by sorry

-- Expression 3
theorem expression_3_equality (x y : ℝ) (h : |x+2| + (y-1)^2 = 0) :
  2*(3*x^2*y + x*y^2) - 3*(2*x^2*y - x*y) - 2*x*y^2 + 1 = -5 := by sorry

end NUMINAMATH_CALUDE_expression_1_equality_expression_2_equality_expression_3_equality_l758_75849


namespace NUMINAMATH_CALUDE_solution_set_inequality_l758_75801

theorem solution_set_inequality (x : ℝ) :
  (2*x - 3) * (x + 1) < 0 ↔ -1 < x ∧ x < 3/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l758_75801


namespace NUMINAMATH_CALUDE_inequality_implies_identity_or_negation_l758_75838

/-- A function satisfying the given inequality for all real x and y -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2) - f (y^2) ≤ (f x + y) * (x - f y)

/-- The main theorem stating that a function satisfying the inequality
    must be either the identity function or its negation -/
theorem inequality_implies_identity_or_negation (f : ℝ → ℝ) 
  (h : SatisfiesInequality f) : 
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_identity_or_negation_l758_75838


namespace NUMINAMATH_CALUDE_smallest_m_divisible_by_31_l758_75844

theorem smallest_m_divisible_by_31 :
  ∃ (m : ℕ), m = 30 ∧
  (∀ (n : ℕ), n > 0 → 31 ∣ (m + 2^(5*n))) ∧
  (∀ (k : ℕ), k < m → ∃ (n : ℕ), n > 0 ∧ ¬(31 ∣ (k + 2^(5*n)))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_divisible_by_31_l758_75844


namespace NUMINAMATH_CALUDE_cats_remaining_l758_75815

/-- The number of cats remaining after a sale in a pet store -/
theorem cats_remaining (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 19 → house = 45 → sold = 56 → siamese + house - sold = 8 := by
  sorry

end NUMINAMATH_CALUDE_cats_remaining_l758_75815


namespace NUMINAMATH_CALUDE_lottery_winning_probability_l758_75865

/-- The number of balls for MegaBall selection -/
def megaBallCount : ℕ := 30

/-- The number of balls for WinnerBall selection -/
def winnerBallCount : ℕ := 45

/-- The number of WinnerBalls to be drawn -/
def winnerBallDrawCount : ℕ := 6

/-- The probability of winning the lottery -/
def winningProbability : ℚ := 1 / 244351800

/-- Theorem stating the probability of winning the lottery -/
theorem lottery_winning_probability :
  (1 / megaBallCount) * (1 / (winnerBallCount.choose winnerBallDrawCount)) = winningProbability := by
  sorry

end NUMINAMATH_CALUDE_lottery_winning_probability_l758_75865
