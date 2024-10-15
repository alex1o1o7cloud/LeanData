import Mathlib

namespace NUMINAMATH_CALUDE_angle_sum_l1235_123533

theorem angle_sum (α β : Real) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : Real.sin (α - β) = 5/6) (h4 : Real.tan α / Real.tan β = -1/4) :
  α + β = 7 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_l1235_123533


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l1235_123553

theorem pure_imaginary_fraction (a : ℝ) : 
  let z : ℂ := (a - Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = -1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l1235_123553


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l1235_123565

def f (x : ℝ) : ℝ := x^3 + x^2 - 5*x - 5

theorem f_strictly_increasing :
  (∀ x y, x < y ∧ ((x < -5/3 ∧ y < -5/3) ∨ (x > 1 ∧ y > 1)) → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l1235_123565


namespace NUMINAMATH_CALUDE_binomial_sum_l1235_123542

theorem binomial_sum : 
  let p := Nat.choose 20 6
  let q := Nat.choose 20 5
  p + q = 62016 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_l1235_123542


namespace NUMINAMATH_CALUDE_geometric_series_equation_solution_l1235_123516

theorem geometric_series_equation_solution (x : ℝ) : 
  (|x| < 0.5) →
  (∑' n, (2*x)^n = 3.4 - 1.2*x) →
  x = 1/3 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_series_equation_solution_l1235_123516


namespace NUMINAMATH_CALUDE_product_difference_equality_l1235_123584

theorem product_difference_equality : 2012.25 * 2013.75 - 2010.25 * 2015.75 = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_equality_l1235_123584


namespace NUMINAMATH_CALUDE_swan_percentage_among_non_ducks_l1235_123510

theorem swan_percentage_among_non_ducks (geese swan heron duck : ℚ) :
  geese = 1/5 →
  swan = 3/10 →
  heron = 1/4 →
  duck = 1/4 →
  geese + swan + heron + duck = 1 →
  swan / (geese + swan + heron) = 2/5 :=
sorry

end NUMINAMATH_CALUDE_swan_percentage_among_non_ducks_l1235_123510


namespace NUMINAMATH_CALUDE_perpendicular_lines_l1235_123559

theorem perpendicular_lines (b : ℚ) : 
  (∀ x y : ℚ, 2 * x + 3 * y + 4 = 0 → ∃ m₁ : ℚ, y = m₁ * x + (-4/3)) →
  (∀ x y : ℚ, b * x + 3 * y + 4 = 0 → ∃ m₂ : ℚ, y = m₂ * x + (-4/3)) →
  (∃ m₁ m₂ : ℚ, m₁ * m₂ = -1) →
  b = -9/2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l1235_123559


namespace NUMINAMATH_CALUDE_min_value_of_f_l1235_123556

def f (x : ℝ) := 2 * x^3 - 6 * x^2 + 3

theorem min_value_of_f :
  (∀ x ∈ Set.Icc (-2) 2, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc (-2) 2, f x = 3) →
  (∃ x₀ ∈ Set.Icc (-2) 2, ∀ x ∈ Set.Icc (-2) 2, f x₀ ≤ f x) ∧
  (∀ x ∈ Set.Icc (-2) 2, f x ≥ -37) ∧
  (∃ x ∈ Set.Icc (-2) 2, f x = -37) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1235_123556


namespace NUMINAMATH_CALUDE_hash_property_l1235_123599

/-- Definition of operation # for non-negative integers -/
def hash (a b : ℕ) : ℕ := 100 + 4 * b^2 + 8 * a * b

/-- Theorem stating the properties of the hash operation -/
theorem hash_property (a b : ℕ) : 
  hash a b = 100 ∧ a + b = 5 → hash a b = 100 + 4 * b^2 + 8 * a * b := by
  sorry

end NUMINAMATH_CALUDE_hash_property_l1235_123599


namespace NUMINAMATH_CALUDE_f_monotone_range_of_a_l1235_123501

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 5*x + 6

-- Define the property of being monotonically increasing on an interval
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem f_monotone_range_of_a :
  {a : ℝ | MonotonicallyIncreasing (f a) 1 3} = {a | a ≤ -3 ∨ a ≥ -3} :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_range_of_a_l1235_123501


namespace NUMINAMATH_CALUDE_consecutive_product_prime_factors_l1235_123589

theorem consecutive_product_prime_factors (n : ℕ) (hn : n ≥ 1) :
  ∃ x : ℕ+, ∃ p : Fin n → ℕ, 
    (∀ i : Fin n, Prime (p i)) ∧ 
    (∀ i j : Fin n, i ≠ j → p i ≠ p j) ∧
    (∀ i : Fin n, (p i) ∣ (x * (x + 1) + 1)) :=
sorry

end NUMINAMATH_CALUDE_consecutive_product_prime_factors_l1235_123589


namespace NUMINAMATH_CALUDE_right_triangle_sides_l1235_123541

theorem right_triangle_sides : ∃! (a b c : ℕ), 
  ((a = 3 ∧ b = 4 ∧ c = 5) ∨ 
   (a = 2 ∧ b = 3 ∧ c = 4) ∨ 
   (a = 4 ∧ b = 5 ∧ c = 6) ∨ 
   (a = 5 ∧ b = 6 ∧ c = 7)) ∧ 
  a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l1235_123541


namespace NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_solution_l1235_123588

theorem smallest_integer_y (y : ℤ) : (8 - 3 * y < 26) ↔ (-5 ≤ y) := by sorry

theorem smallest_integer_solution : ∃ (y : ℤ), (8 - 3 * y < 26) ∧ (∀ (z : ℤ), z < y → 8 - 3 * z ≥ 26) := by sorry

end NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_solution_l1235_123588


namespace NUMINAMATH_CALUDE_sebastian_orchestra_size_l1235_123527

/-- Represents the number of musicians in each section of the orchestra -/
structure OrchestraSection :=
  (percussion : ℕ)
  (brass : ℕ)
  (strings : ℕ)
  (woodwinds : ℕ)
  (keyboardsAndHarp : ℕ)
  (conductor : ℕ)

/-- Calculates the total number of musicians in the orchestra -/
def totalMusicians (o : OrchestraSection) : ℕ :=
  o.percussion + o.brass + o.strings + o.woodwinds + o.keyboardsAndHarp + o.conductor

/-- The specific orchestra composition as described in the problem -/
def sebastiansOrchestra : OrchestraSection :=
  { percussion := 4
  , brass := 13
  , strings := 18
  , woodwinds := 10
  , keyboardsAndHarp := 3
  , conductor := 1 }

/-- Theorem stating that the total number of musicians in Sebastian's orchestra is 49 -/
theorem sebastian_orchestra_size :
  totalMusicians sebastiansOrchestra = 49 := by
  sorry


end NUMINAMATH_CALUDE_sebastian_orchestra_size_l1235_123527


namespace NUMINAMATH_CALUDE_f_has_root_in_interval_l1235_123572

-- Define the function f(x) = x^3 + 4x - 3
def f (x : ℝ) := x^3 + 4*x - 3

-- State the theorem
theorem f_has_root_in_interval :
  ∃ c ∈ Set.Icc 0 1, f c = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_f_has_root_in_interval_l1235_123572


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l1235_123596

theorem completing_square_quadratic (x : ℝ) :
  x^2 - 2*x = 9 ↔ (x - 1)^2 = 10 := by sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l1235_123596


namespace NUMINAMATH_CALUDE_dinner_bill_proof_l1235_123551

/-- The number of friends in the group -/
def num_friends : ℕ := 10

/-- The additional amount each paying friend contributes to cover the non-paying friend -/
def extra_payment : ℚ := 4

/-- The total bill for the group dinner -/
def total_bill : ℚ := 360

theorem dinner_bill_proof :
  ∃ (individual_share : ℚ),
    (num_friends - 1 : ℚ) * (individual_share + extra_payment) = total_bill :=
by sorry

end NUMINAMATH_CALUDE_dinner_bill_proof_l1235_123551


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1235_123575

/-- The line 2x + 4y + m = 0 is tangent to the parabola y^2 = 16x if and only if m = 32 -/
theorem line_tangent_to_parabola (m : ℝ) :
  (∀ x y : ℝ, 2 * x + 4 * y + m = 0 → y^2 = 16 * x) ∧
  (∃! p : ℝ × ℝ, 2 * p.1 + 4 * p.2 + m = 0 ∧ p.2^2 = 16 * p.1) ↔
  m = 32 := by sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1235_123575


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1235_123567

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 4) (h3 : x ≠ -2) :
  (x^2 + 4*x + 11) / ((x - 1)*(x - 4)*(x + 2)) = 
  (-16/9) / (x - 1) + (35/18) / (x - 4) + (1/6) / (x + 2) := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1235_123567


namespace NUMINAMATH_CALUDE_students_with_dogs_l1235_123558

theorem students_with_dogs 
  (total_students : ℕ) 
  (girls_percentage : ℚ) 
  (boys_percentage : ℚ) 
  (girls_with_dogs_percentage : ℚ) 
  (boys_with_dogs_percentage : ℚ) :
  total_students = 100 →
  girls_percentage = 1/2 →
  boys_percentage = 1/2 →
  girls_with_dogs_percentage = 1/5 →
  boys_with_dogs_percentage = 1/10 →
  (girls_percentage * total_students * girls_with_dogs_percentage +
   boys_percentage * total_students * boys_with_dogs_percentage : ℚ) = 15 := by
sorry

end NUMINAMATH_CALUDE_students_with_dogs_l1235_123558


namespace NUMINAMATH_CALUDE_total_frogs_in_pond_l1235_123549

def frogs_on_lilypads : ℕ := 5
def frogs_on_logs : ℕ := 3
def dozen : ℕ := 12
def baby_frogs_dozens : ℕ := 2

theorem total_frogs_in_pond : 
  frogs_on_lilypads + frogs_on_logs + baby_frogs_dozens * dozen = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_frogs_in_pond_l1235_123549


namespace NUMINAMATH_CALUDE_dog_treat_cost_is_six_l1235_123561

/-- The cost of dog treats for a month -/
def dog_treat_cost (treats_per_day : ℕ) (cost_per_treat : ℚ) (days_in_month : ℕ) : ℚ :=
  (treats_per_day : ℚ) * cost_per_treat * (days_in_month : ℚ)

/-- Theorem: The cost of dog treats for a month under given conditions is $6 -/
theorem dog_treat_cost_is_six :
  dog_treat_cost 2 (1/10) 30 = 6 := by
sorry

end NUMINAMATH_CALUDE_dog_treat_cost_is_six_l1235_123561


namespace NUMINAMATH_CALUDE_baking_powder_yesterday_l1235_123574

def baking_powder_today : ℝ := 0.3
def difference_yesterday : ℝ := 0.1

theorem baking_powder_yesterday : baking_powder_today + difference_yesterday = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_baking_powder_yesterday_l1235_123574


namespace NUMINAMATH_CALUDE_sin_15_cos_15_equals_quarter_l1235_123514

theorem sin_15_cos_15_equals_quarter : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_equals_quarter_l1235_123514


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1235_123532

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |2*x - 8| = 5 - x ↔ x = 13/3 ∨ x = 3 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1235_123532


namespace NUMINAMATH_CALUDE_smallest_total_is_47_l1235_123535

/-- Represents the number of students in each grade --/
structure StudentCounts where
  ninth : ℕ
  seventh : ℕ
  sixth : ℕ

/-- Checks if the given student counts satisfy the required ratios --/
def satisfiesRatios (counts : StudentCounts) : Prop :=
  3 * counts.seventh = 2 * counts.ninth ∧
  7 * counts.sixth = 4 * counts.ninth

/-- The smallest possible total number of students --/
def smallestTotal : ℕ := 47

/-- Theorem stating that the smallest possible total number of students is 47 --/
theorem smallest_total_is_47 :
  ∃ (counts : StudentCounts),
    satisfiesRatios counts ∧
    counts.ninth + counts.seventh + counts.sixth = smallestTotal ∧
    (∀ (other : StudentCounts),
      satisfiesRatios other →
      other.ninth + other.seventh + other.sixth ≥ smallestTotal) :=
  sorry

end NUMINAMATH_CALUDE_smallest_total_is_47_l1235_123535


namespace NUMINAMATH_CALUDE_sin_zero_degrees_l1235_123546

theorem sin_zero_degrees : Real.sin (0 * π / 180) = 0 := by sorry

end NUMINAMATH_CALUDE_sin_zero_degrees_l1235_123546


namespace NUMINAMATH_CALUDE_total_clothing_cost_l1235_123547

def shirt_cost : ℚ := 13.04
def jacket_cost : ℚ := 12.27

theorem total_clothing_cost : shirt_cost + jacket_cost = 25.31 := by
  sorry

end NUMINAMATH_CALUDE_total_clothing_cost_l1235_123547


namespace NUMINAMATH_CALUDE_mauras_remaining_seashells_l1235_123597

/-- The number of seashells Maura found -/
def total_seashells : ℕ := 75

/-- The number of seashells Maura gave to her sister -/
def given_seashells : ℕ := 18

/-- The number of days Maura's family stays at the beach house -/
def beach_days : ℕ := 21

/-- Theorem stating that Maura has 57 seashells left -/
theorem mauras_remaining_seashells :
  total_seashells - given_seashells = 57 := by sorry

end NUMINAMATH_CALUDE_mauras_remaining_seashells_l1235_123597


namespace NUMINAMATH_CALUDE_triangle_inequality_l1235_123555

/-- Given a triangle with semi-perimeter s, circumradius R, and inradius r,
    prove the inequality relating these quantities. -/
theorem triangle_inequality (s R r : ℝ) (hs : s > 0) (hR : R > 0) (hr : r > 0) :
  2 * Real.sqrt (r * (r + 4 * R)) < 2 * s ∧ 
  2 * s ≤ Real.sqrt (4 * (r + 2 * R)^2 + 2 * R^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1235_123555


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l1235_123566

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| > a^2 + a + 1) ↔ -1 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l1235_123566


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1235_123583

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) : 
  (30 / 360 : ℝ) * (2 * π * r₁) = (24 / 360 : ℝ) * (2 * π * r₂) →
  (π * r₁^2) / (π * r₂^2) = 16 / 25 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1235_123583


namespace NUMINAMATH_CALUDE_rod_triangle_theorem_l1235_123571

/-- A triple of natural numbers representing the side lengths of a triangle --/
structure TriangleSides where
  a : ℕ
  b : ℕ
  c : ℕ
  a_le_b : a ≤ b
  b_le_c : b ≤ c

/-- Checks if a natural number is prime --/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- Checks if a TriangleSides forms an isosceles triangle --/
def isIsosceles (t : TriangleSides) : Prop :=
  t.a = t.b ∨ t.b = t.c

/-- The main theorem --/
theorem rod_triangle_theorem :
  ∃! (sol : Finset TriangleSides),
    (∀ t ∈ sol, 
      t.a + t.b + t.c = 25 ∧ 
      isPrime t.a ∧ isPrime t.b ∧ isPrime t.c) ∧
    sol.card = 2 ∧
    (∀ t ∈ sol, isIsosceles t) := by sorry

end NUMINAMATH_CALUDE_rod_triangle_theorem_l1235_123571


namespace NUMINAMATH_CALUDE_cross_section_distance_l1235_123525

/-- Represents a right hexagonal pyramid -/
structure RightHexagonalPyramid where
  /-- Height of the pyramid -/
  height : ℝ
  /-- Side length of the base hexagon -/
  base_side : ℝ

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  /-- Distance from the apex of the pyramid -/
  distance_from_apex : ℝ
  /-- Area of the cross section -/
  area : ℝ

/-- Theorem about the distance of a cross section in a right hexagonal pyramid -/
theorem cross_section_distance
  (pyramid : RightHexagonalPyramid)
  (cs1 cs2 : CrossSection)
  (h_parallel : cs1.distance_from_apex < cs2.distance_from_apex)
  (h_areas : cs1.area = 150 * Real.sqrt 3 ∧ cs2.area = 600 * Real.sqrt 3)
  (h_distance : cs2.distance_from_apex - cs1.distance_from_apex = 8) :
  cs2.distance_from_apex = 16 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_distance_l1235_123525


namespace NUMINAMATH_CALUDE_flour_calculation_l1235_123591

/-- Given a sugar to flour ratio and an amount of sugar, calculate the required amount of flour -/
def flour_amount (sugar_flour_ratio : ℚ) (sugar_amount : ℚ) : ℚ :=
  sugar_amount / sugar_flour_ratio

theorem flour_calculation (sugar_amount : ℚ) :
  sugar_amount = 50 →
  flour_amount (10 / 1) sugar_amount = 5 := by
sorry

end NUMINAMATH_CALUDE_flour_calculation_l1235_123591


namespace NUMINAMATH_CALUDE_max_tax_revenue_l1235_123562

-- Define the market conditions
def supply (P : ℝ) : ℝ := 6 * P - 312
def demand_slope : ℝ := 4
def tax_rate : ℝ := 30
def consumer_price : ℝ := 118

-- Define the demand function
def demand (P : ℝ) : ℝ := 688 - demand_slope * P

-- Define the tax revenue function
def tax_revenue (t : ℝ) : ℝ := (288 - 2.4 * t) * t

-- Theorem statement
theorem max_tax_revenue :
  ∃ (t : ℝ), ∀ (t' : ℝ), tax_revenue t ≥ tax_revenue t' ∧ tax_revenue t = 8640 :=
sorry

end NUMINAMATH_CALUDE_max_tax_revenue_l1235_123562


namespace NUMINAMATH_CALUDE_six_digit_multiple_of_nine_l1235_123548

theorem six_digit_multiple_of_nine :
  ∃ (d : ℕ), d < 10 ∧ (567890 + d) % 9 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_six_digit_multiple_of_nine_l1235_123548


namespace NUMINAMATH_CALUDE_lollipop_sharing_ratio_l1235_123512

theorem lollipop_sharing_ratio : 
  ∀ (total_lollipops : ℕ) (total_cost : ℚ) (shared_cost : ℚ),
  total_lollipops = 12 →
  total_cost = 3 →
  shared_cost = 3/4 →
  (shared_cost / (total_cost / total_lollipops)) / total_lollipops = 1/4 := by
sorry

end NUMINAMATH_CALUDE_lollipop_sharing_ratio_l1235_123512


namespace NUMINAMATH_CALUDE_milk_cost_proof_l1235_123569

def total_cost : ℕ := 42
def banana_cost : ℕ := 12
def bread_cost : ℕ := 9
def apple_cost : ℕ := 14

theorem milk_cost_proof :
  total_cost - (banana_cost + bread_cost + apple_cost) = 7 := by
sorry

end NUMINAMATH_CALUDE_milk_cost_proof_l1235_123569


namespace NUMINAMATH_CALUDE_y_is_odd_square_l1235_123580

def x : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * x (n + 1) - 2 * x n

def y (n : ℕ) : ℤ := x n ^ 2 + 2 ^ (n + 2)

theorem y_is_odd_square (n : ℕ) (h : n > 0) :
  ∃ k : ℤ, Odd k ∧ y n = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_y_is_odd_square_l1235_123580


namespace NUMINAMATH_CALUDE_sin_721_degrees_equals_sin_1_degree_l1235_123505

theorem sin_721_degrees_equals_sin_1_degree :
  Real.sin (721 * π / 180) = Real.sin (π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_721_degrees_equals_sin_1_degree_l1235_123505


namespace NUMINAMATH_CALUDE_realtor_earnings_problem_l1235_123564

/-- A realtor's earnings and house sales problem -/
theorem realtor_earnings_problem 
  (base_salary : ℕ) 
  (commission_rate : ℚ) 
  (num_houses : ℕ) 
  (total_earnings : ℕ) 
  (house_a_cost : ℕ) :
  base_salary = 3000 →
  commission_rate = 2 / 100 →
  num_houses = 3 →
  total_earnings = 8000 →
  house_a_cost = 60000 →
  ∃ (house_b_cost house_c_cost : ℕ),
    house_b_cost = 3 * house_a_cost ∧
    ∃ (subtracted_amount : ℕ),
      house_c_cost = 2 * house_a_cost - subtracted_amount ∧
      house_a_cost + house_b_cost + house_c_cost = 
        (total_earnings - base_salary) / commission_rate ∧
      subtracted_amount = 110000 :=
by sorry

end NUMINAMATH_CALUDE_realtor_earnings_problem_l1235_123564


namespace NUMINAMATH_CALUDE_prime_iff_divisibility_condition_l1235_123590

theorem prime_iff_divisibility_condition (n : ℕ) (h : n ≥ 2) :
  Prime n ↔ ∀ d : ℕ, d > 1 → d ∣ n → (d^2 + n) ∣ (n^2 + d) :=
sorry

end NUMINAMATH_CALUDE_prime_iff_divisibility_condition_l1235_123590


namespace NUMINAMATH_CALUDE_books_per_shelf_l1235_123576

theorem books_per_shelf (total_books : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ)
  (h1 : total_books = 72)
  (h2 : mystery_shelves = 3)
  (h3 : picture_shelves = 5)
  (h4 : ∃ x : ℕ, total_books = x * (mystery_shelves + picture_shelves)) :
  ∃ x : ℕ, x = 9 ∧ total_books = x * (mystery_shelves + picture_shelves) :=
by sorry

end NUMINAMATH_CALUDE_books_per_shelf_l1235_123576


namespace NUMINAMATH_CALUDE_pete_calculation_l1235_123531

theorem pete_calculation (x y z : ℕ+) : 
  (x + y) * z = 14 ∧ 
  x * y + z = 14 → 
  ∃ (s : Finset ℕ+), s.card = 4 ∧ ∀ a : ℕ+, a ∈ s ↔ 
    ∃ (b c : ℕ+), ((a + b) * c = 14 ∧ a * b + c = 14) := by
  sorry

end NUMINAMATH_CALUDE_pete_calculation_l1235_123531


namespace NUMINAMATH_CALUDE_cowbell_coloring_l1235_123577

theorem cowbell_coloring (n : ℕ) (hn : n ≥ 3) :
  ∃ (m : ℕ), m = n + 1 ∧
  (∀ (k : ℕ), k > m → 
    ∃ (f : ℕ → Fin n), 
      ∀ (i : ℕ), (∀ (c : Fin n), ∃ (j : ℕ), j < n + 1 ∧ f ((i + j) % k) = c)) ∧
  (¬ ∃ (f : ℕ → Fin n), 
    ∀ (i : ℕ), (∀ (c : Fin n), ∃ (j : ℕ), j < n + 1 ∧ f ((i + j) % m) = c)) :=
by sorry

end NUMINAMATH_CALUDE_cowbell_coloring_l1235_123577


namespace NUMINAMATH_CALUDE_increased_amount_proof_l1235_123585

theorem increased_amount_proof (x : ℝ) (y : ℝ) (h1 : x > 0) (h2 : x = 3) :
  x + y = 60 * (1 / x) → y = 17 := by
  sorry

end NUMINAMATH_CALUDE_increased_amount_proof_l1235_123585


namespace NUMINAMATH_CALUDE_sin_squared_sum_three_angles_l1235_123557

theorem sin_squared_sum_three_angles (α : ℝ) : 
  (Real.sin (α - Real.pi / 3))^2 + (Real.sin α)^2 + (Real.sin (α + Real.pi / 3))^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_sum_three_angles_l1235_123557


namespace NUMINAMATH_CALUDE_min_product_of_squares_plus_one_l1235_123509

/-- The polynomial P(x) = x^4 + ax^3 + bx^2 + cx + d -/
def P (a b c d x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem min_product_of_squares_plus_one (a b c d : ℝ) (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : b - d ≥ 5)
  (h₂ : P a b c d x₁ = 0)
  (h₃ : P a b c d x₂ = 0)
  (h₄ : P a b c d x₃ = 0)
  (h₅ : P a b c d x₄ = 0) :
  (x₁^2 + 1) * (x₂^2 + 1) * (x₃^2 + 1) * (x₄^2 + 1) ≥ 16 :=
sorry

end NUMINAMATH_CALUDE_min_product_of_squares_plus_one_l1235_123509


namespace NUMINAMATH_CALUDE_sum_of_digits_is_23_l1235_123521

/-- A structure representing a four-digit number with unique digits -/
structure FourDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  d1_pos : d1 > 0
  d2_pos : d2 > 0
  d3_pos : d3 > 0
  d4_pos : d4 > 0
  d1_lt_10 : d1 < 10
  d2_lt_10 : d2 < 10
  d3_lt_10 : d3 < 10
  d4_lt_10 : d4 < 10
  unique : d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4

/-- Theorem stating that for a four-digit number with product of digits 810 and unique digits, the sum of digits is 23 -/
theorem sum_of_digits_is_23 (n : FourDigitNumber) (h : n.d1 * n.d2 * n.d3 * n.d4 = 810) :
  n.d1 + n.d2 + n.d3 + n.d4 = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_is_23_l1235_123521


namespace NUMINAMATH_CALUDE_lapis_share_l1235_123504

def treasure_problem (total_treasure : ℚ) (fonzie_contribution : ℚ) (aunt_bee_contribution : ℚ) (lapis_contribution : ℚ) : Prop :=
  let total_contribution := fonzie_contribution + aunt_bee_contribution + lapis_contribution
  let lapis_fraction := lapis_contribution / total_contribution
  lapis_fraction * total_treasure = 337500

theorem lapis_share :
  treasure_problem 900000 7000 8000 9000 := by
  sorry

#check lapis_share

end NUMINAMATH_CALUDE_lapis_share_l1235_123504


namespace NUMINAMATH_CALUDE_age_difference_l1235_123528

theorem age_difference (alice_age bob_age : ℕ) : 
  alice_age + 5 = 19 →
  alice_age + 6 = 2 * (bob_age + 6) →
  alice_age - bob_age = 10 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l1235_123528


namespace NUMINAMATH_CALUDE_problem_statement_l1235_123598

theorem problem_statement (x y z a b c : ℝ) 
  (h1 : x * y = 2 * a) 
  (h2 : x * z = 3 * b) 
  (h3 : y * z = 4 * c) 
  (h4 : x ≠ 0) 
  (h5 : y ≠ 0) 
  (h6 : z ≠ 0) 
  (h7 : a ≠ 0) 
  (h8 : b ≠ 0) 
  (h9 : c ≠ 0) : 
  2 * x^2 + 3 * y^2 + 4 * z^2 = 12 * b / a + 8 * c / b + 6 * a / c := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1235_123598


namespace NUMINAMATH_CALUDE_unique_N_leads_to_five_l1235_123592

def machine_rule (N : ℕ) : ℕ :=
  if N % 2 = 1 then 2 * N + 2 else N / 2 + 1

def apply_rule_n_times (N : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => N
  | m + 1 => machine_rule (apply_rule_n_times N m)

theorem unique_N_leads_to_five : ∃! N : ℕ, N > 0 ∧ apply_rule_n_times N 6 = 5 ∧ N = 66 := by
  sorry

end NUMINAMATH_CALUDE_unique_N_leads_to_five_l1235_123592


namespace NUMINAMATH_CALUDE_kangaroo_problem_l1235_123594

/-- Represents the number of exchanges required to sort kangaroos -/
def kangaroo_exchanges (total : ℕ) (right_facing : ℕ) (left_facing : ℕ) : ℕ :=
  (right_facing.min 3) * left_facing + (right_facing - 3).max 0 * (left_facing - 2).max 0

/-- Theorem stating that for 10 kangaroos with 6 facing right and 4 facing left, 
    the number of exchanges is 18 -/
theorem kangaroo_problem : 
  kangaroo_exchanges 10 6 4 = 18 := by sorry

end NUMINAMATH_CALUDE_kangaroo_problem_l1235_123594


namespace NUMINAMATH_CALUDE_M_subset_N_l1235_123587

def M : Set ℚ := {x | ∃ k : ℤ, x = k / 2 + 1 / 4}
def N : Set ℚ := {x | ∃ k : ℤ, x = k / 4 + 1 / 2}

theorem M_subset_N : M ⊆ N := by sorry

end NUMINAMATH_CALUDE_M_subset_N_l1235_123587


namespace NUMINAMATH_CALUDE_smallest_a_value_l1235_123544

theorem smallest_a_value (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) 
  (h : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x)) :
  ∀ a' ≥ 0, (∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (17 * ↑x)) → a' ≥ 17 := by
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l1235_123544


namespace NUMINAMATH_CALUDE_kenny_monday_jumping_jacks_l1235_123513

/-- Represents the number of jumping jacks Kenny did on each day of the week. -/
structure WeeklyJumpingJacks where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Calculates the total number of jumping jacks for the week. -/
def totalJumpingJacks (week : WeeklyJumpingJacks) : ℕ :=
  week.sunday + week.monday + week.tuesday + week.wednesday + week.thursday + week.friday + week.saturday

/-- Theorem stating that Kenny must have done 20 jumping jacks on Monday. -/
theorem kenny_monday_jumping_jacks :
  ∃ (this_week : WeeklyJumpingJacks),
    this_week.sunday = 34 ∧
    this_week.tuesday = 0 ∧
    this_week.wednesday = 123 ∧
    this_week.thursday = 64 ∧
    this_week.friday = 23 ∧
    this_week.saturday = 61 ∧
    totalJumpingJacks this_week = 325 ∧
    this_week.monday = 20 := by
  sorry

#check kenny_monday_jumping_jacks

end NUMINAMATH_CALUDE_kenny_monday_jumping_jacks_l1235_123513


namespace NUMINAMATH_CALUDE_tank_A_height_l1235_123581

-- Define the tanks
structure Tank where
  circumference : ℝ
  height : ℝ

-- Define the problem parameters
def tank_A : Tank := { circumference := 8, height := 10 }
def tank_B : Tank := { circumference := 10, height := 8 }

-- Define the capacity ratio
def capacity_ratio : ℝ := 0.8000000000000001

-- Theorem statement
theorem tank_A_height :
  tank_A.height = 10 ∧
  tank_A.circumference = 8 ∧
  tank_B.circumference = 10 ∧
  tank_B.height = 8 ∧
  (tank_A.circumference * tank_A.height) / (tank_B.circumference * tank_B.height) = capacity_ratio :=
by sorry

end NUMINAMATH_CALUDE_tank_A_height_l1235_123581


namespace NUMINAMATH_CALUDE_sum_odd_numbers_eq_square_last_term_eq_2n_minus_1_sum_odd_numbers_40_times_3_eq_4800_l1235_123595

/-- The sum of the first n odd numbers -/
def sum_odd_numbers (n : ℕ) : ℕ := (Finset.range n).sum (fun i => 2 * i + 1)

theorem sum_odd_numbers_eq_square (n : ℕ) : sum_odd_numbers n = n^2 :=
  by sorry

theorem last_term_eq_2n_minus_1 (n : ℕ) : 2 * n - 1 = sum_odd_numbers n - sum_odd_numbers (n - 1) :=
  by sorry

theorem sum_odd_numbers_40_times_3_eq_4800 : 3 * sum_odd_numbers 40 = 4800 :=
  by sorry

end NUMINAMATH_CALUDE_sum_odd_numbers_eq_square_last_term_eq_2n_minus_1_sum_odd_numbers_40_times_3_eq_4800_l1235_123595


namespace NUMINAMATH_CALUDE_triangle_side_area_relation_l1235_123579

/-- Given a triangle with altitudes m₁, m₂, m₃ to sides a, b, c respectively,
    prove the relation between sides and area. -/
theorem triangle_side_area_relation (m₁ m₂ m₃ a b c S : ℝ) 
  (h₁ : m₁ = 20)
  (h₂ : m₂ = 24)
  (h₃ : m₃ = 30)
  (ha : S = a * m₁ / 2)
  (hb : S = b * m₂ / 2)
  (hc : S = c * m₃ / 2) :
  (a / b = 6 / 5 ∧ b / c = 5 / 4) ∧ S = 10 * a ∧ S = 12 * b ∧ S = 15 * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_area_relation_l1235_123579


namespace NUMINAMATH_CALUDE_parabola_and_line_l1235_123570

/-- A parabola with focus F and point A on it -/
structure Parabola where
  p : ℝ
  y₀ : ℝ
  h_p_pos : p > 0
  h_on_parabola : y₀^2 = 2 * p * 2
  h_focus_dist : (2 - p/2)^2 + y₀^2 = 4^2

/-- A line intersecting the parabola -/
structure IntersectingLine (par : Parabola) where
  m : ℝ
  h_not_origin : m ≠ 0
  h_two_points : ∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    (x₁^2 + (2*m - 8)*x₁ + m^2 = 0) ∧ 
    (x₂^2 + (2*m - 8)*x₂ + m^2 = 0)
  h_perpendicular : ∃ x₁ x₂ y₁ y₂, 
    x₁ ≠ x₂ ∧ 
    y₁ = x₁ + m ∧ 
    y₂ = x₂ + m ∧ 
    x₁*x₂ + y₁*y₂ = 0

/-- The main theorem -/
theorem parabola_and_line (par : Parabola) (l : IntersectingLine par) :
  par.p = 4 ∧ l.m = -8 := by sorry

end NUMINAMATH_CALUDE_parabola_and_line_l1235_123570


namespace NUMINAMATH_CALUDE_complex_power_difference_l1235_123554

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) :
  (1 + 2*i)^24 - (1 - 2*i)^24 = 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_power_difference_l1235_123554


namespace NUMINAMATH_CALUDE_three_planes_divide_space_l1235_123573

-- Define a plane in 3D space
def Plane : Type := ℝ × ℝ × ℝ → Prop

-- Define a function to check if three planes intersect pairwise
def intersect_pairwise (p1 p2 p3 : Plane) : Prop := sorry

-- Define a function to check if three lines are mutually parallel
def mutually_parallel_intersections (p1 p2 p3 : Plane) : Prop := sorry

-- Define a function to count the number of parts the space is divided into
def count_parts (p1 p2 p3 : Plane) : ℕ := sorry

-- Theorem statement
theorem three_planes_divide_space :
  ∀ (p1 p2 p3 : Plane),
    intersect_pairwise p1 p2 p3 →
    mutually_parallel_intersections p1 p2 p3 →
    count_parts p1 p2 p3 = 7 :=
by sorry

end NUMINAMATH_CALUDE_three_planes_divide_space_l1235_123573


namespace NUMINAMATH_CALUDE_area_XPQ_is_435_div_48_l1235_123536

/-- Triangle XYZ with points P and Q -/
structure TriangleXYZ where
  /-- Length of side XY -/
  xy : ℝ
  /-- Length of side YZ -/
  yz : ℝ
  /-- Length of side XZ -/
  xz : ℝ
  /-- Distance XP on side XY -/
  xp : ℝ
  /-- Distance XQ on side XZ -/
  xq : ℝ
  /-- xy is positive -/
  xy_pos : 0 < xy
  /-- yz is positive -/
  yz_pos : 0 < yz
  /-- xz is positive -/
  xz_pos : 0 < xz
  /-- xp is positive and less than or equal to xy -/
  xp_bounds : 0 < xp ∧ xp ≤ xy
  /-- xq is positive and less than or equal to xz -/
  xq_bounds : 0 < xq ∧ xq ≤ xz

/-- The area of triangle XPQ in the given configuration -/
def areaXPQ (t : TriangleXYZ) : ℝ := sorry

/-- Theorem stating the area of triangle XPQ is 435/48 for the given configuration -/
theorem area_XPQ_is_435_div_48 (t : TriangleXYZ) 
    (h_xy : t.xy = 8) 
    (h_yz : t.yz = 9) 
    (h_xz : t.xz = 10) 
    (h_xp : t.xp = 3) 
    (h_xq : t.xq = 6) : 
  areaXPQ t = 435 / 48 := by
  sorry

end NUMINAMATH_CALUDE_area_XPQ_is_435_div_48_l1235_123536


namespace NUMINAMATH_CALUDE_smallest_in_consecutive_odd_integers_l1235_123500

/-- A set of consecutive odd integers -/
def ConsecutiveOddIntegers := Set ℤ

/-- The median of a set of integers -/
def median (s : Set ℤ) : ℤ := sorry

/-- The smallest element in a set of integers -/
def smallest (s : Set ℤ) : ℤ := sorry

/-- The largest element in a set of integers -/
def largest (s : Set ℤ) : ℤ := sorry

theorem smallest_in_consecutive_odd_integers 
  (S : ConsecutiveOddIntegers) 
  (h_median : median S = 152) 
  (h_largest : largest S = 163) : 
  smallest S = 138 := by sorry

end NUMINAMATH_CALUDE_smallest_in_consecutive_odd_integers_l1235_123500


namespace NUMINAMATH_CALUDE_fermat_numbers_coprime_l1235_123519

theorem fermat_numbers_coprime (n m : ℕ) (h : n ≠ m) :
  Nat.gcd (2^(2^(n-1)) + 1) (2^(2^(m-1)) + 1) = 1 := by
sorry

end NUMINAMATH_CALUDE_fermat_numbers_coprime_l1235_123519


namespace NUMINAMATH_CALUDE_sarah_ate_36_candies_l1235_123593

/-- The number of candy pieces Sarah ate -/
def candyEaten (initialCandy : ℕ) (piles : ℕ) (piecesPerPile : ℕ) : ℕ :=
  initialCandy - (piles * piecesPerPile)

/-- Proof that Sarah ate 36 pieces of candy -/
theorem sarah_ate_36_candies :
  candyEaten 108 8 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sarah_ate_36_candies_l1235_123593


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1235_123552

-- Define the triangle ABC
theorem triangle_abc_properties (A B C : ℝ) (AB BC : ℝ) :
  AB = Real.sqrt 3 →
  BC = 2 →
  -- Part I
  (Real.cos B = -1/2 → Real.sin C = Real.sqrt 3 / 2) ∧
  -- Part II
  (∃ (lower upper : ℝ), lower = 0 ∧ upper = 2 * Real.pi / 3 ∧
    ∀ (x : ℝ), lower < C ∧ C ≤ upper) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1235_123552


namespace NUMINAMATH_CALUDE_composition_properties_l1235_123517

variable {X Y V : Type*}
variable (f : X → Y) (g : Y → V)

theorem composition_properties :
  ((∀ x₁ x₂ : X, g (f x₁) = g (f x₂) → x₁ = x₂) → (∀ x₁ x₂ : X, f x₁ = f x₂ → x₁ = x₂)) ∧
  ((∀ v : V, ∃ x : X, g (f x) = v) → (∀ v : V, ∃ y : Y, g y = v)) := by
  sorry

end NUMINAMATH_CALUDE_composition_properties_l1235_123517


namespace NUMINAMATH_CALUDE_elixir_combinations_eq_18_l1235_123526

/-- Represents the number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- Represents the number of enchanted gems available. -/
def num_gems : ℕ := 6

/-- Represents the number of incompatible gem-herb pairs. -/
def num_incompatible : ℕ := 6

/-- Calculates the number of ways the sorcerer can prepare the elixir. -/
def num_elixir_combinations : ℕ := num_herbs * num_gems - num_incompatible

/-- Proves that the number of ways to prepare the elixir is 18. -/
theorem elixir_combinations_eq_18 : num_elixir_combinations = 18 := by
  sorry

end NUMINAMATH_CALUDE_elixir_combinations_eq_18_l1235_123526


namespace NUMINAMATH_CALUDE_fraction_problem_l1235_123543

theorem fraction_problem (x : ℝ) : 
  (0.3 * x = 63.0000000000001) → 
  (∃ f : ℝ, f = 0.4 * x + 12 ∧ f = 96) :=
by sorry

end NUMINAMATH_CALUDE_fraction_problem_l1235_123543


namespace NUMINAMATH_CALUDE_seashell_collection_l1235_123524

theorem seashell_collection (joan_daily : ℕ) (jessica_daily : ℕ) (days : ℕ) : 
  joan_daily = 6 → jessica_daily = 8 → days = 7 → 
  (joan_daily + jessica_daily) * days = 98 := by
  sorry

end NUMINAMATH_CALUDE_seashell_collection_l1235_123524


namespace NUMINAMATH_CALUDE_range_of_a_l1235_123503

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 2 3 → (a * x^2 + 2 * y^2) / (x * y) - 1 > 0) → 
  a ∈ Set.Ioi (-1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1235_123503


namespace NUMINAMATH_CALUDE_king_paths_count_l1235_123534

/-- The number of paths for a king on a 7x7 chessboard -/
def numPaths : Fin 7 → Fin 7 → ℕ
| ⟨i, hi⟩, ⟨j, hj⟩ =>
  if i = 3 ∧ j = 3 then 0  -- Central cell (4,4) is forbidden
  else if i = 0 ∨ j = 0 then 1  -- First row and column
  else 
    have hi' : i - 1 < 7 := by sorry
    have hj' : j - 1 < 7 := by sorry
    numPaths ⟨i - 1, hi'⟩ ⟨j, hj⟩ + 
    numPaths ⟨i, hi⟩ ⟨j - 1, hj'⟩ + 
    numPaths ⟨i - 1, hi'⟩ ⟨j - 1, hj'⟩

/-- The theorem stating the number of paths for the king -/
theorem king_paths_count : numPaths ⟨6, by simp⟩ ⟨6, by simp⟩ = 5020 := by
  sorry

end NUMINAMATH_CALUDE_king_paths_count_l1235_123534


namespace NUMINAMATH_CALUDE_right_triangle_area_and_perimeter_l1235_123522

theorem right_triangle_area_and_perimeter :
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  c = 13 →
  a = 5 →
  b > a →
  (1/2 * a * b = 30 ∧ a + b + c = 30) :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_and_perimeter_l1235_123522


namespace NUMINAMATH_CALUDE_math_test_paper_probability_l1235_123537

theorem math_test_paper_probability :
  let total_papers : ℕ := 12
  let math_papers : ℕ := 4
  let probability := math_papers / total_papers
  probability = (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_math_test_paper_probability_l1235_123537


namespace NUMINAMATH_CALUDE_opponent_total_score_l1235_123540

def volleyball_problem (team_scores : List Nat) : Prop :=
  let n := team_scores.length
  n = 6 ∧
  team_scores = [2, 3, 5, 7, 11, 13] ∧
  (∃ lost_scores : List Nat,
    lost_scores.length = 3 ∧
    lost_scores ⊆ team_scores ∧
    (∀ score ∈ lost_scores, ∃ opp_score, opp_score = score + 2)) ∧
  (∃ won_scores : List Nat,
    won_scores.length = 3 ∧
    won_scores ⊆ team_scores ∧
    (∀ score ∈ won_scores, ∃ opp_score, 3 * opp_score = score))

theorem opponent_total_score (team_scores : List Nat) 
  (h : volleyball_problem team_scores) : 
  (List.sum (team_scores.map (λ score => 
    if score ∈ [2, 3, 5] then score + 2 
    else score / 3))) = 25 := by
  sorry

end NUMINAMATH_CALUDE_opponent_total_score_l1235_123540


namespace NUMINAMATH_CALUDE_symmetry_of_point_l1235_123550

def point_symmetric_to_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), -(p.2))

theorem symmetry_of_point :
  let A : ℝ × ℝ := (-1, 2)
  let A' : ℝ × ℝ := point_symmetric_to_origin A
  A' = (1, -2) := by sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l1235_123550


namespace NUMINAMATH_CALUDE_xiao_ming_final_score_l1235_123539

/-- Calculates the final score of a speech contest given individual scores and weights -/
def final_score (speech_image : ℝ) (content : ℝ) (effectiveness : ℝ) 
  (weight_image : ℝ) (weight_content : ℝ) (weight_effectiveness : ℝ) : ℝ :=
  speech_image * weight_image + content * weight_content + effectiveness * weight_effectiveness

/-- Xiao Ming's speech contest scores and weights -/
def xiao_ming_scores : ℝ × ℝ × ℝ := (9, 8, 8)
def xiao_ming_weights : ℝ × ℝ × ℝ := (0.3, 0.4, 0.3)

theorem xiao_ming_final_score :
  final_score xiao_ming_scores.1 xiao_ming_scores.2.1 xiao_ming_scores.2.2
              xiao_ming_weights.1 xiao_ming_weights.2.1 xiao_ming_weights.2.2 = 8.3 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_final_score_l1235_123539


namespace NUMINAMATH_CALUDE_pie_crust_flour_redistribution_l1235_123511

theorem pie_crust_flour_redistribution 
  (initial_crusts : ℕ) 
  (initial_flour_per_crust : ℚ) 
  (new_crusts : ℕ) 
  (total_flour : ℚ) 
  (h1 : initial_crusts = 40)
  (h2 : initial_flour_per_crust = 1 / 8)
  (h3 : new_crusts = 25)
  (h4 : total_flour = initial_crusts * initial_flour_per_crust)
  : (total_flour / new_crusts : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_pie_crust_flour_redistribution_l1235_123511


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l1235_123507

/-- Given a circle C₁ with equation (x+1)² + (y-1)² = 1 and a line L with equation x - y - 1 = 0,
    the circle C₂ symmetric to C₁ about L has equation (x-2)² + (y+2)² = 1 -/
theorem symmetric_circle_equation (x y : ℝ) : 
  (∀ X Y : ℝ, (X + 1)^2 + (Y - 1)^2 = 1 → 
    (X - Y - 1 = 0 → (x + 1 = Y ∧ y - 1 = X) → (x - 2)^2 + (y + 2)^2 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l1235_123507


namespace NUMINAMATH_CALUDE_second_number_20th_row_l1235_123518

/-- The first number in the nth row of the sequence -/
def first_number (n : ℕ) : ℕ := (n + 1)^2 - 1

/-- The second number in the nth row of the sequence -/
def second_number (n : ℕ) : ℕ := first_number n - 1

/-- Theorem stating that the second number in the 20th row is 439 -/
theorem second_number_20th_row : second_number 20 = 439 := by sorry

end NUMINAMATH_CALUDE_second_number_20th_row_l1235_123518


namespace NUMINAMATH_CALUDE_hotel_pricing_l1235_123523

/-- The hotel pricing problem -/
theorem hotel_pricing
  (night_rate : ℝ)
  (night_hours : ℝ)
  (morning_hours : ℝ)
  (initial_money : ℝ)
  (remaining_money : ℝ)
  (h1 : night_rate = 1.5)
  (h2 : night_hours = 6)
  (h3 : morning_hours = 4)
  (h4 : initial_money = 80)
  (h5 : remaining_money = 63)
  : ∃ (morning_rate : ℝ), 
    night_rate * night_hours + morning_rate * morning_hours = initial_money - remaining_money ∧
    morning_rate = 2 := by
  sorry

end NUMINAMATH_CALUDE_hotel_pricing_l1235_123523


namespace NUMINAMATH_CALUDE_y_value_l1235_123545

theorem y_value (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 14) : y = 2 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l1235_123545


namespace NUMINAMATH_CALUDE_cube_root_8000_l1235_123506

theorem cube_root_8000 (c d : ℕ+) (h1 : (8000 : ℝ)^(1/3) = c * d^(1/3)) 
  (h2 : ∀ (k : ℕ+), (8000 : ℝ)^(1/3) = c * k^(1/3) → d ≤ k) : 
  c + d = 21 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_8000_l1235_123506


namespace NUMINAMATH_CALUDE_largest_three_digit_number_satisfying_condition_l1235_123578

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Calculates the sum of digits of a ThreeDigitNumber -/
def ThreeDigitNumber.digitSum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

/-- Checks if a ThreeDigitNumber satisfies the given condition -/
def ThreeDigitNumber.satisfiesCondition (n : ThreeDigitNumber) : Prop :=
  n.toNat = n.digitSum + (2 * n.digitSum)^2

theorem largest_three_digit_number_satisfying_condition :
  ∃ (n : ThreeDigitNumber), n.toNat = 915 ∧
    n.satisfiesCondition ∧
    ∀ (m : ThreeDigitNumber), m.satisfiesCondition → m.toNat ≤ n.toNat :=
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_number_satisfying_condition_l1235_123578


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1235_123560

/-- An even function that is increasing on (-∞, 0] -/
def EvenIncreasingNonPositive (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y)

/-- The theorem statement -/
theorem solution_set_of_inequality 
  (f : ℝ → ℝ) 
  (h : EvenIncreasingNonPositive f) :
  {x : ℝ | f (x - 1) ≥ f 1} = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1235_123560


namespace NUMINAMATH_CALUDE_trajectory_difference_latitude_l1235_123515

/-- The latitude at which the difference in trajectory lengths equals the height difference -/
theorem trajectory_difference_latitude (R h : ℝ) (θ : ℝ) 
  (h_pos : h > 0) 
  (r₁_def : R * Real.cos θ = R * Real.cos θ)
  (r₂_def : (R + h) * Real.cos θ = (R + h) * Real.cos θ)
  (s_def : 2 * Real.pi * (R + h) * Real.cos θ - 2 * Real.pi * R * Real.cos θ = h) :
  θ = Real.arccos (1 / (2 * Real.pi)) := by
  sorry

end NUMINAMATH_CALUDE_trajectory_difference_latitude_l1235_123515


namespace NUMINAMATH_CALUDE_f_even_implies_a_eq_two_l1235_123530

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x * exp(x) / (exp(a*x) - 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

/-- If f(x) = x * exp(x) / (exp(a*x) - 1) is an even function, then a = 2 -/
theorem f_even_implies_a_eq_two (a : ℝ) :
  IsEven (f a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_even_implies_a_eq_two_l1235_123530


namespace NUMINAMATH_CALUDE_polynomial_substitution_l1235_123529

theorem polynomial_substitution (x y : ℝ) :
  y = x + 1 →
  3 * x^3 + 7 * x^2 + 9 * x + 6 = 3 * y^3 - 2 * y^2 + 4 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_substitution_l1235_123529


namespace NUMINAMATH_CALUDE_equidistant_point_coordinates_l1235_123502

/-- A point with coordinates (4-a, 2a+1) that has equal distances to both coordinate axes -/
structure EquidistantPoint where
  a : ℝ
  equal_distance : |4 - a| = |2*a + 1|

theorem equidistant_point_coordinates (P : EquidistantPoint) :
  (P.a = 1 ∧ (4 - P.a, 2*P.a + 1) = (3, 3)) ∨
  (P.a = -5 ∧ (4 - P.a, 2*P.a + 1) = (9, -9)) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_coordinates_l1235_123502


namespace NUMINAMATH_CALUDE_arithmetic_mean_property_l1235_123508

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression (α : Type*) [Add α] [Mul α] where
  a₁ : α  -- First term
  d : α   -- Common difference

variable {α : Type*} [LinearOrderedField α]

/-- The nth term of an arithmetic progression -/
def ArithmeticProgression.nthTerm (ap : ArithmeticProgression α) (n : ℕ) : α :=
  ap.a₁ + (n - 1 : α) * ap.d

/-- Theorem: In an arithmetic progression, any term (starting from the second)
    is the arithmetic mean of two terms equidistant from it. -/
theorem arithmetic_mean_property (ap : ArithmeticProgression α) (k p : ℕ) 
    (h1 : k ≥ 2) (h2 : p > 0) :
  ap.nthTerm k = (ap.nthTerm (k - p) + ap.nthTerm (k + p)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_property_l1235_123508


namespace NUMINAMATH_CALUDE_num_triangles_equals_closest_integer_l1235_123538

/-- The number of distinct triangles in a regular n-gon -/
def num_triangles (n : ℕ) : ℕ := sorry

/-- The integer closest to n^2/12 -/
def closest_integer (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of distinct triangles in a regular n-gon
    is equal to the integer closest to n^2/12 -/
theorem num_triangles_equals_closest_integer (n : ℕ) (h : n ≥ 3) :
  num_triangles n = closest_integer n := by sorry

end NUMINAMATH_CALUDE_num_triangles_equals_closest_integer_l1235_123538


namespace NUMINAMATH_CALUDE_parabola_translation_theorem_l1235_123582

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_translation_theorem :
  let original := Parabola.mk 1 0 (-2)
  let translated := translate_parabola original 3 1
  translated = Parabola.mk 1 (-6) (-3) := by sorry

end NUMINAMATH_CALUDE_parabola_translation_theorem_l1235_123582


namespace NUMINAMATH_CALUDE_specific_trapezoid_dimensions_l1235_123568

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedIsoscelesTrapezoid where
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The angle at the base of the trapezoid -/
  baseAngle : ℝ
  /-- The length of the shorter base -/
  shorterBase : ℝ
  /-- The length of the longer base -/
  longerBase : ℝ
  /-- The length of the legs (equal for isosceles trapezoid) -/
  legLength : ℝ

/-- The theorem about the specific trapezoid -/
theorem specific_trapezoid_dimensions (t : CircumscribedIsoscelesTrapezoid) 
  (h_area : t.area = 8)
  (h_angle : t.baseAngle = π / 6) :
  t.shorterBase = 4 - 2 * Real.sqrt 3 ∧ 
  t.longerBase = 4 + 2 * Real.sqrt 3 ∧ 
  t.legLength = 4 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_dimensions_l1235_123568


namespace NUMINAMATH_CALUDE_hall_covering_expenditure_l1235_123586

/-- Calculates the total expenditure for covering the interior of a rectangular hall with a mat -/
def calculate_expenditure (length width height cost_per_sqm : ℝ) : ℝ :=
  let floor_area := length * width
  let wall_area := 2 * (length * height + width * height)
  let total_area := 2 * floor_area + wall_area
  total_area * cost_per_sqm

/-- Proves that the expenditure for covering a specific hall with a mat is 19000 -/
theorem hall_covering_expenditure :
  calculate_expenditure 20 15 5 20 = 19000 := by
  sorry

end NUMINAMATH_CALUDE_hall_covering_expenditure_l1235_123586


namespace NUMINAMATH_CALUDE_division_with_remainder_l1235_123563

theorem division_with_remainder (m k : ℤ) (h : m ≠ 0) : 
  ∃ (q r : ℤ), mk + 1 = m * q + r ∧ q = k ∧ r = 1 :=
sorry

end NUMINAMATH_CALUDE_division_with_remainder_l1235_123563


namespace NUMINAMATH_CALUDE_range_of_f_l1235_123520

-- Define the function f
def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -8 ≤ y ∧ y ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l1235_123520
