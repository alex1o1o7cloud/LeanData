import Mathlib

namespace NUMINAMATH_CALUDE_circle_area_irrational_l868_86801

/-- If the diameter of a circle is rational, then its area is irrational. -/
theorem circle_area_irrational (d : ℚ) : Irrational (π * (d^2 / 4)) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_irrational_l868_86801


namespace NUMINAMATH_CALUDE_total_pears_picked_l868_86863

theorem total_pears_picked (sara_pears sally_pears : ℕ) 
  (h1 : sara_pears = 45)
  (h2 : sally_pears = 11) :
  sara_pears + sally_pears = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_picked_l868_86863


namespace NUMINAMATH_CALUDE_sequence_property_l868_86862

/-- Given a sequence a_n and S_n where a_{n+1} = 3S_n for all n ≥ 1,
    prove that a_n can be arithmetic but not geometric -/
theorem sequence_property (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, a (n + 1) = 3 * S n) :
    (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
    (¬ ∃ r : ℝ, r ≠ 1 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = r * a n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l868_86862


namespace NUMINAMATH_CALUDE_simplify_expression_l868_86886

variable (R : Type*) [Ring R]
variable (a b c : R)

theorem simplify_expression :
  (12 * a + 35 * b + 17 * c) + (13 * a - 15 * b + 8 * c) - (8 * a + 28 * b - 25 * c) =
  17 * a - 8 * b + 50 * c := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l868_86886


namespace NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l868_86824

theorem remainder_444_power_444_mod_13 : 444^444 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l868_86824


namespace NUMINAMATH_CALUDE_germination_rate_1000_estimated_probability_close_germinable_seeds_weight_l868_86812

/-- Represents the germination data for a batch of seeds -/
structure GerminationData where
  seeds : ℕ
  germinations : ℕ

/-- The germination experiment data -/
def experimentData : List GerminationData := [
  ⟨100, 94⟩,
  ⟨500, 442⟩,
  ⟨800, 728⟩,
  ⟨1000, 902⟩,
  ⟨2000, 1798⟩,
  ⟨5000, 4505⟩
]

/-- Calculates the germination rate for a given GerminationData -/
def germinationRate (data : GerminationData) : ℚ :=
  data.germinations / data.seeds

/-- Theorem stating the germination rate for 1000 seeds -/
theorem germination_rate_1000 :
  ∃ data ∈ experimentData, data.seeds = 1000 ∧ germinationRate data = 902 / 1000 := by sorry

/-- Estimated germination probability -/
def estimatedProbability : ℚ := 9 / 10

/-- Theorem stating the estimated germination probability is close to actual rates -/
theorem estimated_probability_close :
  ∀ data ∈ experimentData, abs (germinationRate data - estimatedProbability) < 1 / 10 := by sorry

/-- Theorem calculating the weight of germinable seeds in 10 kg -/
theorem germinable_seeds_weight (totalWeight : ℚ) :
  totalWeight * estimatedProbability = 9 / 10 * totalWeight := by sorry

end NUMINAMATH_CALUDE_germination_rate_1000_estimated_probability_close_germinable_seeds_weight_l868_86812


namespace NUMINAMATH_CALUDE_benny_spent_34_dollars_l868_86833

/-- Calculates the amount spent on baseball gear given the initial amount and the amount left over. -/
def amount_spent (initial : ℕ) (left_over : ℕ) : ℕ :=
  initial - left_over

/-- Proves that Benny spent 34 dollars on baseball gear. -/
theorem benny_spent_34_dollars (initial : ℕ) (left_over : ℕ) 
    (h1 : initial = 67) (h2 : left_over = 33) : 
    amount_spent initial left_over = 34 := by
  sorry

#eval amount_spent 67 33

end NUMINAMATH_CALUDE_benny_spent_34_dollars_l868_86833


namespace NUMINAMATH_CALUDE_daily_water_evaporation_l868_86879

/-- Given a glass with initial water amount, evaporation period, and total evaporation percentage,
    calculate the amount of water that evaporates each day. -/
theorem daily_water_evaporation
  (initial_water : ℝ)
  (evaporation_period : ℕ)
  (total_evaporation_percentage : ℝ)
  (h1 : initial_water = 25)
  (h2 : evaporation_period = 10)
  (h3 : total_evaporation_percentage = 1.6)
  : (initial_water * total_evaporation_percentage / 100) / evaporation_period = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_daily_water_evaporation_l868_86879


namespace NUMINAMATH_CALUDE_range_of_a_l868_86861

-- Define the sets A and B
def A : Set ℝ := {x | x < -1 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∪ B a = B a → a ∈ Set.Icc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l868_86861


namespace NUMINAMATH_CALUDE_center_of_given_hyperbola_l868_86864

/-- The equation of a hyperbola in the form (ay + b)^2/c^2 - (dx + e)^2/f^2 = 1 --/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- The center of a hyperbola --/
def center (h : Hyperbola) : ℝ × ℝ := sorry

/-- The given hyperbola --/
def given_hyperbola : Hyperbola :=
  { a := 4
    b := 8
    c := 7
    d := 5
    e := -5
    f := 3 }

/-- Theorem: The center of the given hyperbola is (1, -2) --/
theorem center_of_given_hyperbola :
  center given_hyperbola = (1, -2) := by sorry

end NUMINAMATH_CALUDE_center_of_given_hyperbola_l868_86864


namespace NUMINAMATH_CALUDE_polynomial_expansion_l868_86831

theorem polynomial_expansion (z : ℝ) :
  (3 * z^3 + 2 * z^2 - 4 * z + 1) * (4 * z^4 - 3 * z^2 + 2) =
  12 * z^7 + 8 * z^6 - 25 * z^5 - 2 * z^4 + 18 * z^3 + z^2 - 8 * z + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l868_86831


namespace NUMINAMATH_CALUDE_hairstylist_monthly_earnings_l868_86860

/-- Represents the hairstylist's pricing and schedule --/
structure HairstylistData where
  normal_price : ℕ
  special_price : ℕ
  trendy_price : ℕ
  deluxe_price : ℕ
  mwf_normal : ℕ
  mwf_special : ℕ
  mwf_trendy : ℕ
  tth_normal : ℕ
  tth_special : ℕ
  tth_deluxe : ℕ
  weekend_trendy : ℕ
  weekend_deluxe : ℕ
  weeks_per_month : ℕ

/-- Calculates the monthly earnings of the hairstylist --/
def monthlyEarnings (data : HairstylistData) : ℕ :=
  let mwf_daily := data.mwf_normal * data.normal_price + data.mwf_special * data.special_price + data.mwf_trendy * data.trendy_price
  let tth_daily := data.tth_normal * data.normal_price + data.tth_special * data.special_price + data.tth_deluxe * data.deluxe_price
  let weekend_daily := data.weekend_trendy * data.trendy_price + data.weekend_deluxe * data.deluxe_price
  let weekly_total := 3 * mwf_daily + 2 * tth_daily + 2 * weekend_daily
  weekly_total * data.weeks_per_month

/-- Theorem stating the monthly earnings of the hairstylist --/
theorem hairstylist_monthly_earnings :
  let data : HairstylistData := {
    normal_price := 10
    special_price := 15
    trendy_price := 22
    deluxe_price := 30
    mwf_normal := 4
    mwf_special := 3
    mwf_trendy := 1
    tth_normal := 6
    tth_special := 2
    tth_deluxe := 3
    weekend_trendy := 10
    weekend_deluxe := 5
    weeks_per_month := 4
  }
  monthlyEarnings data = 5684 := by
  sorry

end NUMINAMATH_CALUDE_hairstylist_monthly_earnings_l868_86860


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l868_86866

/-- Two circles are tangent internally if the distance between their centers
    is equal to the absolute difference of their radii. -/
def InternallyTangent (r₁ r₂ d : ℝ) : Prop :=
  d = |r₁ - r₂|

/-- The problem statement -/
theorem circles_internally_tangent :
  let r₁ : ℝ := 3  -- radius of circle O₁
  let r₂ : ℝ := 5  -- radius of circle O₂
  let d : ℝ := 2   -- distance between centers
  InternallyTangent r₁ r₂ d :=
by sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_l868_86866


namespace NUMINAMATH_CALUDE_rectangle_longest_side_l868_86818

/-- Given a rectangle with perimeter 240 feet and area equal to 8 times its perimeter,
    the length of its longest side is 96 feet. -/
theorem rectangle_longest_side (l w : ℝ) : 
  l > 0 → w > 0 →
  2 * l + 2 * w = 240 →
  l * w = 8 * (2 * l + 2 * w) →
  max l w = 96 := by
sorry

end NUMINAMATH_CALUDE_rectangle_longest_side_l868_86818


namespace NUMINAMATH_CALUDE_slope_range_for_line_l868_86816

/-- Given a line passing through (1, 1) with y-intercept in (0, 2), its slope is in (-1, 1) -/
theorem slope_range_for_line (l : Set (ℝ × ℝ)) (y_intercept : ℝ) (k : ℝ) : 
  (∀ x y, (x, y) ∈ l ↔ y = k * x + (1 - k)) →  -- Line equation
  (1, 1) ∈ l →  -- Line passes through (1, 1)
  0 < y_intercept ∧ y_intercept < 2 →  -- y-intercept in (0, 2)
  y_intercept = 1 - k →  -- y-intercept calculation
  -1 < k ∧ k < 1 :=  -- Slope is in (-1, 1)
by sorry

end NUMINAMATH_CALUDE_slope_range_for_line_l868_86816


namespace NUMINAMATH_CALUDE_salt_mixture_proof_l868_86802

/-- Proves that adding 50 ounces of 60% salt solution to 50 ounces of 20% salt solution results in a 40% salt solution -/
theorem salt_mixture_proof :
  let initial_volume : ℝ := 50
  let initial_concentration : ℝ := 0.20
  let added_volume : ℝ := 50
  let added_concentration : ℝ := 0.60
  let final_concentration : ℝ := 0.40
  let final_volume : ℝ := initial_volume + added_volume
  let initial_salt : ℝ := initial_volume * initial_concentration
  let added_salt : ℝ := added_volume * added_concentration
  let final_salt : ℝ := initial_salt + added_salt
  (final_salt / final_volume) = final_concentration :=
by sorry


end NUMINAMATH_CALUDE_salt_mixture_proof_l868_86802


namespace NUMINAMATH_CALUDE_conference_handshakes_count_l868_86821

/-- The number of handshakes at a conference of wizards and elves -/
def conference_handshakes (num_wizards num_elves : ℕ) : ℕ :=
  let wizard_handshakes := num_wizards.choose 2
  let elf_wizard_handshakes := num_wizards * num_elves
  wizard_handshakes + elf_wizard_handshakes

/-- Theorem: The total number of handshakes at the conference is 750 -/
theorem conference_handshakes_count :
  conference_handshakes 25 18 = 750 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_count_l868_86821


namespace NUMINAMATH_CALUDE_inverse_function_solution_l868_86828

/-- Given a function g(x) = 1 / (2ax + b), where a and b are non-zero constants and a ≠ b,
    prove that the solution to g^(-1)(x) = 0 is x = 1/b. -/
theorem inverse_function_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  let g : ℝ → ℝ := λ x ↦ 1 / (2 * a * x + b)
  ∃! x, Function.invFun g x = 0 ∧ x = 1 / b :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_solution_l868_86828


namespace NUMINAMATH_CALUDE_max_sphere_in_cones_l868_86891

/-- Right circular cone -/
structure Cone :=
  (base_radius : ℝ)
  (height : ℝ)

/-- Configuration of two intersecting cones -/
structure ConePair :=
  (cone : Cone)
  (intersection_distance : ℝ)

/-- The maximum squared radius of a sphere fitting in both cones -/
def max_sphere_radius_squared (cp : ConePair) : ℝ :=
  sorry

/-- Theorem statement -/
theorem max_sphere_in_cones :
  let cp := ConePair.mk (Cone.mk 5 12) 4
  max_sphere_radius_squared cp = 1600 / 169 := by
  sorry

end NUMINAMATH_CALUDE_max_sphere_in_cones_l868_86891


namespace NUMINAMATH_CALUDE_cheese_problem_l868_86889

theorem cheese_problem (k : ℕ) (h1 : k > 7) : ∃ (initial : ℕ), initial = 11 ∧ 
  (10 : ℚ) / k + 7 * ((5 : ℚ) / k) = initial ∧ (35 : ℕ) % k = 0 := by
  sorry

end NUMINAMATH_CALUDE_cheese_problem_l868_86889


namespace NUMINAMATH_CALUDE_polynomial_factorization_l868_86842

theorem polynomial_factorization :
  ∀ x : ℝ, x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3) * (8*x^2 + x - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l868_86842


namespace NUMINAMATH_CALUDE_teacher_student_ratio_l868_86896

theorem teacher_student_ratio 
  (initial_student_teacher_ratio : ℚ) 
  (current_teachers : ℕ) 
  (student_increase : ℕ) 
  (teacher_increase : ℕ) 
  (new_student_teacher_ratio : ℚ) 
  (h1 : initial_student_teacher_ratio = 50 / current_teachers)
  (h2 : current_teachers = 3)
  (h3 : student_increase = 50)
  (h4 : teacher_increase = 5)
  (h5 : new_student_teacher_ratio = 25)
  (h6 : (initial_student_teacher_ratio * current_teachers + student_increase) / 
        (current_teachers + teacher_increase) = new_student_teacher_ratio) :
  (1 : ℚ) / initial_student_teacher_ratio = 1 / 50 :=
sorry

end NUMINAMATH_CALUDE_teacher_student_ratio_l868_86896


namespace NUMINAMATH_CALUDE_total_pizza_cost_l868_86898

def number_of_pizzas : ℕ := 3
def price_per_pizza : ℕ := 8

theorem total_pizza_cost : number_of_pizzas * price_per_pizza = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_pizza_cost_l868_86898


namespace NUMINAMATH_CALUDE_randy_blocks_difference_l868_86867

/-- Given that Randy has 95 blocks in total, uses 20 blocks for a house and 50 blocks for a tower,
    prove that he used 30 more blocks for the tower than for the house. -/
theorem randy_blocks_difference (total : ℕ) (house : ℕ) (tower : ℕ) 
    (h1 : total = 95)
    (h2 : house = 20)
    (h3 : tower = 50) :
  tower - house = 30 := by
  sorry

end NUMINAMATH_CALUDE_randy_blocks_difference_l868_86867


namespace NUMINAMATH_CALUDE_binary_product_example_l868_86884

/-- Given two binary numbers represented as natural numbers, 
    this function computes their product in binary representation -/
def binary_multiply (a b : ℕ) : ℕ := 
  (a.digits 2).foldl (λ acc d => acc * 2 + d) 0 * 
  (b.digits 2).foldl (λ acc d => acc * 2 + d) 0

/-- Theorem stating that the product of 1101₂ and 1011₂ is 10011011₂ -/
theorem binary_product_example : binary_multiply 13 11 = 155 := by
  sorry

#eval binary_multiply 13 11

end NUMINAMATH_CALUDE_binary_product_example_l868_86884


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l868_86858

theorem sum_of_reciprocals_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 6*x₁ + 6 = 0 → 
  x₂^2 - 6*x₂ + 6 = 0 → 
  x₁ ≠ x₂ → 
  (1/x₁) + (1/x₂) = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l868_86858


namespace NUMINAMATH_CALUDE_quadratic_function_sign_dependence_l868_86856

/-- Given a quadratic function f(x) = x^2 - x + a where f(-m) < 0,
    the sign of f(m+1) cannot be determined without additional information about m. -/
theorem quadratic_function_sign_dependence 
  (a m : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x^2 - x + a)
  (h2 : f (-m) < 0) :
  ∃ m1 m2 : ℝ, f (m1 + 1) > 0 ∧ f (m2 + 1) < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_sign_dependence_l868_86856


namespace NUMINAMATH_CALUDE_rectangle_area_l868_86817

theorem rectangle_area (square_side : ℝ) (circle_radius : ℝ) (rectangle_length : ℝ) (rectangle_breadth : ℝ) : 
  square_side ^ 2 = 16 →
  circle_radius = square_side →
  rectangle_length = 5 * circle_radius →
  rectangle_breadth = 11 →
  rectangle_length * rectangle_breadth = 220 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l868_86817


namespace NUMINAMATH_CALUDE_cuboid_height_l868_86808

/-- Proves that the height of a cuboid with given base area and volume is 7 cm -/
theorem cuboid_height (base_area volume : ℝ) (h_base : base_area = 36) (h_volume : volume = 252) :
  volume / base_area = 7 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_l868_86808


namespace NUMINAMATH_CALUDE_probability_allison_wins_l868_86869

structure Cube where
  faces : List ℕ
  valid : faces.length = 6

def allison_cube : Cube := ⟨List.replicate 6 5, rfl⟩
def brian_cube : Cube := ⟨[1, 2, 3, 4, 5, 6], rfl⟩
def noah_cube : Cube := ⟨[2, 2, 2, 6, 6, 6], rfl⟩

def prob_roll_less_than (n : ℕ) (c : Cube) : ℚ :=
  (c.faces.filter (· < n)).length / c.faces.length

theorem probability_allison_wins : 
  prob_roll_less_than 5 brian_cube * prob_roll_less_than 5 noah_cube = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_allison_wins_l868_86869


namespace NUMINAMATH_CALUDE_five_single_beds_weight_l868_86893

/-- The weight of a single bed in kg -/
def single_bed_weight : ℝ := sorry

/-- The weight of a double bed in kg -/
def double_bed_weight : ℝ := sorry

/-- A double bed is 10 kg heavier than a single bed -/
axiom double_bed_heavier : double_bed_weight = single_bed_weight + 10

/-- The total weight of 2 single beds and 4 double beds is 100 kg -/
axiom total_weight : 2 * single_bed_weight + 4 * double_bed_weight = 100

theorem five_single_beds_weight :
  5 * single_bed_weight = 50 := by sorry

end NUMINAMATH_CALUDE_five_single_beds_weight_l868_86893


namespace NUMINAMATH_CALUDE_negative_double_less_than_self_l868_86826

theorem negative_double_less_than_self (a : ℝ) : a < 0 → 2 * a < a := by sorry

end NUMINAMATH_CALUDE_negative_double_less_than_self_l868_86826


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l868_86815

def isArithmeticProgression (a : Fin 2008 → ℕ) : Prop :=
  ∃ (start d : ℕ), ∀ i : Fin 10, a i = start + i.val * d

theorem exists_valid_coloring :
  ∃ (f : Fin 2008 → Fin 4),
    ∀ (a : Fin 10 → Fin 2008),
      isArithmeticProgression (λ i => (a i).val + 1) →
        ∃ (i j : Fin 10), f (a i) ≠ f (a j) :=
by sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l868_86815


namespace NUMINAMATH_CALUDE_triangle_area_product_l868_86835

theorem triangle_area_product (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ 2 * a * x + 3 * b * y = 24) →
  (1/2 * (24 / (2 * a)) * (24 / (3 * b)) = 12) →
  a * b = 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_product_l868_86835


namespace NUMINAMATH_CALUDE_cylinder_volume_on_sphere_l868_86845

theorem cylinder_volume_on_sphere (h : ℝ) (d : ℝ) : 
  h = 1 → d = 2 → 
  let r := Real.sqrt (1^2 - (d/2)^2)
  (π * r^2 * h) = (3*π)/4 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_on_sphere_l868_86845


namespace NUMINAMATH_CALUDE_jims_journey_distance_l868_86874

/-- The total distance of Jim's journey, given the miles driven and miles left to drive -/
def total_distance (miles_driven : ℕ) (miles_left : ℕ) : ℕ :=
  miles_driven + miles_left

/-- Theorem stating that the total distance of Jim's journey is 1200 miles -/
theorem jims_journey_distance :
  total_distance 384 816 = 1200 := by sorry

end NUMINAMATH_CALUDE_jims_journey_distance_l868_86874


namespace NUMINAMATH_CALUDE_objective_function_range_l868_86885

/-- The objective function z in terms of x and y -/
def z (x y : ℝ) : ℝ := 3 * x + 2 * y

/-- The constraint function s in terms of x and y -/
def s (x y : ℝ) : ℝ := x + y

theorem objective_function_range :
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → 3 ≤ s x y → s x y ≤ 5 →
  9 ≤ z x y ∧ z x y ≤ 15 :=
sorry

end NUMINAMATH_CALUDE_objective_function_range_l868_86885


namespace NUMINAMATH_CALUDE_fejes_toth_inequality_l868_86837

/-- A convex function on [-1, 1] with absolute value at most 1 -/
structure ConvexBoundedFunction :=
  (f : ℝ → ℝ)
  (convex : ConvexOn ℝ (Set.Icc (-1) 1) f)
  (bounded : ∀ x ∈ Set.Icc (-1) 1, |f x| ≤ 1)

/-- The theorem statement -/
theorem fejes_toth_inequality (F : ConvexBoundedFunction) :
  ∃ (a b : ℝ), ∫ x in Set.Icc (-1) 1, |F.f x - (a * x + b)| ≤ 4 - Real.sqrt 8 := by
  sorry

end NUMINAMATH_CALUDE_fejes_toth_inequality_l868_86837


namespace NUMINAMATH_CALUDE_math_book_cost_l868_86819

/-- Proves that the cost of a math book is $4 given the conditions of the book purchase problem -/
theorem math_book_cost (total_books : ℕ) (math_books : ℕ) (history_book_cost : ℕ) (total_cost : ℕ) 
  (h1 : total_books = 90)
  (h2 : math_books = 54)
  (h3 : history_book_cost = 5)
  (h4 : total_cost = 396) :
  ∃ (math_book_cost : ℕ), 
    math_book_cost * math_books + (total_books - math_books) * history_book_cost = total_cost ∧ 
    math_book_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_math_book_cost_l868_86819


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_problem_l868_86852

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of the geometric sequence -/
def a : ℚ := 1/3

/-- The common ratio of the geometric sequence -/
def r : ℚ := 1/3

/-- The sum we're looking for -/
def target_sum : ℚ := 80/243

theorem geometric_sequence_sum_problem :
  ∃ n : ℕ, geometric_sum a r n = target_sum ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_problem_l868_86852


namespace NUMINAMATH_CALUDE_cannot_afford_both_phones_l868_86843

/-- Represents the financial situation of a couple --/
structure FinancialSituation where
  income : ℕ
  expenses : ℕ
  phoneACost : ℕ
  phoneBCost : ℕ

/-- Determines if a couple can afford to buy both phones --/
def canAffordBothPhones (situation : FinancialSituation) : Prop :=
  situation.income - situation.expenses ≥ situation.phoneACost + situation.phoneBCost

/-- The specific financial situation of Alexander and Natalia --/
def alexanderAndNatalia : FinancialSituation :=
  { income := 186000
    expenses := 119000
    phoneACost := 57000
    phoneBCost := 37000 }

/-- Theorem stating that Alexander and Natalia cannot afford both phones --/
theorem cannot_afford_both_phones :
  ¬(canAffordBothPhones alexanderAndNatalia) := by
  sorry


end NUMINAMATH_CALUDE_cannot_afford_both_phones_l868_86843


namespace NUMINAMATH_CALUDE_division_remainder_problem_l868_86880

theorem division_remainder_problem (L S : ℕ) : 
  L - S = 1325 → 
  L = 1650 → 
  ∃ (R : ℕ), L = 5 * S + R ∧ R < S → 
  R = 25 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l868_86880


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l868_86859

theorem root_difference_implies_k_value (k : ℝ) : 
  (∃ r s : ℝ, r^2 + k*r + 8 = 0 ∧ s^2 + k*s + 8 = 0 ∧ 
   (r+7)^2 - k*(r+7) + 8 = 0 ∧ (s+7)^2 - k*(s+7) + 8 = 0) → 
  k = 7 := by
  sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l868_86859


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l868_86829

/-- Given the cost of 3 pens and 5 pencils is Rs. 200, and the cost ratio of one pen to one pencil
    is 5:1, prove that the cost of one dozen pens is Rs. 600. -/
theorem cost_of_dozen_pens (pen_cost pencil_cost : ℚ) : 
  3 * pen_cost + 5 * pencil_cost = 200 →
  pen_cost = 5 * pencil_cost →
  12 * pen_cost = 600 := by
sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l868_86829


namespace NUMINAMATH_CALUDE_red_ball_probability_three_drawers_l868_86846

/-- Represents the contents of a drawer --/
structure Drawer where
  red_balls : ℕ
  white_balls : ℕ

/-- Calculates the probability of drawing a red ball from a drawer --/
def red_ball_probability (d : Drawer) : ℚ :=
  d.red_balls / (d.red_balls + d.white_balls)

/-- The probability of randomly selecting each drawer --/
def drawer_selection_probability : ℚ := 1 / 3

theorem red_ball_probability_three_drawers 
  (left middle right : Drawer)
  (h_left : left = ⟨0, 5⟩)
  (h_middle : middle = ⟨1, 1⟩)
  (h_right : right = ⟨2, 1⟩) :
  drawer_selection_probability * red_ball_probability middle +
  drawer_selection_probability * red_ball_probability right = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_red_ball_probability_three_drawers_l868_86846


namespace NUMINAMATH_CALUDE_cos_pi_plus_two_alpha_l868_86805

/-- 
Given that the terminal side of angle α passes through point (3,4),
prove that cos(π+2α) = -7/25.
-/
theorem cos_pi_plus_two_alpha (α : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ r * Real.cos α = 3 ∧ r * Real.sin α = 4) → 
  Real.cos (π + 2 * α) = -7/25 := by
sorry

end NUMINAMATH_CALUDE_cos_pi_plus_two_alpha_l868_86805


namespace NUMINAMATH_CALUDE_five_dice_same_number_probability_l868_86897

theorem five_dice_same_number_probability : 
  let number_of_dice : ℕ := 5
  let faces_per_die : ℕ := 6
  let total_outcomes : ℕ := faces_per_die ^ number_of_dice
  let favorable_outcomes : ℕ := faces_per_die
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 1296 := by
sorry

end NUMINAMATH_CALUDE_five_dice_same_number_probability_l868_86897


namespace NUMINAMATH_CALUDE_count_triangles_in_dodecagon_l868_86807

/-- The number of triangles that can be formed from the vertices of a dodecagon -/
def triangles_in_dodecagon : ℕ := 220

/-- The number of vertices in a dodecagon -/
def dodecagon_vertices : ℕ := 12

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- Theorem: The number of triangles that can be formed by selecting 3 vertices
    from a 12-vertex polygon is equal to 220 -/
theorem count_triangles_in_dodecagon :
  Nat.choose dodecagon_vertices triangle_vertices = triangles_in_dodecagon := by
  sorry

end NUMINAMATH_CALUDE_count_triangles_in_dodecagon_l868_86807


namespace NUMINAMATH_CALUDE_perpendicular_vectors_difference_magnitude_l868_86803

/-- Given plane vectors a = (-2, k) and b = (2, 4), if a is perpendicular to b, 
    then |a - b| = 5 -/
theorem perpendicular_vectors_difference_magnitude 
  (k : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (-2, k)) 
  (hb : b = (2, 4)) 
  (hperp : a.1 * b.1 + a.2 * b.2 = 0) : 
  ‖a - b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_difference_magnitude_l868_86803


namespace NUMINAMATH_CALUDE_trajectory_equation_tangent_relation_constant_triangle_area_l868_86847

noncomputable def trajectory (x y : ℝ) : Prop :=
  Real.sqrt ((x + Real.sqrt 3)^2 + y^2) + Real.sqrt ((x - Real.sqrt 3)^2 + y^2) = 4

theorem trajectory_equation (x y : ℝ) :
  trajectory x y → x^2/4 + y^2 = 1 := by sorry

theorem tangent_relation (x y k m : ℝ) :
  trajectory x y → y = k*x + m → m^2 = 1 + 4*k^2 := by sorry

theorem constant_triangle_area (x y k m : ℝ) (A B : ℝ × ℝ) :
  trajectory x y →
  y = k*x + m →
  A.1^2/16 + A.2^2/4 = 1 →
  B.1^2/16 + B.2^2/4 = 1 →
  A.2 = k*A.1 + m →
  B.2 = k*B.1 + m →
  (1/2) * |m| * |A.1 - B.1| = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_trajectory_equation_tangent_relation_constant_triangle_area_l868_86847


namespace NUMINAMATH_CALUDE_remaining_amount_is_correct_l868_86895

-- Define the problem parameters
def initial_amount : ℚ := 100
def action_figure_quantity : ℕ := 3
def board_game_quantity : ℕ := 2
def puzzle_set_quantity : ℕ := 4
def action_figure_price : ℚ := 12
def board_game_price : ℚ := 11
def puzzle_set_price : ℚ := 6
def action_figure_discount : ℚ := 0.25
def sales_tax_rate : ℚ := 0.05

-- Define the function to calculate the remaining amount
def calculate_remaining_amount : ℚ :=
  let discounted_action_figure_price := action_figure_price * (1 - action_figure_discount)
  let action_figure_total := discounted_action_figure_price * action_figure_quantity
  let board_game_total := board_game_price * board_game_quantity
  let puzzle_set_total := puzzle_set_price * puzzle_set_quantity
  let subtotal := action_figure_total + board_game_total + puzzle_set_total
  let total_with_tax := subtotal * (1 + sales_tax_rate)
  initial_amount - total_with_tax

-- Theorem statement
theorem remaining_amount_is_correct :
  calculate_remaining_amount = 23.35 := by sorry

end NUMINAMATH_CALUDE_remaining_amount_is_correct_l868_86895


namespace NUMINAMATH_CALUDE_merchant_discount_theorem_l868_86825

/-- Proves that a merchant offering a 10% discount on a 20% markup results in an 8% profit -/
theorem merchant_discount_theorem (cost_price : ℝ) (markup_percentage : ℝ) (profit_percentage : ℝ) 
  (discount_percentage : ℝ) (h1 : markup_percentage = 20) (h2 : profit_percentage = 8) :
  discount_percentage = 10 ↔ 
    cost_price * (1 + markup_percentage / 100) * (1 - discount_percentage / 100) = 
    cost_price * (1 + profit_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_merchant_discount_theorem_l868_86825


namespace NUMINAMATH_CALUDE_x_divisibility_l868_86832

def x : ℕ := 48 + 64 + 192 + 256 + 384 + 768 + 1024

theorem x_divisibility :
  (∃ k : ℕ, x = 4 * k) ∧
  (∃ k : ℕ, x = 16 * k) ∧
  ¬(∀ k : ℕ, x = 64 * k) ∧
  ¬(∀ k : ℕ, x = 128 * k) := by
  sorry

end NUMINAMATH_CALUDE_x_divisibility_l868_86832


namespace NUMINAMATH_CALUDE_perpendicular_planes_l868_86836

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (non_coincident : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes 
  (l : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : parallel l β)
  (h3 : contained_in l α)
  (h4 : non_coincident α β) :
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l868_86836


namespace NUMINAMATH_CALUDE_sachin_rahul_age_difference_l868_86834

theorem sachin_rahul_age_difference :
  ∀ (sachin_age rahul_age : ℝ),
    sachin_age = 38.5 →
    sachin_age / rahul_age = 11 / 9 →
    sachin_age - rahul_age = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_sachin_rahul_age_difference_l868_86834


namespace NUMINAMATH_CALUDE_sum_squares_products_bound_l868_86875

theorem sum_squares_products_bound (a b c d : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : d ≥ 0) (h5 : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
sorry

end NUMINAMATH_CALUDE_sum_squares_products_bound_l868_86875


namespace NUMINAMATH_CALUDE_egyptian_fraction_decomposition_l868_86872

theorem egyptian_fraction_decomposition (n : ℕ) (h : n ≥ 5 ∧ Odd n) :
  (2 : ℚ) / 11 = 1 / 6 + 1 / 66 ∧
  (2 : ℚ) / n = 1 / ((n + 1) / 2) + 1 / (n * (n + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_egyptian_fraction_decomposition_l868_86872


namespace NUMINAMATH_CALUDE_partition_cases_num_partitions_formula_l868_86850

/-- The number of partitions of a set with n+1 elements into n subsets -/
def num_partitions (n : ℕ) : ℕ := (2^n - 1)^(n+1)

/-- Theorem stating the number of partitions for specific cases -/
theorem partition_cases :
  (num_partitions 2 = 3^3) ∧
  (num_partitions 3 = 7^4) ∧
  (num_partitions 4 = 15^5) := by sorry

/-- Main theorem: The number of partitions of a set with n+1 elements into n subsets is (2^n - 1)^(n+1) -/
theorem num_partitions_formula (n : ℕ) :
  num_partitions n = (2^n - 1)^(n+1) := by sorry

end NUMINAMATH_CALUDE_partition_cases_num_partitions_formula_l868_86850


namespace NUMINAMATH_CALUDE_biology_score_calculation_l868_86851

def math_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def english_score : ℕ := 67
def average_score : ℕ := 69
def total_subjects : ℕ := 5

theorem biology_score_calculation :
  let known_subjects_total := math_score + science_score + social_studies_score + english_score
  let all_subjects_total := average_score * total_subjects
  all_subjects_total - known_subjects_total = 55 := by
sorry

end NUMINAMATH_CALUDE_biology_score_calculation_l868_86851


namespace NUMINAMATH_CALUDE_integer_pairs_sum_reciprocals_l868_86854

theorem integer_pairs_sum_reciprocals (x y : ℤ) : 
  x ≤ y ∧ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 4 ↔ 
  (x = -4 ∧ y = 2) ∨ 
  (x = -12 ∧ y = 3) ∨ 
  (x = 5 ∧ y = 20) ∨ 
  (x = 6 ∧ y = 12) ∨ 
  (x = 8 ∧ y = 8) := by
sorry

end NUMINAMATH_CALUDE_integer_pairs_sum_reciprocals_l868_86854


namespace NUMINAMATH_CALUDE_equation_solution_l868_86820

theorem equation_solution : 
  ∃! y : ℚ, (7 * y - 2) / (y + 4) - 5 / (y + 4) = 2 / (y + 4) ∧ y = 9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l868_86820


namespace NUMINAMATH_CALUDE_rectangular_garden_area_l868_86855

/-- Proves that the area of a rectangular garden with width 14 meters and length three times its width is 588 square meters. -/
theorem rectangular_garden_area :
  ∀ (width length area : ℝ),
    width = 14 →
    length = 3 * width →
    area = length * width →
    area = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_area_l868_86855


namespace NUMINAMATH_CALUDE_first_player_wins_l868_86887

/-- Represents the state of the game -/
structure GameState :=
  (bags : Fin 2008 → ℕ)

/-- The game rules -/
def gameRules (state : GameState) (bagNumber : Fin 2008) (frogsLeft : ℕ) : GameState :=
  { bags := λ i => if i < bagNumber then state.bags i
                   else if i = bagNumber then frogsLeft
                   else min (state.bags i) frogsLeft }

/-- Initial game state -/
def initialState : GameState :=
  { bags := λ _ => 2008 }

/-- Checks if the game is over (only one frog left in bag 1) -/
def isGameOver (state : GameState) : Prop :=
  state.bags 1 = 1 ∧ ∀ i > 1, state.bags i ≤ 1

/-- Theorem: The first player has a winning strategy -/
theorem first_player_wins :
  ∃ (strategy : GameState → Fin 2008 × ℕ),
    ∀ (opponent_move : Fin 2008 × ℕ),
      let (bag, frogs) := strategy initialState
      let state1 := gameRules initialState bag frogs
      let (opponentBag, opponentFrogs) := opponent_move
      let state2 := gameRules state1 opponentBag opponentFrogs
      ¬isGameOver state2 →
        ∃ (next_move : Fin 2008 × ℕ),
          let (nextBag, nextFrogs) := next_move
          let state3 := gameRules state2 nextBag nextFrogs
          isGameOver state3 :=
sorry


end NUMINAMATH_CALUDE_first_player_wins_l868_86887


namespace NUMINAMATH_CALUDE_job_completion_time_l868_86890

/-- Given two workers can finish a job in 15 days and a third worker can finish the job in 30 days,
    prove that all three workers together can finish the job in 10 days. -/
theorem job_completion_time 
  (work_rate_ab : ℝ) 
  (work_rate_c : ℝ) 
  (h1 : work_rate_ab = 1 / 15) 
  (h2 : work_rate_c = 1 / 30) : 
  1 / (work_rate_ab + work_rate_c) = 10 := by
  sorry

#check job_completion_time

end NUMINAMATH_CALUDE_job_completion_time_l868_86890


namespace NUMINAMATH_CALUDE_final_amount_calculation_l868_86822

-- Define the variables
def initial_amount : ℕ := 45
def amount_spent : ℕ := 20
def additional_amount : ℕ := 46

-- Define the theorem
theorem final_amount_calculation :
  initial_amount - amount_spent + additional_amount = 71 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_calculation_l868_86822


namespace NUMINAMATH_CALUDE_fruit_vendor_sales_l868_86892

/-- Calculates the total sales for a fruit vendor given the prices and quantities sold --/
theorem fruit_vendor_sales
  (apple_price : ℚ)
  (orange_price : ℚ)
  (morning_apples : ℕ)
  (morning_oranges : ℕ)
  (afternoon_apples : ℕ)
  (afternoon_oranges : ℕ)
  (h1 : apple_price = 3/2)
  (h2 : orange_price = 1)
  (h3 : morning_apples = 40)
  (h4 : morning_oranges = 30)
  (h5 : afternoon_apples = 50)
  (h6 : afternoon_oranges = 40) :
  let morning_sales := apple_price * morning_apples + orange_price * morning_oranges
  let afternoon_sales := apple_price * afternoon_apples + orange_price * afternoon_oranges
  morning_sales + afternoon_sales = 205 :=
by sorry

end NUMINAMATH_CALUDE_fruit_vendor_sales_l868_86892


namespace NUMINAMATH_CALUDE_prime_sum_square_fourth_power_l868_86868

theorem prime_sum_square_fourth_power : 
  ∀ p q r : ℕ, 
    Prime p → Prime q → Prime r → 
    p + q^2 = r^4 → 
    p = 7 ∧ q = 3 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_square_fourth_power_l868_86868


namespace NUMINAMATH_CALUDE_field_length_is_180_l868_86853

/-- Represents a rectangular field with a surrounding path -/
structure FieldWithPath where
  fieldLength : ℝ
  fieldWidth : ℝ
  pathWidth : ℝ

/-- Calculates the area of the path around the field -/
def pathArea (f : FieldWithPath) : ℝ :=
  (f.fieldLength + 2 * f.pathWidth) * (f.fieldWidth + 2 * f.pathWidth) - f.fieldLength * f.fieldWidth

/-- Theorem: If a rectangular field has width 55m, a surrounding path of 2.5m width, 
    and the path area is 1200 sq m, then the field length is 180m -/
theorem field_length_is_180 (f : FieldWithPath) 
    (h1 : f.fieldWidth = 55)
    (h2 : f.pathWidth = 2.5)
    (h3 : pathArea f = 1200) : 
  f.fieldLength = 180 := by
  sorry


end NUMINAMATH_CALUDE_field_length_is_180_l868_86853


namespace NUMINAMATH_CALUDE_larger_number_proof_l868_86865

theorem larger_number_proof (A B : ℕ+) : 
  (Nat.gcd A B = 20) → 
  (∃ (k : ℕ+), Nat.lcm A B = 20 * 21 * 23 * k) → 
  (max A B = 460) :=
by sorry

end NUMINAMATH_CALUDE_larger_number_proof_l868_86865


namespace NUMINAMATH_CALUDE_binary_111011001001_equals_3785_l868_86814

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_111011001001_equals_3785 :
  binary_to_decimal [true, false, false, true, false, false, true, true, false, true, true, true] = 3785 := by
  sorry

end NUMINAMATH_CALUDE_binary_111011001001_equals_3785_l868_86814


namespace NUMINAMATH_CALUDE_salary_comparison_l868_86811

/-- Given salaries of A, B, and C with specified relationships, prove the percentage differences -/
theorem salary_comparison (a b c : ℝ) 
  (h1 : a = b * 0.8)  -- A's salary is 20% less than B's
  (h2 : c = a * 1.3)  -- C's salary is 30% more than A's
  : (b - a) / a = 0.25 ∧ (c - b) / b = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_salary_comparison_l868_86811


namespace NUMINAMATH_CALUDE_sum_of_even_coefficients_l868_86876

theorem sum_of_even_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ) :
  (∀ x : ℝ, (x + 1)^4 * (x + 4)^8 = a + a₁*(x + 3) + a₂*(x + 3)^2 + a₃*(x + 3)^3 + 
    a₄*(x + 3)^4 + a₅*(x + 3)^5 + a₆*(x + 3)^6 + a₇*(x + 3)^7 + a₈*(x + 3)^8 + 
    a₉*(x + 3)^9 + a₁₀*(x + 3)^10 + a₁₁*(x + 3)^11 + a₁₂*(x + 3)^12) →
  a₂ + a₄ + a₆ + a₈ + a₁₀ + a₁₂ = 112 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_coefficients_l868_86876


namespace NUMINAMATH_CALUDE_number_equation_solution_l868_86838

theorem number_equation_solution : ∃ x : ℝ, 3 * x + 4 = 19 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l868_86838


namespace NUMINAMATH_CALUDE_disjunction_false_l868_86857

-- Define proposition p
def prop_p (a b : ℝ) : Prop := (a * b > 0) → (|a| + |b| > |a + b|)

-- Define proposition q
def prop_q (a b c : ℝ) : Prop := (c > a^2 + b^2) → (c > 2*a*b)

-- Theorem statement
theorem disjunction_false :
  ¬(∀ a b : ℝ, prop_p a b ∨ ¬(∀ c : ℝ, prop_q a b c)) :=
sorry

end NUMINAMATH_CALUDE_disjunction_false_l868_86857


namespace NUMINAMATH_CALUDE_quarter_count_l868_86844

theorem quarter_count (total : ℕ) (quarters : ℕ) (dimes : ℕ) : 
  total = 77 →
  total = quarters + dimes →
  total - quarters = 48 →
  quarters = 29 := by
sorry

end NUMINAMATH_CALUDE_quarter_count_l868_86844


namespace NUMINAMATH_CALUDE_double_discount_price_l868_86804

/-- Proves that if a price P is discounted twice by 25% and the final price is $15, then the original price P is equal to $26.67 -/
theorem double_discount_price (P : ℝ) : 
  (0.75 * (0.75 * P) = 15) → P = 26.67 := by
sorry

end NUMINAMATH_CALUDE_double_discount_price_l868_86804


namespace NUMINAMATH_CALUDE_john_video_release_l868_86899

/-- Calculates the total minutes of video released per week by John --/
def total_video_minutes_per_week (short_video_length : ℕ) (long_video_multiplier : ℕ) (short_videos_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  let long_video_length := short_video_length * long_video_multiplier
  let total_minutes_per_day := short_video_length * short_videos_per_day + long_video_length
  total_minutes_per_day * days_per_week

/-- Theorem stating that John releases 112 minutes of video per week --/
theorem john_video_release : 
  total_video_minutes_per_week 2 6 2 7 = 112 := by
  sorry


end NUMINAMATH_CALUDE_john_video_release_l868_86899


namespace NUMINAMATH_CALUDE_f_has_max_and_min_l868_86877

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a+2)*x + 1

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*(a+2)

/-- Theorem stating the range of a for which f(x) has both a maximum and a minimum -/
theorem f_has_max_and_min (a : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f_derivative a x₁ = 0 ∧ f_derivative a x₂ = 0) ↔ 
  (a < -1 ∨ a > 2) :=
sorry

end NUMINAMATH_CALUDE_f_has_max_and_min_l868_86877


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l868_86883

def set_A : Set ℝ := {x | 2 * x - 1 ≤ 0}
def set_B : Set ℝ := {x | 1 / x > 1}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x : ℝ | 0 < x ∧ x ≤ 1/2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l868_86883


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l868_86882

theorem unique_solution_to_equation :
  ∃! z : ℝ, (z + 2)^4 + (2 - z)^4 = 258 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l868_86882


namespace NUMINAMATH_CALUDE_right_triangle_angle_bisector_l868_86827

theorem right_triangle_angle_bisector (DE DF : ℝ) (h_DE : DE = 13) (h_DF : DF = 5) : ∃ XY₁ : ℝ, XY₁ = (10 * Real.sqrt 6) / 17 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_angle_bisector_l868_86827


namespace NUMINAMATH_CALUDE_project_remaining_time_l868_86809

/-- Given the time spent on various tasks of a project, proves that the remaining time for writing the report is 9 hours. -/
theorem project_remaining_time (total_time research_time proposal_time visual_aids_time editing_time rehearsal_time : ℕ)
  (h_total : total_time = 40)
  (h_research : research_time = 12)
  (h_proposal : proposal_time = 4)
  (h_visual : visual_aids_time = 7)
  (h_editing : editing_time = 5)
  (h_rehearsal : rehearsal_time = 3) :
  total_time - (research_time + proposal_time + visual_aids_time + editing_time + rehearsal_time) = 9 := by
  sorry

end NUMINAMATH_CALUDE_project_remaining_time_l868_86809


namespace NUMINAMATH_CALUDE_log_value_proof_l868_86871

theorem log_value_proof (a : ℝ) (h1 : a > 0) (h2 : a^(1/2) = 4/9) :
  Real.log a / Real.log (2/3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_value_proof_l868_86871


namespace NUMINAMATH_CALUDE_problem_statement_l868_86800

theorem problem_statement (a b m n x : ℝ) 
  (h1 : a = -b) 
  (h2 : m * n = 1) 
  (h3 : abs x = 2) : 
  -2 * m * n + (a + b) / 2023 + x^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l868_86800


namespace NUMINAMATH_CALUDE_smallest_perfect_square_sum_24_consecutive_l868_86848

/-- The sum of 24 consecutive positive integers starting from n -/
def sum_24_consecutive (n : ℕ) : ℕ := 12 * (2 * n + 23)

/-- A number is a perfect square -/
def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

theorem smallest_perfect_square_sum_24_consecutive :
  (∃ n : ℕ, is_perfect_square (sum_24_consecutive n)) ∧
  (∀ n : ℕ, is_perfect_square (sum_24_consecutive n) → sum_24_consecutive n ≥ 300) :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_sum_24_consecutive_l868_86848


namespace NUMINAMATH_CALUDE_dot_position_after_operations_l868_86870

-- Define a square
structure Square where
  side : ℝ
  side_pos : side > 0

-- Define a point (for the dot)
structure Point where
  x : ℝ
  y : ℝ

-- Define the operations
def fold_diagonal (s : Square) (p : Point) : Point := sorry

def rotate_90_clockwise (s : Square) (p : Point) : Point := sorry

def unfold (s : Square) (p : Point) : Point := sorry

-- Theorem statement
theorem dot_position_after_operations (s : Square) : 
  let initial_dot : Point := ⟨s.side, s.side⟩
  let folded_dot := fold_diagonal s initial_dot
  let rotated_dot := rotate_90_clockwise s folded_dot
  let final_dot := unfold s rotated_dot
  final_dot.x > s.side / 2 ∧ final_dot.y < s.side / 2 := by sorry

end NUMINAMATH_CALUDE_dot_position_after_operations_l868_86870


namespace NUMINAMATH_CALUDE_endpoint_sum_l868_86873

/-- Given a line segment with one endpoint (1, -2) and midpoint (5, 4),
    the sum of coordinates of the other endpoint is 19. -/
theorem endpoint_sum (x y : ℝ) : 
  (1 + x) / 2 = 5 ∧ (-2 + y) / 2 = 4 → x + y = 19 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_sum_l868_86873


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l868_86813

theorem least_addition_for_divisibility : 
  ∃ (x : ℕ), x > 0 ∧ (1049 + x) % 25 = 0 ∧ ∀ (y : ℕ), y > 0 ∧ (1049 + y) % 25 = 0 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l868_86813


namespace NUMINAMATH_CALUDE_exam_failure_percentage_l868_86849

theorem exam_failure_percentage (total : ℝ) (failed_hindi : ℝ) (failed_both : ℝ) (passed_both : ℝ) :
  total = 100 →
  failed_hindi = 35 →
  failed_both = 20 →
  passed_both = 40 →
  ∃ failed_english : ℝ, failed_english = 45 :=
by sorry

end NUMINAMATH_CALUDE_exam_failure_percentage_l868_86849


namespace NUMINAMATH_CALUDE_odd_function_value_l868_86830

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value (f : ℝ → ℝ) :
  is_odd_function f →
  (∀ x < 0, f x = 2 * x^3 + x^2) →
  f 2 = 12 := by sorry

end NUMINAMATH_CALUDE_odd_function_value_l868_86830


namespace NUMINAMATH_CALUDE_cube_edge_length_l868_86878

theorem cube_edge_length (surface_area : ℝ) (h : surface_area = 150) :
  ∃ edge_length : ℝ, edge_length > 0 ∧ 6 * edge_length^2 = surface_area ∧ edge_length = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l868_86878


namespace NUMINAMATH_CALUDE_work_completion_days_l868_86810

/-- The number of days B can finish the work -/
def b_days : ℕ := 15

/-- The number of days B worked before leaving -/
def b_worked : ℕ := 10

/-- The number of days A needs to finish the remaining work after B left -/
def a_remaining : ℕ := 2

/-- The number of days A can finish the entire work -/
def a_days : ℕ := 6

theorem work_completion_days :
  (b_worked : ℚ) / b_days + a_remaining / a_days = 1 :=
sorry

end NUMINAMATH_CALUDE_work_completion_days_l868_86810


namespace NUMINAMATH_CALUDE_max_close_interval_length_l868_86840

-- Define the functions m and n
def m (x : ℝ) : ℝ := x^2 - 3*x + 4
def n (x : ℝ) : ℝ := 2*x - 3

-- Define the property of being close functions on an interval
def close_functions (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

-- State the theorem
theorem max_close_interval_length :
  ∃ (a b : ℝ), close_functions m n a b ∧ 
  ∀ (c d : ℝ), close_functions m n c d → d - c ≤ b - a :=
by sorry

end NUMINAMATH_CALUDE_max_close_interval_length_l868_86840


namespace NUMINAMATH_CALUDE_coefficient_x3y2_eq_neg_ten_l868_86839

/-- The coefficient of x^3 * y^2 in the expansion of (x^2 - x + y)^5 -/
def coefficient_x3y2 : ℤ :=
  (-1) * (Nat.choose 5 3)

theorem coefficient_x3y2_eq_neg_ten : coefficient_x3y2 = -10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y2_eq_neg_ten_l868_86839


namespace NUMINAMATH_CALUDE_counterexample_absolute_value_inequality_l868_86823

theorem counterexample_absolute_value_inequality : 
  ∃ (a b : ℝ), (abs a > abs b) ∧ (a ≤ b) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_absolute_value_inequality_l868_86823


namespace NUMINAMATH_CALUDE_a_upper_bound_l868_86881

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 5

-- State the theorem
theorem a_upper_bound (a : ℝ) :
  (∀ x y, 5/2 < x ∧ x < y → f a x < f a y) →
  a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_a_upper_bound_l868_86881


namespace NUMINAMATH_CALUDE_polynomial_simplification_l868_86888

theorem polynomial_simplification (p : ℝ) : 
  (5 * p^4 - 4 * p^3 + 3 * p - 7) + (8 - 9 * p^2 + p^4 + 6 * p) = 
  6 * p^4 - 4 * p^3 - 9 * p^2 + 9 * p + 1 := by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l868_86888


namespace NUMINAMATH_CALUDE_scientific_notation_of_280000_l868_86806

theorem scientific_notation_of_280000 : 
  280000 = 2.8 * (10 : ℝ)^5 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_280000_l868_86806


namespace NUMINAMATH_CALUDE_f_monotonic_increasing_interval_l868_86841

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 1)

theorem f_monotonic_increasing_interval :
  ∀ x y, 1 < x ∧ x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_f_monotonic_increasing_interval_l868_86841


namespace NUMINAMATH_CALUDE_stratified_sampling_l868_86894

/-- Stratified sampling problem -/
theorem stratified_sampling 
  (total_employees : ℕ) 
  (middle_managers : ℕ) 
  (senior_managers : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 150) 
  (h2 : middle_managers = 30) 
  (h3 : senior_managers = 10) 
  (h4 : sample_size = 30) :
  (sample_size * middle_managers / total_employees = 6) ∧ 
  (sample_size * senior_managers / total_employees = 2) :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_l868_86894
