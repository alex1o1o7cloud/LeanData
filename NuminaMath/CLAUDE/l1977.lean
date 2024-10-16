import Mathlib

namespace NUMINAMATH_CALUDE_nail_trimming_customers_l1977_197775

/-- The number of customers who had their nails trimmed -/
def number_of_customers (total_sounds : ℕ) (nails_per_person : ℕ) : ℕ :=
  total_sounds / nails_per_person

/-- Theorem: Given 60 nail trimming sounds and 20 nails per person, 
    the number of customers who had their nails trimmed is 3 -/
theorem nail_trimming_customers :
  number_of_customers 60 20 = 3 := by
  sorry

end NUMINAMATH_CALUDE_nail_trimming_customers_l1977_197775


namespace NUMINAMATH_CALUDE_max_value_of_f_l1977_197714

theorem max_value_of_f (x : ℝ) : 
  ∃ (M : ℝ), M = 2 ∧ ∀ (y : ℝ), min (3 - x^2) (2*x) ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1977_197714


namespace NUMINAMATH_CALUDE_michelle_crayons_l1977_197778

theorem michelle_crayons (num_boxes : ℕ) (crayons_per_box : ℕ) (h1 : num_boxes = 7) (h2 : crayons_per_box = 5) :
  num_boxes * crayons_per_box = 35 := by
  sorry

end NUMINAMATH_CALUDE_michelle_crayons_l1977_197778


namespace NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_l1977_197756

/-- Prove that if an ellipse and a hyperbola are tangent, then the parameter m of the hyperbola is 8 -/
theorem tangent_ellipse_hyperbola (x y m : ℝ) :
  (∃ x y, x^2 + 9*y^2 = 9 ∧ x^2 - m*(y+3)^2 = 1) →  -- Ellipse and hyperbola equations
  (∀ x y, x^2 + 9*y^2 = 9 → x^2 - m*(y+3)^2 ≤ 1) →  -- Tangency condition
  (∃ x y, x^2 + 9*y^2 = 9 ∧ x^2 - m*(y+3)^2 = 1) →  -- Point of tangency exists
  m = 8 :=
by sorry

end NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_l1977_197756


namespace NUMINAMATH_CALUDE_probability_in_specific_sequence_l1977_197774

/-- Represents an arithmetic sequence with given parameters -/
structure ArithmeticSequence where
  first_term : ℕ
  common_difference : ℕ
  last_term : ℕ

/-- Calculates the number of terms in the arithmetic sequence -/
def number_of_terms (seq : ArithmeticSequence) : ℕ :=
  (seq.last_term - seq.first_term) / seq.common_difference + 1

/-- Calculates the number of terms divisible by 6 in the sequence -/
def divisible_by_six (seq : ArithmeticSequence) : ℕ :=
  (number_of_terms seq) / 3

/-- The probability of selecting a number divisible by 6 from the sequence -/
def probability_divisible_by_six (seq : ArithmeticSequence) : ℚ :=
  (divisible_by_six seq : ℚ) / (number_of_terms seq)

theorem probability_in_specific_sequence :
  let seq := ArithmeticSequence.mk 50 4 998
  probability_divisible_by_six seq = 79 / 238 := by
  sorry

end NUMINAMATH_CALUDE_probability_in_specific_sequence_l1977_197774


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1977_197706

theorem smallest_n_congruence :
  ∃ (n : ℕ), n > 0 ∧ (3 * n) % 26 = 8 ∧ ∀ (m : ℕ), m > 0 ∧ (3 * m) % 26 = 8 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1977_197706


namespace NUMINAMATH_CALUDE_regular_pyramid_lateral_area_l1977_197779

/-- Theorem: The lateral surface area of a regular pyramid equals the area of the base
    divided by the cosine of the dihedral angle between a lateral face and the base. -/
theorem regular_pyramid_lateral_area 
  (n : ℕ) -- number of sides in the base
  (S : ℝ) -- area of one lateral face
  (A : ℝ) -- area of the base
  (α : ℝ) -- dihedral angle between a lateral face and the base
  (h1 : n > 0) -- the pyramid has at least 3 sides
  (h2 : S > 0) -- lateral face area is positive
  (h3 : A > 0) -- base area is positive
  (h4 : 0 < α ∧ α < π / 2) -- dihedral angle is between 0 and π/2
  : n * S = A / Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_regular_pyramid_lateral_area_l1977_197779


namespace NUMINAMATH_CALUDE_prob_at_least_four_girls_value_l1977_197763

def num_children : ℕ := 7
def prob_girl : ℚ := 3/5
def prob_boy : ℚ := 2/5

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def prob_at_least_four_girls : ℚ :=
  (binomial num_children 4 : ℚ) * (prob_girl ^ 4) * (prob_boy ^ 3) +
  (binomial num_children 5 : ℚ) * (prob_girl ^ 5) * (prob_boy ^ 2) +
  (binomial num_children 6 : ℚ) * (prob_girl ^ 6) * (prob_boy ^ 1) +
  (binomial num_children 7 : ℚ) * (prob_girl ^ 7) * (prob_boy ^ 0)

theorem prob_at_least_four_girls_value : prob_at_least_four_girls = 35325/78125 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_four_girls_value_l1977_197763


namespace NUMINAMATH_CALUDE_is_ellipse_l1977_197705

/-- The equation √((x-2)² + (y+2)²) + √((x-6)² + y²) = 12 represents an ellipse -/
theorem is_ellipse (x y : ℝ) : 
  (∃ (f₁ f₂ : ℝ × ℝ), f₁ ≠ f₂ ∧ 
  (∀ (p : ℝ × ℝ), Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) + 
                   Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) = 12) →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (p : ℝ × ℝ), (p.1^2 / a^2) + (p.2^2 / b^2) = 1) :=
by sorry

end NUMINAMATH_CALUDE_is_ellipse_l1977_197705


namespace NUMINAMATH_CALUDE_work_completion_theorem_l1977_197766

/-- Calculates the number of additional workers needed to complete a job earlier -/
def additional_workers (initial_workers : ℕ) (initial_days : ℕ) (actual_days : ℕ) : ℕ :=
  (initial_workers * initial_days / actual_days) - initial_workers

theorem work_completion_theorem (initial_workers : ℕ) (initial_days : ℕ) (actual_days : ℕ) 
  (h1 : initial_workers = 30)
  (h2 : initial_days = 8)
  (h3 : actual_days = 5) :
  additional_workers initial_workers initial_days actual_days = 18 := by
  sorry

#eval additional_workers 30 8 5

end NUMINAMATH_CALUDE_work_completion_theorem_l1977_197766


namespace NUMINAMATH_CALUDE_dart_probability_l1977_197736

/-- Represents an equilateral triangle divided into regions -/
structure DividedTriangle where
  total_regions : ℕ
  shaded_regions : ℕ
  h_positive : 0 < total_regions
  h_shaded_le_total : shaded_regions ≤ total_regions

/-- The probability of a dart landing in a shaded region -/
def shaded_probability (triangle : DividedTriangle) : ℚ :=
  triangle.shaded_regions / triangle.total_regions

/-- The specific triangle described in the problem -/
def problem_triangle : DividedTriangle where
  total_regions := 6
  shaded_regions := 3
  h_positive := by norm_num
  h_shaded_le_total := by norm_num

theorem dart_probability :
  shaded_probability problem_triangle = 1/2 := by sorry

end NUMINAMATH_CALUDE_dart_probability_l1977_197736


namespace NUMINAMATH_CALUDE_max_value_product_l1977_197707

theorem max_value_product (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) 
  (h_sum : x + y + z = 2) : 
  x^4 * y^3 * z^2 ≤ (1 : ℝ) / 9765625 :=
sorry

end NUMINAMATH_CALUDE_max_value_product_l1977_197707


namespace NUMINAMATH_CALUDE_wendy_facebook_pictures_l1977_197754

theorem wendy_facebook_pictures (total_albums : ℕ) (pics_in_first_album : ℕ) 
  (pics_per_other_album : ℕ) (other_albums : ℕ) :
  total_albums = other_albums + 1 →
  pics_in_first_album = 44 →
  pics_per_other_album = 7 →
  other_albums = 5 →
  pics_in_first_album + other_albums * pics_per_other_album = 79 := by
  sorry

end NUMINAMATH_CALUDE_wendy_facebook_pictures_l1977_197754


namespace NUMINAMATH_CALUDE_valid_outfits_count_l1977_197738

/-- The number of shirts, pants, and hats available -/
def num_items : ℕ := 5

/-- The number of colors available for each item -/
def num_colors : ℕ := 5

/-- The number of outfits where no two items are the same color -/
def num_valid_outfits : ℕ := num_items * (num_items - 1) * (num_items - 2)

theorem valid_outfits_count :
  num_valid_outfits = 60 :=
by sorry

end NUMINAMATH_CALUDE_valid_outfits_count_l1977_197738


namespace NUMINAMATH_CALUDE_profit_margin_relation_l1977_197702

theorem profit_margin_relation (S C : ℝ) (n : ℝ) (h1 : S > 0) (h2 : C > 0) (h3 : n > 0) : 
  ((1 / 3 : ℝ) * S = (1 / n : ℝ) * C) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_profit_margin_relation_l1977_197702


namespace NUMINAMATH_CALUDE_inequality_solution_l1977_197710

theorem inequality_solution (x : ℝ) : 
  3 - 1 / (3 * x + 4) < 5 ↔ x < -4/3 ∨ x > -3/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1977_197710


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_six_mod_seventeen_l1977_197777

theorem least_five_digit_congruent_to_six_mod_seventeen :
  ∃ (n : ℕ), 
    (n ≥ 10000 ∧ n < 100000) ∧  -- Five-digit positive integer
    (n % 17 = 6) ∧              -- Congruent to 6 (mod 17)
    (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ (m % 17 = 6) → n ≤ m) ∧  -- Least such number
    n = 10002                   -- The number is 10,002
  := by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_six_mod_seventeen_l1977_197777


namespace NUMINAMATH_CALUDE_icosahedron_edge_probability_l1977_197776

/-- A regular icosahedron -/
structure Icosahedron :=
  (vertices : Nat)
  (edges_per_vertex : Nat)
  (h_vertices : vertices = 12)
  (h_edges_per_vertex : edges_per_vertex = 5)

/-- The probability of selecting two vertices that form an edge in an icosahedron -/
def edge_probability (i : Icosahedron) : ℚ :=
  5 / 11

/-- Theorem: The probability of randomly selecting two vertices that form an edge in a regular icosahedron is 5/11 -/
theorem icosahedron_edge_probability (i : Icosahedron) : 
  edge_probability i = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_icosahedron_edge_probability_l1977_197776


namespace NUMINAMATH_CALUDE_polynomial_has_negative_root_l1977_197735

theorem polynomial_has_negative_root : ∃ x : ℝ, x < 0 ∧ x^7 + 2*x^5 + 5*x^3 - x + 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_has_negative_root_l1977_197735


namespace NUMINAMATH_CALUDE_provisions_last_20_days_l1977_197790

/-- Calculates the number of days provisions will last after reinforcement -/
def provisions_duration (initial_men : ℕ) (initial_days : ℕ) (days_passed : ℕ) (reinforcement : ℕ) : ℚ :=
  let total_provisions := initial_men * initial_days
  let remaining_provisions := total_provisions - (initial_men * days_passed)
  let new_total_men := initial_men + reinforcement
  remaining_provisions / new_total_men

/-- Proves that given the initial conditions, the provisions will last for 20 days after reinforcement -/
theorem provisions_last_20_days :
  provisions_duration 2000 54 21 1300 = 20 := by
  sorry

end NUMINAMATH_CALUDE_provisions_last_20_days_l1977_197790


namespace NUMINAMATH_CALUDE_rotation_90_clockwise_effect_l1977_197717

-- Define the shapes
inductive Shape
  | Pentagon
  | Ellipse
  | Rectangle

-- Define the positions on the circle
structure Position :=
  (angle : ℝ)

-- Define the configuration of shapes on the circle
structure Configuration :=
  (pentagon_pos : Position)
  (ellipse_pos : Position)
  (rectangle_pos : Position)

-- Define the rotation operation
def rotate_90_clockwise (config : Configuration) : Configuration :=
  { pentagon_pos := config.ellipse_pos,
    ellipse_pos := config.rectangle_pos,
    rectangle_pos := config.pentagon_pos }

-- Theorem statement
theorem rotation_90_clockwise_effect (initial_config : Configuration) :
  let final_config := rotate_90_clockwise initial_config
  (final_config.pentagon_pos = initial_config.ellipse_pos) ∧
  (final_config.ellipse_pos = initial_config.rectangle_pos) ∧
  (final_config.rectangle_pos = initial_config.pentagon_pos) :=
by
  sorry


end NUMINAMATH_CALUDE_rotation_90_clockwise_effect_l1977_197717


namespace NUMINAMATH_CALUDE_locus_of_centers_l1977_197768

-- Define the two circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the property of being externally tangent to C₁ and internally tangent to C₂
def externally_internally_tangent (a b r : ℝ) : Prop :=
  ∃ (x y : ℝ), C₁ x y ∧ (x - a)^2 + (y - b)^2 = (r + 2)^2 ∧
                C₂ x y ∧ (x - a)^2 + (y - b)^2 = (3 - r)^2

-- State the theorem
theorem locus_of_centers : 
  ∀ (a b : ℝ), (∃ r : ℝ, externally_internally_tangent a b r) ↔ 
  16 * a^2 + 25 * b^2 - 48 * a - 64 = 0 := by sorry

end NUMINAMATH_CALUDE_locus_of_centers_l1977_197768


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1977_197731

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) : 
  (∀ angle : ℝ, angle = 156 ∧ (180 * (n - 2) : ℝ) = n * angle) → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1977_197731


namespace NUMINAMATH_CALUDE_problem_statement_l1977_197724

theorem problem_statement (a b c : ℝ) 
  (h_diff1 : a ≠ b) (h_diff2 : b ≠ c) (h_diff3 : a ≠ c)
  (h_eq : Real.sqrt (a^3 * (b-a)^3) - Real.sqrt (a^3 * (c-a)^3) = Real.sqrt (a-b) - Real.sqrt (c-a)) :
  a^2 + b^2 + c^2 - 2*a*b + 2*b*c - 2*a*c = 0 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1977_197724


namespace NUMINAMATH_CALUDE_youngest_daughter_cost_l1977_197728

/-- Represents the cost of dresses and hats bought by the daughters -/
structure Purchase where
  dresses : ℕ
  hats : ℕ
  cost : ℕ

/-- The problem setup -/
def merchant_problem : Prop :=
  ∃ (dress_cost hat_cost : ℕ),
    let eldest := Purchase.mk 6 3 105
    let second := Purchase.mk 3 5 70
    let youngest := Purchase.mk 1 2 0
    eldest.cost = eldest.dresses * dress_cost + eldest.hats * hat_cost ∧
    second.cost = second.dresses * dress_cost + second.hats * hat_cost ∧
    youngest.dresses * dress_cost + youngest.hats * hat_cost = 25

/-- The theorem to be proved -/
theorem youngest_daughter_cost :
  merchant_problem := by sorry

end NUMINAMATH_CALUDE_youngest_daughter_cost_l1977_197728


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l1977_197772

/-- A parabola with its focus on the line x-2y+2=0 has a standard equation of either x^2 = 4y or y^2 = -8x -/
theorem parabola_standard_equation (F : ℝ × ℝ) :
  (F.1 - 2 * F.2 + 2 = 0) →
  (∃ (x y : ℝ → ℝ), (∀ t, x t ^ 2 = 4 * y t) ∨ (∀ t, y t ^ 2 = -8 * x t)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l1977_197772


namespace NUMINAMATH_CALUDE_x_value_in_set_l1977_197733

theorem x_value_in_set (x : ℝ) : 1 ∈ ({x, x^2} : Set ℝ) → x ≠ x^2 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_in_set_l1977_197733


namespace NUMINAMATH_CALUDE_odd_log_properties_l1977_197759

noncomputable section

-- Define the logarithm function with base a
def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := log a x

-- Theorem statement
theorem odd_log_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  -- Part 1: The value of m
  (∀ x > 0, log a x + log a (-x) = 0) →
  -- Part 2: The derivative of f
  (∀ x ≠ 0, deriv (f a) x = (Real.log a)⁻¹ / x) ∧
  -- Part 3: The value of a given the range condition
  (∀ x ∈ Set.Ioo 1 (a - 2), f a x ∈ Set.Ioi 1) →
  a = 2 + Real.sqrt 5 := by
sorry

end

end NUMINAMATH_CALUDE_odd_log_properties_l1977_197759


namespace NUMINAMATH_CALUDE_two_digit_number_property_l1977_197742

def P (n : Nat) : Nat :=
  (n / 10) * (n % 10)

def S (n : Nat) : Nat :=
  (n / 10) + (n % 10)

theorem two_digit_number_property : ∃! N : Nat, 
  10 ≤ N ∧ N < 100 ∧ N = P N + 2 * S N :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l1977_197742


namespace NUMINAMATH_CALUDE_walmart_sales_theorem_walmart_december_sales_l1977_197749

/-- Calculates the total sales amount for Wal-Mart in December -/
theorem walmart_sales_theorem (thermometer_price : ℕ) (hot_water_bottle_price : ℕ) 
  (thermometer_to_bottle_ratio : ℕ) (hot_water_bottles_sold : ℕ) : ℕ :=
  let thermometers_sold := thermometer_to_bottle_ratio * hot_water_bottles_sold
  let thermometer_sales := thermometers_sold * thermometer_price
  let hot_water_bottle_sales := hot_water_bottles_sold * hot_water_bottle_price
  thermometer_sales + hot_water_bottle_sales

/-- Proves that the total sales amount for Wal-Mart in December is $1200 -/
theorem walmart_december_sales : 
  walmart_sales_theorem 2 6 7 60 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_walmart_sales_theorem_walmart_december_sales_l1977_197749


namespace NUMINAMATH_CALUDE_even_function_inequality_range_l1977_197785

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem even_function_inequality_range
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_mono : monotone_increasing_on f (Set.Ici 0)) :
  {x : ℝ | f (2*x - 1) < f (-1)} = Set.Ioo 0 1 := by
sorry

end NUMINAMATH_CALUDE_even_function_inequality_range_l1977_197785


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l1977_197792

/-- Given two circles where one has a diameter of 80 cm and its radius is 4 times
    the radius of the other, prove that the radius of the smaller circle is 10 cm. -/
theorem smaller_circle_radius (d : ℝ) (r₁ r₂ : ℝ) : 
  d = 80 → r₁ = d / 2 → r₁ = 4 * r₂ → r₂ = 10 := by
  sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l1977_197792


namespace NUMINAMATH_CALUDE_inequality_proof_l1977_197762

theorem inequality_proof (a b c : ℝ) 
  (ha : a = Real.exp (-3))
  (hb : b = Real.log 1.02)
  (hc : c = Real.sin 0.04) : 
  b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1977_197762


namespace NUMINAMATH_CALUDE_james_overtime_multiple_l1977_197793

/-- Harry's pay rate for the first 24 hours -/
def harry_base_rate (x : ℝ) : ℝ := x

/-- Harry's pay rate for additional hours -/
def harry_overtime_rate (x : ℝ) : ℝ := 1.5 * x

/-- James's pay rate for the first 40 hours -/
def james_base_rate (x : ℝ) : ℝ := x

/-- James's pay rate for additional hours -/
def james_overtime_rate (x : ℝ) (m : ℝ) : ℝ := m * x

/-- Total hours worked by James -/
def james_total_hours : ℝ := 41

/-- Theorem stating the multiple of x dollars for James's overtime -/
theorem james_overtime_multiple (x : ℝ) (m : ℝ) :
  (harry_base_rate x * 24 + harry_overtime_rate x * (james_total_hours - 24) =
   james_base_rate x * 40 + james_overtime_rate x m * (james_total_hours - 40)) →
  m = 9.5 := by
  sorry

#check james_overtime_multiple

end NUMINAMATH_CALUDE_james_overtime_multiple_l1977_197793


namespace NUMINAMATH_CALUDE_ball_pricing_theorem_l1977_197715

/-- Represents the price and quantity of basketballs and volleyballs -/
structure BallPrices where
  basketball_price : ℕ
  volleyball_price : ℕ
  basketball_quantity : ℕ
  volleyball_quantity : ℕ

/-- Conditions of the ball purchasing problem -/
def ball_conditions (prices : BallPrices) : Prop :=
  prices.basketball_quantity + prices.volleyball_quantity = 20 ∧
  2 * prices.basketball_price + 3 * prices.volleyball_price = 190 ∧
  3 * prices.basketball_price = 5 * prices.volleyball_price

/-- Cost calculation for a given quantity of basketballs and volleyballs -/
def total_cost (prices : BallPrices) (b_qty : ℕ) (v_qty : ℕ) : ℕ :=
  b_qty * prices.basketball_price + v_qty * prices.volleyball_price

/-- Theorem stating the correct prices and most cost-effective plan -/
theorem ball_pricing_theorem (prices : BallPrices) :
  ball_conditions prices →
  prices.basketball_price = 50 ∧
  prices.volleyball_price = 30 ∧
  (∀ b v, b + v = 20 → b ≥ 8 → total_cost prices b v ≤ 800 →
    total_cost prices 8 12 ≤ total_cost prices b v) :=
sorry

end NUMINAMATH_CALUDE_ball_pricing_theorem_l1977_197715


namespace NUMINAMATH_CALUDE_cab_ride_cost_per_mile_l1977_197746

/-- Calculates the cost per mile for Briar's cab rides -/
theorem cab_ride_cost_per_mile
  (days : ℕ)
  (distance_to_event : ℝ)
  (total_cost : ℝ)
  (h1 : days = 7)
  (h2 : distance_to_event = 200)
  (h3 : total_cost = 7000) :
  total_cost / (2 * days * distance_to_event) = 2.5 := by
  sorry

#check cab_ride_cost_per_mile

end NUMINAMATH_CALUDE_cab_ride_cost_per_mile_l1977_197746


namespace NUMINAMATH_CALUDE_justine_colored_sheets_l1977_197797

/-- Given a total number of sheets and binders, calculate the number of sheets Justine colored. -/
def sheets_colored (total_sheets : ℕ) (num_binders : ℕ) : ℕ :=
  let sheets_per_binder := total_sheets / num_binders
  (2 * sheets_per_binder) / 3

/-- Prove that Justine colored 356 sheets given the problem conditions. -/
theorem justine_colored_sheets :
  sheets_colored 3750 7 = 356 := by
  sorry

end NUMINAMATH_CALUDE_justine_colored_sheets_l1977_197797


namespace NUMINAMATH_CALUDE_layer_sum_2014_implies_digit_sum_13_l1977_197718

/-- Represents a four-digit positive integer --/
structure FourDigitInt where
  w : Nat
  x : Nat
  y : Nat
  z : Nat
  w_nonzero : w ≠ 0
  w_upper_bound : w < 10
  x_upper_bound : x < 10
  y_upper_bound : y < 10
  z_upper_bound : z < 10

/-- Calculates the layer sum of a four-digit integer --/
def layerSum (n : FourDigitInt) : Nat :=
  1000 * n.w + 100 * n.x + 10 * n.y + n.z +
  100 * n.x + 10 * n.y + n.z +
  10 * n.y + n.z +
  n.z

/-- Main theorem --/
theorem layer_sum_2014_implies_digit_sum_13 (n : FourDigitInt) :
  layerSum n = 2014 → n.w + n.x + n.y + n.z = 13 := by
  sorry

end NUMINAMATH_CALUDE_layer_sum_2014_implies_digit_sum_13_l1977_197718


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_expression_99996_divisible_by_seven_largest_n_is_99996_l1977_197782

def expression (n : ℕ) : ℤ :=
  9 * (n - 3)^6 - 3 * n^3 + 21 * n - 42

theorem largest_n_divisible_by_seven :
  ∀ n : ℕ, n < 100000 →
    (expression n) % 7 = 0 →
    n ≤ 99996 :=
by sorry

theorem expression_99996_divisible_by_seven :
  (expression 99996) % 7 = 0 :=
by sorry

theorem largest_n_is_99996 :
  ∀ n : ℕ, n < 100000 →
    (expression n) % 7 = 0 →
    n ≤ 99996 ∧
    (expression 99996) % 7 = 0 ∧
    99996 < 100000 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_expression_99996_divisible_by_seven_largest_n_is_99996_l1977_197782


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l1977_197725

theorem set_equality_implies_sum (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = ({0, a^2, a+b} : Set ℝ) → a^2022 + b^2023 = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l1977_197725


namespace NUMINAMATH_CALUDE_negation_equivalence_l1977_197752

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → Real.sqrt x > x + 1) ↔ (∃ x : ℝ, x > 0 ∧ Real.sqrt x ≤ x + 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1977_197752


namespace NUMINAMATH_CALUDE_two_digit_condition_l1977_197799

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- Define the property that we want to prove
def satisfiesCondition (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n = 6 * sumOfDigits (n + 7)

-- Statement of the theorem
theorem two_digit_condition :
  ∀ n : ℕ, satisfiesCondition n ↔ (n = 24 ∨ n = 78) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_condition_l1977_197799


namespace NUMINAMATH_CALUDE_absolute_value_minus_self_nonnegative_l1977_197757

theorem absolute_value_minus_self_nonnegative (a : ℝ) : |a| - a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_minus_self_nonnegative_l1977_197757


namespace NUMINAMATH_CALUDE_positive_integer_solutions_inequality_l1977_197788

theorem positive_integer_solutions_inequality (x : ℕ+) :
  2 * (x.val - 1) < 7 - x.val ↔ x = 1 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_inequality_l1977_197788


namespace NUMINAMATH_CALUDE_insulation_project_proof_l1977_197700

/-- The daily completion rate of Team A in square meters -/
def team_a_rate : ℝ := 200

/-- The daily completion rate of Team B in square meters -/
def team_b_rate : ℝ := 1.5 * team_a_rate

/-- The total area to be insulated in square meters -/
def total_area : ℝ := 9000

/-- The difference in completion time between Team A and Team B in days -/
def time_difference : ℝ := 15

theorem insulation_project_proof :
  (total_area / team_a_rate) - (total_area / team_b_rate) = time_difference :=
by sorry

end NUMINAMATH_CALUDE_insulation_project_proof_l1977_197700


namespace NUMINAMATH_CALUDE_two_person_subcommittees_l1977_197791

theorem two_person_subcommittees (n : ℕ) (h : n = 8) : 
  Nat.choose n 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_two_person_subcommittees_l1977_197791


namespace NUMINAMATH_CALUDE_employee_salary_proof_l1977_197751

/-- The weekly salary of employee n -/
def salary_n : ℝ := 270

/-- The weekly salary of employee m -/
def salary_m : ℝ := 1.2 * salary_n

/-- The total weekly salary for both employees -/
def total_salary : ℝ := 594

theorem employee_salary_proof :
  salary_n + salary_m = total_salary :=
by sorry

end NUMINAMATH_CALUDE_employee_salary_proof_l1977_197751


namespace NUMINAMATH_CALUDE_certain_number_problem_l1977_197741

theorem certain_number_problem (x N : ℝ) :
  625^(-x) + N^(-2*x) + 5^(-4*x) = 11 →
  x = 0.25 →
  N = 25/2809 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1977_197741


namespace NUMINAMATH_CALUDE_total_potatoes_l1977_197767

theorem total_potatoes (nancy_potatoes sandy_potatoes andy_potatoes : ℕ)
  (h1 : nancy_potatoes = 6)
  (h2 : sandy_potatoes = 7)
  (h3 : andy_potatoes = 9) :
  nancy_potatoes + sandy_potatoes + andy_potatoes = 22 :=
by sorry

end NUMINAMATH_CALUDE_total_potatoes_l1977_197767


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l1977_197745

theorem complex_magnitude_example : Complex.abs (1 - Complex.I / 2) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l1977_197745


namespace NUMINAMATH_CALUDE_remote_sensing_primary_for_sea_level_info_l1977_197761

/-- Represents different technologies used in geographic information systems -/
inductive GISTechnology
  | RemoteSensing
  | GPS
  | GIS
  | DigitalEarth

/-- Represents the capability of a technology to acquire sea level rise information -/
def can_acquire_sea_level_info (tech : GISTechnology) : Prop :=
  match tech with
  | GISTechnology.RemoteSensing => true
  | _ => false

/-- Theorem stating that Remote Sensing is the primary technology for acquiring sea level rise information -/
theorem remote_sensing_primary_for_sea_level_info :
  ∀ (tech : GISTechnology),
    can_acquire_sea_level_info tech → tech = GISTechnology.RemoteSensing :=
by
  sorry


end NUMINAMATH_CALUDE_remote_sensing_primary_for_sea_level_info_l1977_197761


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l1977_197727

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively,
    if a / c = b / d = 2 / 3, then the ratio of the area of rectangle A
    to the area of rectangle B is 4:9. -/
theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 2 / 3) (h2 : b / d = 2 / 3) :
  (a * b) / (c * d) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l1977_197727


namespace NUMINAMATH_CALUDE_inverse_73_mod_74_l1977_197723

theorem inverse_73_mod_74 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 73 ∧ (73 * x) % 74 = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_73_mod_74_l1977_197723


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l1977_197721

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 104 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 104 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l1977_197721


namespace NUMINAMATH_CALUDE_first_player_wins_l1977_197712

/-- Represents a move in the coin game -/
structure Move where
  player : Nat
  coins : Nat

/-- Represents the state of the game -/
structure GameState where
  coins : Nat
  turn : Nat

/-- Checks if a move is valid for a given player -/
def isValidMove (m : Move) (gs : GameState) : Prop :=
  (m.player = gs.turn % 2) ∧
  (if m.player = 0
   then m.coins % 2 = 1 ∧ m.coins ≥ 1 ∧ m.coins ≤ 99
   else m.coins % 2 = 0 ∧ m.coins ≥ 2 ∧ m.coins ≤ 100) ∧
  (m.coins ≤ gs.coins)

/-- Applies a move to a game state -/
def applyMove (m : Move) (gs : GameState) : GameState :=
  { coins := gs.coins - m.coins, turn := gs.turn + 1 }

/-- Defines a winning strategy for the first player -/
def firstPlayerStrategy (gs : GameState) : Move :=
  if gs.turn = 0 then
    { player := 0, coins := 99 }
  else
    { player := 0, coins := 101 - (gs.coins % 101) }

/-- Theorem stating that the first player has a winning strategy -/
theorem first_player_wins :
  ∃ (strategy : GameState → Move),
    ∀ (gs : GameState),
      gs.coins = 2019 →
      (∀ (m : Move), isValidMove m gs → 
        ∃ (nextMove : Move), 
          isValidMove nextMove (applyMove m gs) ∧
          strategy (applyMove m gs) = nextMove) ∧
      (∀ (sequence : Nat → Move),
        (∀ (i : Nat), isValidMove (sequence i) (applyMove (sequence (i-1)) gs)) →
        ∃ (n : Nat), ¬isValidMove (sequence n) (applyMove (sequence (n-1)) gs)) :=
sorry


end NUMINAMATH_CALUDE_first_player_wins_l1977_197712


namespace NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_65_l1977_197765

theorem right_triangle_with_hypotenuse_65 :
  ∀ a b c : ℕ,
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 25 :=          -- Shorter leg length
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_65_l1977_197765


namespace NUMINAMATH_CALUDE_triangle_min_perimeter_l1977_197760

theorem triangle_min_perimeter (a b x : ℕ) (ha : a = 36) (hb : b = 50) : 
  (a + b > x ∧ a + x > b ∧ b + x > a) → (a + b + x ≥ 101) := by
  sorry

end NUMINAMATH_CALUDE_triangle_min_perimeter_l1977_197760


namespace NUMINAMATH_CALUDE_find_k_l1977_197771

theorem find_k (a b c d k : ℝ) 
  (h1 : a * b * c * d = 2007)
  (h2 : a = Real.sqrt (55 + Real.sqrt (k + a)))
  (h3 : b = Real.sqrt (55 - Real.sqrt (k + b)))
  (h4 : c = Real.sqrt (55 + Real.sqrt (k - c)))
  (h5 : d = Real.sqrt (55 - Real.sqrt (k - d))) :
  k = 1018 := by sorry

end NUMINAMATH_CALUDE_find_k_l1977_197771


namespace NUMINAMATH_CALUDE_integer_root_values_l1977_197722

def polynomial (a x : ℤ) : ℤ := x^3 + 3*x^2 + a*x + 9

def has_integer_root (a : ℤ) : Prop :=
  ∃ x : ℤ, polynomial a x = 0

theorem integer_root_values :
  {a : ℤ | has_integer_root a} = {-109, -21, -13, 3, 11, 53} := by sorry

end NUMINAMATH_CALUDE_integer_root_values_l1977_197722


namespace NUMINAMATH_CALUDE_range_of_ab_plus_a_plus_b_l1977_197744

def f (x : ℝ) := |x^2 + 2*x - 1|

theorem range_of_ab_plus_a_plus_b 
  (a b : ℝ) 
  (h1 : a < b) 
  (h2 : b < -1) 
  (h3 : f a = f b) :
  ∀ y : ℝ, (∃ x : ℝ, a < x ∧ x < b ∧ y = a*b + a + b) → -1 < y ∧ y < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_ab_plus_a_plus_b_l1977_197744


namespace NUMINAMATH_CALUDE_equation_no_real_roots_and_positive_quadratic_l1977_197758

theorem equation_no_real_roots_and_positive_quadratic :
  (∃ m : ℝ, ∀ x : ℝ, x^2 + x - m ≠ 0) ∧
  (∀ x : ℝ, x^2 + x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_no_real_roots_and_positive_quadratic_l1977_197758


namespace NUMINAMATH_CALUDE_reggies_lost_games_l1977_197769

/-- Given the conditions of Reggie's marble game, prove the number of games he lost. -/
theorem reggies_lost_games 
  (total_games : ℕ) 
  (initial_marbles : ℕ) 
  (bet_per_game : ℕ) 
  (final_marbles : ℕ) 
  (h1 : total_games = 9)
  (h2 : initial_marbles = 100)
  (h3 : bet_per_game = 10)
  (h4 : final_marbles = 90) :
  (initial_marbles - final_marbles) / bet_per_game = 1 :=
by sorry

end NUMINAMATH_CALUDE_reggies_lost_games_l1977_197769


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1977_197730

theorem complex_equation_solution :
  ∀ b : ℝ, (6 - b * I) / (1 + 2 * I) = 2 - 2 * I → b = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1977_197730


namespace NUMINAMATH_CALUDE_january_salary_l1977_197743

/-- Represents the monthly salary for a person -/
structure MonthlySalary where
  january : ℕ
  february : ℕ
  march : ℕ
  april : ℕ
  may : ℕ

/-- The average salary calculation is correct -/
def average_salary_correct (s : MonthlySalary) : Prop :=
  (s.january + s.february + s.march + s.april) / 4 = 8000 ∧
  (s.february + s.march + s.april + s.may) / 4 = 9500

/-- The salary for May is 6500 -/
def may_salary_correct (s : MonthlySalary) : Prop :=
  s.may = 6500

/-- The theorem stating that given the conditions, the salary for January is 500 -/
theorem january_salary (s : MonthlySalary) 
  (h1 : average_salary_correct s) 
  (h2 : may_salary_correct s) : 
  s.january = 500 := by
  sorry

end NUMINAMATH_CALUDE_january_salary_l1977_197743


namespace NUMINAMATH_CALUDE_pyramid_sum_l1977_197729

theorem pyramid_sum (x : ℝ) : 
  let row2_left : ℝ := 11
  let row2_middle : ℝ := 6 + x
  let row2_right : ℝ := x + 7
  let row3_left : ℝ := row2_left + row2_middle
  let row3_right : ℝ := row2_middle + row2_right
  let row4 : ℝ := row3_left + row3_right
  row4 = 60 → x = 10 := by
sorry

end NUMINAMATH_CALUDE_pyramid_sum_l1977_197729


namespace NUMINAMATH_CALUDE_zhang_san_correct_probability_l1977_197786

theorem zhang_san_correct_probability :
  let total_questions : ℕ := 4
  let questions_with_ideas : ℕ := 3
  let questions_unclear : ℕ := 1
  let prob_correct_with_idea : ℚ := 3/4
  let prob_correct_when_unclear : ℚ := 1/4
  let prob_selecting_question_with_idea : ℚ := questions_with_ideas / total_questions
  let prob_selecting_question_unclear : ℚ := questions_unclear / total_questions

  prob_selecting_question_with_idea * prob_correct_with_idea +
  prob_selecting_question_unclear * prob_correct_when_unclear = 5/8 :=
by sorry

end NUMINAMATH_CALUDE_zhang_san_correct_probability_l1977_197786


namespace NUMINAMATH_CALUDE_set_A_properties_l1977_197783

-- Define the set A
def A : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

-- Theorem statement
theorem set_A_properties :
  (∀ n : ℤ, (2*n + 1) ∈ A) ∧
  (∀ k : ℤ, (4*k - 2) ∉ A) ∧
  (∀ a b : ℤ, a ∈ A → b ∈ A → (a * b) ∈ A) := by
  sorry

end NUMINAMATH_CALUDE_set_A_properties_l1977_197783


namespace NUMINAMATH_CALUDE_magazine_budget_cut_percentage_l1977_197780

def original_budget : ℝ := 840
def cut_amount : ℝ := 588

theorem magazine_budget_cut_percentage :
  (cut_amount / original_budget) * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_magazine_budget_cut_percentage_l1977_197780


namespace NUMINAMATH_CALUDE_traffic_light_probability_l1977_197750

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  total_time : ℕ
  change_times : List ℕ

/-- Calculates the probability of observing a color change in a given interval -/
def probability_of_change (cycle : TrafficLightCycle) (interval : ℕ) : ℚ :=
  let change_windows := cycle.change_times.map (λ t => if t ≤ cycle.total_time - interval then interval else t + interval - cycle.total_time)
  let total_change_time := change_windows.sum
  total_change_time / cycle.total_time

/-- The main theorem: probability of observing a color change is 1/7 -/
theorem traffic_light_probability :
  let cycle : TrafficLightCycle := { total_time := 63, change_times := [30, 33, 63] }
  probability_of_change cycle 3 = 1/7 := by
  sorry


end NUMINAMATH_CALUDE_traffic_light_probability_l1977_197750


namespace NUMINAMATH_CALUDE_zookeeper_fish_count_l1977_197703

theorem zookeeper_fish_count (penguins_fed : ℕ) (total_penguins : ℕ) (penguins_to_feed : ℕ) :
  penguins_fed = 19 →
  total_penguins = 36 →
  penguins_to_feed = 17 →
  penguins_fed + penguins_to_feed = total_penguins :=
by sorry

end NUMINAMATH_CALUDE_zookeeper_fish_count_l1977_197703


namespace NUMINAMATH_CALUDE_triangle_properties_l1977_197787

-- Define the triangle
def triangle_ABC (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem triangle_properties :
  ∀ (a b c : ℝ),
  triangle_ABC a b c →
  a = 2 →
  c = 3 →
  Real.cos (Real.arccos (1/4)) = 1/4 →
  b = Real.sqrt 10 ∧
  Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))) = (3 * Real.sqrt 6) / 8 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1977_197787


namespace NUMINAMATH_CALUDE_water_remaining_l1977_197737

theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 4/3 → remaining = initial - used → remaining = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_l1977_197737


namespace NUMINAMATH_CALUDE_x_value_l1977_197739

theorem x_value : ∃ x : ℝ, x ≠ 0 ∧ x = 3 * (1/x * (-x)) + 5 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1977_197739


namespace NUMINAMATH_CALUDE_problem_statement_l1977_197747

theorem problem_statement (a : ℝ) (h : (a + 1/a)^3 = 3) :
  a^4 + 1/a^4 = Real.rpow 9 (1/3) - 4 * Real.rpow 3 (1/3) + 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1977_197747


namespace NUMINAMATH_CALUDE_smallest_box_volume_l1977_197732

theorem smallest_box_volume (l w h : ℕ) (h1 : l > 0) (h2 : w = 3 * l) (h3 : h = 4 * l) :
  l * w * h = 96 ∨ l * w * h > 96 :=
sorry

end NUMINAMATH_CALUDE_smallest_box_volume_l1977_197732


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l1977_197789

/-- Given a vector a = (2, 1) and another vector b such that a · b = 10 and |a + b| = 5, prove that |b| = 2√10. -/
theorem vector_magnitude_proof (b : ℝ × ℝ) :
  let a : ℝ × ℝ := (2, 1)
  (a.1 * b.1 + a.2 * b.2 = 10) →
  ((a.1 + b.1)^2 + (a.2 + b.2)^2 = 25) →
  (b.1^2 + b.2^2 = 40) := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l1977_197789


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_in_solution_set_l1977_197796

/-- A system of two linear equations in two variables with parameters a and b -/
structure LinearSystem (a b : ℝ) where
  eq1 : ∀ x y : ℝ, 3 * (a + b) * x + 12 * y = a
  eq2 : ∀ x y : ℝ, 4 * b * x + (a + b) * b * y = 1

/-- The condition for the system to have infinitely many solutions -/
def HasInfinitelySolutions (a b : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 3 * (a + b) = 4 * b * k ∧ 12 = (a + b) * b * k ∧ a = k

/-- The set of pairs (a, b) that satisfy the condition -/
def SolutionSet : Set (ℝ × ℝ) :=
  {(1, 3), (3, 1), (-2 - Real.sqrt 7, Real.sqrt 7 - 2), (Real.sqrt 7 - 2, -2 - Real.sqrt 7)}

/-- The main theorem stating the equivalence -/
theorem infinite_solutions_iff_in_solution_set (a b : ℝ) :
  HasInfinitelySolutions a b ↔ (a, b) ∈ SolutionSet := by sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_in_solution_set_l1977_197796


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1977_197720

theorem arithmetic_sequence_sum (n : ℕ) (s : ℕ → ℝ) :
  (∀ k, s (k + 1) - s k = s (k + 2) - s (k + 1)) →  -- arithmetic sequence condition
  s n = 48 →                                        -- sum of first n terms
  s (2 * n) = 60 →                                  -- sum of first 2n terms
  s (3 * n) = 36 :=                                 -- sum of first 3n terms
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1977_197720


namespace NUMINAMATH_CALUDE_line_slope_problem_l1977_197784

/-- Given m > 0 and points (m, 4) and (2, m) lie on a line with slope m^2, prove m = 2 -/
theorem line_slope_problem (m : ℝ) (h1 : m > 0) 
  (h2 : (m - 4) / (2 - m) = m^2) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_problem_l1977_197784


namespace NUMINAMATH_CALUDE_seven_digit_number_product_l1977_197719

theorem seven_digit_number_product : ∃ (x y : ℕ), 
  (1000000 ≤ x ∧ x < 10000000) ∧ 
  (1000000 ≤ y ∧ y < 10000000) ∧ 
  (10^7 * x + y = 3 * x * y) ∧ 
  (x = 1666667 ∧ y = 3333334) := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_number_product_l1977_197719


namespace NUMINAMATH_CALUDE_six_graduates_distribution_l1977_197748

/-- The number of ways to distribute n graduates among 2 employers, 
    with each employer receiving at least k graduates -/
def distribution_schemes (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 graduates among 2 employers, 
    with each employer receiving at least 2 graduates, is 50 -/
theorem six_graduates_distribution : distribution_schemes 6 2 = 50 := by sorry

end NUMINAMATH_CALUDE_six_graduates_distribution_l1977_197748


namespace NUMINAMATH_CALUDE_triangle_height_l1977_197711

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 3 → area = 9 → area = (base * height) / 2 → height = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l1977_197711


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_l1977_197764

theorem integer_roots_of_cubic (a : ℤ) : 
  (∃ x : ℤ, x^3 + 3*x^2 + a*x + 8 = 0) ↔ 
  a ∈ ({-89, -39, -30, -14, -12, -6, -2, 10} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_l1977_197764


namespace NUMINAMATH_CALUDE_women_in_sports_club_l1977_197709

/-- The number of women in a sports club -/
def number_of_women (total_members participants : ℕ) : ℕ :=
  let women := 3 * (total_members - participants) / 2
  women

/-- Theorem: The number of women in the sports club is 21 -/
theorem women_in_sports_club :
  number_of_women 36 22 = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_women_in_sports_club_l1977_197709


namespace NUMINAMATH_CALUDE_problem_statement_l1977_197726

theorem problem_statement : (-4)^4 / 4^2 + 2^5 - 7^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1977_197726


namespace NUMINAMATH_CALUDE_worth_of_cloth_is_8540_l1977_197755

/-- Commission rates and sales data for an agent --/
structure SalesData where
  cloth_rate : ℝ
  electronics_rate_low : ℝ
  electronics_rate_high : ℝ
  electronics_threshold : ℝ
  stationery_rate_low : ℝ
  stationery_rate_high : ℝ
  stationery_threshold : ℕ
  total_commission : ℝ
  electronics_sales : ℝ
  stationery_units : ℕ

/-- Calculate the worth of cloth sold given sales data --/
def worth_of_cloth_sold (data : SalesData) : ℝ :=
  sorry

/-- Theorem stating that the worth of cloth sold is 8540 given the specific sales data --/
theorem worth_of_cloth_is_8540 (data : SalesData) 
  (h1 : data.cloth_rate = 0.025)
  (h2 : data.electronics_rate_low = 0.035)
  (h3 : data.electronics_rate_high = 0.045)
  (h4 : data.electronics_threshold = 3000)
  (h5 : data.stationery_rate_low = 10)
  (h6 : data.stationery_rate_high = 15)
  (h7 : data.stationery_threshold = 5)
  (h8 : data.total_commission = 418)
  (h9 : data.electronics_sales = 3100)
  (h10 : data.stationery_units = 8) :
  worth_of_cloth_sold data = 8540 := by
  sorry

end NUMINAMATH_CALUDE_worth_of_cloth_is_8540_l1977_197755


namespace NUMINAMATH_CALUDE_midpoint_chain_l1977_197740

/-- Given points A, B, C, D, E, F on a line segment AB, where:
    C is the midpoint of AB,
    D is the midpoint of AC,
    E is the midpoint of AD,
    F is the midpoint of AE,
    and AF = 3,
    prove that AB = 48. -/
theorem midpoint_chain (A B C D E F : ℝ) 
  (hC : C = (A + B) / 2) 
  (hD : D = (A + C) / 2)
  (hE : E = (A + D) / 2)
  (hF : F = (A + E) / 2)
  (hAF : F - A = 3) : 
  B - A = 48 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_chain_l1977_197740


namespace NUMINAMATH_CALUDE_y_coordinate_is_three_l1977_197770

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the second quadrant of the Cartesian coordinate system -/
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- Theorem: If a point is in the second quadrant and its distance to the x-axis is 3, then its y-coordinate is 3 -/
theorem y_coordinate_is_three (P : Point) 
  (h1 : second_quadrant P) 
  (h2 : distance_to_x_axis P = 3) : 
  P.y = 3 := by
  sorry

end NUMINAMATH_CALUDE_y_coordinate_is_three_l1977_197770


namespace NUMINAMATH_CALUDE_percentage_of_male_students_l1977_197701

theorem percentage_of_male_students
  (total_percentage : ℝ)
  (male_percentage : ℝ)
  (female_percentage : ℝ)
  (male_older_25 : ℝ)
  (female_older_25 : ℝ)
  (prob_younger_25 : ℝ)
  (h1 : total_percentage = male_percentage + female_percentage)
  (h2 : total_percentage = 100)
  (h3 : male_older_25 = 40)
  (h4 : female_older_25 = 20)
  (h5 : prob_younger_25 = 0.72)
  (h6 : prob_younger_25 = (1 - male_older_25 / 100) * male_percentage / 100 +
                          (1 - female_older_25 / 100) * female_percentage / 100) :
  male_percentage = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_male_students_l1977_197701


namespace NUMINAMATH_CALUDE_prob_green_two_containers_l1977_197781

/-- Represents a container with balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from two containers -/
def prob_green (a b : Container) : ℚ :=
  let total_a := a.red + a.green
  let total_b := b.red + b.green
  ((a.green : ℚ) / (2 * total_a : ℚ)) + ((b.green : ℚ) / (2 * total_b : ℚ))

/-- Theorem stating the probability of selecting a green ball -/
theorem prob_green_two_containers :
  let a : Container := ⟨5, 5⟩
  let b : Container := ⟨7, 3⟩
  prob_green a b = 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_prob_green_two_containers_l1977_197781


namespace NUMINAMATH_CALUDE_fiona_probability_l1977_197704

/-- Represents a lily pad with its number and whether it contains a predator -/
structure LilyPad where
  number : Nat
  hasPredator : Bool

/-- Represents Fiona's possible moves -/
inductive Move
  | Hop
  | Jump

/-- Represents the frog's journey -/
def FrogJourney := List Move

def numPads : Nat := 12

def predatorPads : List Nat := [3, 6]

def foodPad : Nat := 10

def startPad : Nat := 0

def moveProb : Rat := 1/2

/-- Calculates the final position after a sequence of moves -/
def finalPosition (journey : FrogJourney) : Nat :=
  journey.foldl (fun pos move =>
    match move with
    | Move.Hop => min (pos + 1) (numPads - 1)
    | Move.Jump => min (pos + 2) (numPads - 1)
  ) startPad

/-- Checks if a journey is safe (doesn't land on predator pads) -/
def isSafeJourney (journey : FrogJourney) : Bool :=
  let positions := List.scanl (fun pos move =>
    match move with
    | Move.Hop => min (pos + 1) (numPads - 1)
    | Move.Jump => min (pos + 2) (numPads - 1)
  ) startPad journey
  positions.all (fun pos => pos ∉ predatorPads)

/-- Calculates the probability of a specific journey -/
def journeyProbability (journey : FrogJourney) : Rat :=
  (moveProb ^ journey.length)

theorem fiona_probability :
  ∃ (successfulJourneys : List FrogJourney),
    (∀ j ∈ successfulJourneys, finalPosition j = foodPad ∧ isSafeJourney j) ∧
    (successfulJourneys.map journeyProbability).sum = 15/256 := by
  sorry

end NUMINAMATH_CALUDE_fiona_probability_l1977_197704


namespace NUMINAMATH_CALUDE_ticket_price_values_l1977_197798

theorem ticket_price_values (x : ℕ) : 
  (∃ (a b c : ℕ), x * a = 72 ∧ x * b = 90 ∧ x * c = 45) ↔ 
  (x = 1 ∨ x = 3 ∨ x = 9) := by
sorry

end NUMINAMATH_CALUDE_ticket_price_values_l1977_197798


namespace NUMINAMATH_CALUDE_ladder_length_l1977_197713

/-- Given a right triangle with an adjacent side of 6.4 meters and an angle of 59.5 degrees
    between the adjacent side and the hypotenuse, the length of the hypotenuse is
    approximately 12.43 meters. -/
theorem ladder_length (adjacent : ℝ) (angle : ℝ) (hypotenuse : ℝ) 
    (h_adjacent : adjacent = 6.4)
    (h_angle : angle = 59.5 * π / 180) -- Convert degrees to radians
    (h_cos : Real.cos angle = adjacent / hypotenuse) :
  abs (hypotenuse - 12.43) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ladder_length_l1977_197713


namespace NUMINAMATH_CALUDE_contractor_work_problem_l1977_197708

theorem contractor_work_problem (M : ℕ) : 
  (M * 6 = (M - 5) * 10) → M = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_contractor_work_problem_l1977_197708


namespace NUMINAMATH_CALUDE_sum_of_digits_82_l1977_197795

theorem sum_of_digits_82 :
  ∀ (tens ones : ℕ),
    tens * 10 + ones = 82 →
    tens - ones = 6 →
    tens + ones = 10 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_82_l1977_197795


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l1977_197773

theorem cubic_inequality_solution (x : ℝ) :
  x^3 + x^2 - 7*x + 6 < 0 ↔ -3 < x ∧ x < 1 ∨ 1 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l1977_197773


namespace NUMINAMATH_CALUDE_circle_parameter_value_l1977_197753

theorem circle_parameter_value (θ : Real) : 
  0 ≤ θ ∧ θ < 2 * Real.pi →
  4 * Real.cos θ = -2 →
  4 * Real.sin θ = 2 * Real.sqrt 3 →
  θ = 2 * Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_circle_parameter_value_l1977_197753


namespace NUMINAMATH_CALUDE_min_value_a_l1977_197794

theorem min_value_a (a b c d : ℕ+) 
  (h1 : a > b ∧ b > c ∧ c > d)
  (h2 : a + b + c + d = 2004)
  (h3 : a^2 - b^2 + c^2 - d^2 = 2004) :
  ∀ x : ℕ+, (x > b ∧ b > c ∧ c > d ∧ 
             x + b + c + d = 2004 ∧ 
             x^2 - b^2 + c^2 - d^2 = 2004) → 
    x ≥ 503 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l1977_197794


namespace NUMINAMATH_CALUDE_stability_comparison_l1977_197716

-- Define the concept of a data set
def DataSet := List ℝ

-- Define the variance of a data set
def variance (s : DataSet) : ℝ := sorry

-- Define the concept of stability for a data set
def is_more_stable (s1 s2 : DataSet) : Prop := 
  variance s1 < variance s2

-- Theorem statement
theorem stability_comparison (A B : DataSet) 
  (h_mean : (A.sum / A.length) = (B.sum / B.length))
  (h_var_A : variance A = 0.3)
  (h_var_B : variance B = 0.02) :
  is_more_stable B A := by sorry

end NUMINAMATH_CALUDE_stability_comparison_l1977_197716


namespace NUMINAMATH_CALUDE_strings_per_normal_guitar_is_6_l1977_197734

/-- Calculates the number of strings on each normal guitar given the following conditions:
  * There are 3 basses with 4 strings each
  * There are twice as many normal guitars as basses
  * There are 3 fewer 8-string guitars than normal guitars
  * The total number of strings needed is 72
-/
def strings_per_normal_guitar : ℕ :=
  let num_basses : ℕ := 3
  let strings_per_bass : ℕ := 4
  let num_normal_guitars : ℕ := 2 * num_basses
  let num_8string_guitars : ℕ := num_normal_guitars - 3
  let total_strings : ℕ := 72
  (total_strings - num_basses * strings_per_bass - num_8string_guitars * 8) / num_normal_guitars

theorem strings_per_normal_guitar_is_6 : strings_per_normal_guitar = 6 := by
  sorry

end NUMINAMATH_CALUDE_strings_per_normal_guitar_is_6_l1977_197734
