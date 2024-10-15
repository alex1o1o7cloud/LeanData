import Mathlib

namespace NUMINAMATH_CALUDE_men_absent_l846_84601

/-- Proves that 15 men became absent given the original group size, planned completion time, and actual completion time. -/
theorem men_absent (total_men : ℕ) (planned_days : ℕ) (actual_days : ℕ) 
  (h1 : total_men = 180) 
  (h2 : planned_days = 55)
  (h3 : actual_days = 60) :
  ∃ (absent_men : ℕ), 
    absent_men = 15 ∧ 
    (total_men * planned_days = (total_men - absent_men) * actual_days) :=
by
  sorry

#check men_absent

end NUMINAMATH_CALUDE_men_absent_l846_84601


namespace NUMINAMATH_CALUDE_joan_seashells_l846_84655

/-- The number of seashells Joan has after giving some away -/
def remaining_seashells (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Proof that Joan has 16 seashells after giving 63 away from her initial 79 -/
theorem joan_seashells : remaining_seashells 79 63 = 16 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_l846_84655


namespace NUMINAMATH_CALUDE_first_fibonacci_exceeding_1000_l846_84675

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem first_fibonacci_exceeding_1000 :
  ∃ n : ℕ, fibonacci n > 1000 ∧ ∀ m : ℕ, m < n → fibonacci m ≤ 1000 ∧ fibonacci n = 1597 :=
by
  sorry

end NUMINAMATH_CALUDE_first_fibonacci_exceeding_1000_l846_84675


namespace NUMINAMATH_CALUDE_total_legs_is_108_l846_84659

-- Define the number of each animal
def num_birds : ℕ := 3
def num_dogs : ℕ := 5
def num_snakes : ℕ := 4
def num_spiders : ℕ := 1
def num_horses : ℕ := 2
def num_rabbits : ℕ := 6
def num_octopuses : ℕ := 3
def num_ants : ℕ := 7

-- Define the number of legs for each animal type
def legs_bird : ℕ := 2
def legs_dog : ℕ := 4
def legs_snake : ℕ := 0
def legs_spider : ℕ := 8
def legs_horse : ℕ := 4
def legs_rabbit : ℕ := 4
def legs_octopus : ℕ := 0
def legs_ant : ℕ := 6

-- Theorem to prove
theorem total_legs_is_108 : 
  num_birds * legs_bird + 
  num_dogs * legs_dog + 
  num_snakes * legs_snake + 
  num_spiders * legs_spider + 
  num_horses * legs_horse + 
  num_rabbits * legs_rabbit + 
  num_octopuses * legs_octopus + 
  num_ants * legs_ant = 108 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_is_108_l846_84659


namespace NUMINAMATH_CALUDE_negation_of_proposition_l846_84697

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), ab > 0 → a > 0) ↔ (∀ (a b : ℝ), ab ≤ 0 → a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l846_84697


namespace NUMINAMATH_CALUDE_stream_speed_l846_84690

/-- The speed of a stream given downstream and upstream speeds -/
theorem stream_speed (downstream_speed upstream_speed : ℝ) :
  downstream_speed = 15 →
  upstream_speed = 8 →
  (downstream_speed - upstream_speed) / 2 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l846_84690


namespace NUMINAMATH_CALUDE_theater_ticket_cost_l846_84699

/-- The cost of tickets at a theater -/
theorem theater_ticket_cost 
  (adult_price : ℝ) 
  (h1 : adult_price > 0)
  (h2 : 4 * adult_price + 3 * (adult_price / 2) + 2 * (3 * adult_price / 4) = 42) :
  6 * adult_price + 5 * (adult_price / 2) + 4 * (3 * adult_price / 4) = 69 := by
  sorry


end NUMINAMATH_CALUDE_theater_ticket_cost_l846_84699


namespace NUMINAMATH_CALUDE_complex_number_simplification_l846_84646

theorem complex_number_simplification :
  3 * (2 - 5 * Complex.I) - 4 * (1 + 3 * Complex.I) = 2 - 27 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l846_84646


namespace NUMINAMATH_CALUDE_fraction_of_powers_equals_3125_l846_84612

theorem fraction_of_powers_equals_3125 : (125000 ^ 5) / (25000 ^ 5) = 3125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_powers_equals_3125_l846_84612


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l846_84609

theorem incorrect_observation_value 
  (n : ℕ) 
  (initial_mean correct_value new_mean : ℝ) 
  (h_n : n = 50)
  (h_initial : initial_mean = 36)
  (h_correct : correct_value = 46)
  (h_new : new_mean = 36.5) :
  ∃ (incorrect_value : ℝ),
    n * new_mean = (n - 1) * initial_mean + correct_value ∧
    incorrect_value = initial_mean * n - (n - 1) * initial_mean - correct_value ∧
    incorrect_value = 21 := by sorry

end NUMINAMATH_CALUDE_incorrect_observation_value_l846_84609


namespace NUMINAMATH_CALUDE_ball_color_equality_l846_84642

theorem ball_color_equality (r g b : ℕ) : 
  (r + g + b = 20) →
  (b ≥ 7) →
  (r ≥ 4) →
  (b = 2 * g) →
  (r = b ∨ r = g) :=
by sorry

end NUMINAMATH_CALUDE_ball_color_equality_l846_84642


namespace NUMINAMATH_CALUDE_heaviest_tv_weight_difference_l846_84695

-- Define the dimensions and weight ratios of the TVs
def bill_width : ℝ := 48
def bill_height : ℝ := 100
def bill_weight_ratio : ℝ := 4

def bob_width : ℝ := 70
def bob_height : ℝ := 60
def bob_weight_ratio : ℝ := 3.5

def steve_width : ℝ := 84
def steve_height : ℝ := 92
def steve_weight_ratio : ℝ := 4.5

-- Define the conversion factor from ounces to pounds
def oz_to_lb : ℝ := 16

-- Theorem to prove
theorem heaviest_tv_weight_difference : 
  let bill_area := bill_width * bill_height
  let bob_area := bob_width * bob_height
  let steve_area := steve_width * steve_height
  
  let bill_weight := bill_area * bill_weight_ratio / oz_to_lb
  let bob_weight := bob_area * bob_weight_ratio / oz_to_lb
  let steve_weight := steve_area * steve_weight_ratio / oz_to_lb
  
  let heaviest_weight := max bill_weight (max bob_weight steve_weight)
  let combined_weight := bill_weight + bob_weight
  
  heaviest_weight - combined_weight = 54.75 := by
  sorry

end NUMINAMATH_CALUDE_heaviest_tv_weight_difference_l846_84695


namespace NUMINAMATH_CALUDE_yarns_are_zorps_and_xings_l846_84673

variable (U : Type) -- Universe set

-- Define the subsets
variable (Zorp Xing Yarn Wit Vamp : Set U)

-- State the given conditions
variable (h1 : Zorp ⊆ Xing)
variable (h2 : Yarn ⊆ Xing)
variable (h3 : Wit ⊆ Zorp)
variable (h4 : Yarn ⊆ Wit)
variable (h5 : Yarn ⊆ Vamp)

-- Theorem to prove
theorem yarns_are_zorps_and_xings : Yarn ⊆ Zorp ∧ Yarn ⊆ Xing := by sorry

end NUMINAMATH_CALUDE_yarns_are_zorps_and_xings_l846_84673


namespace NUMINAMATH_CALUDE_arrangements_not_adjacent_l846_84653

theorem arrangements_not_adjacent (n : ℕ) (h : n = 5) : 
  (n.factorial - 2 * (n - 1).factorial) = 72 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_not_adjacent_l846_84653


namespace NUMINAMATH_CALUDE_chlorine_moles_l846_84665

/-- Represents the chemical reaction between Methane and Chlorine to produce Hydrochloric acid -/
def chemical_reaction (methane : ℝ) (chlorine : ℝ) (hydrochloric_acid : ℝ) : Prop :=
  methane = 1 ∧ hydrochloric_acid = 2 ∧ chlorine = hydrochloric_acid

/-- Theorem stating that 2 moles of Chlorine are combined in the reaction -/
theorem chlorine_moles : ∃ (chlorine : ℝ), chemical_reaction 1 chlorine 2 ∧ chlorine = 2 := by
  sorry

end NUMINAMATH_CALUDE_chlorine_moles_l846_84665


namespace NUMINAMATH_CALUDE_jerry_shelf_difference_l846_84631

/-- The difference between action figures and books on Jerry's shelf -/
def shelf_difference (initial_figures : ℕ) (initial_books : ℕ) (added_books : ℕ) : ℤ :=
  (initial_figures : ℤ) - ((initial_books : ℤ) + (added_books : ℤ))

/-- Theorem stating the difference between action figures and books on Jerry's shelf -/
theorem jerry_shelf_difference :
  shelf_difference 7 2 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_jerry_shelf_difference_l846_84631


namespace NUMINAMATH_CALUDE_nested_expression_value_l846_84605

theorem nested_expression_value : (3 * (3 * (3 * (3 * (3 * (3 + 1) + 2) + 3) + 4) + 5) + 6) = 1272 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l846_84605


namespace NUMINAMATH_CALUDE_novel_sales_theorem_l846_84657

/-- Represents the sale of a novel in hardback and paperback versions -/
structure NovelSales where
  hardback_before_paperback : ℕ
  paperback_total : ℕ
  paperback_to_hardback_ratio : ℕ

/-- Calculates the total number of copies sold given the sales data -/
def total_copies_sold (sales : NovelSales) : ℕ :=
  sales.hardback_before_paperback + 
  sales.paperback_total + 
  (sales.paperback_total / sales.paperback_to_hardback_ratio)

/-- Theorem stating that given the conditions, the total number of copies sold is 440400 -/
theorem novel_sales_theorem (sales : NovelSales) 
  (h1 : sales.hardback_before_paperback = 36000)
  (h2 : sales.paperback_to_hardback_ratio = 9)
  (h3 : sales.paperback_total = 363600) :
  total_copies_sold sales = 440400 := by
  sorry

#eval total_copies_sold ⟨36000, 363600, 9⟩

end NUMINAMATH_CALUDE_novel_sales_theorem_l846_84657


namespace NUMINAMATH_CALUDE_max_c_value_l846_84614

noncomputable section

def is_valid_solution (a b c x y z : ℝ) : Prop :=
  a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  a^x + b^y + c^z = 4 ∧
  x * a^x + y * b^y + z * c^z = 6 ∧
  x^2 * a^x + y^2 * b^y + z^2 * c^z = 9

theorem max_c_value (a b c x y z : ℝ) (h : is_valid_solution a b c x y z) :
  c ≤ Real.rpow 4 (1/3) :=
sorry

end

end NUMINAMATH_CALUDE_max_c_value_l846_84614


namespace NUMINAMATH_CALUDE_sphere_part_volume_l846_84617

theorem sphere_part_volume (circumference : ℝ) (h : circumference = 18 * Real.pi) :
  let radius := circumference / (2 * Real.pi)
  let sphere_volume := (4 / 3) * Real.pi * radius ^ 3
  let part_volume := sphere_volume / 6
  part_volume = 162 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_part_volume_l846_84617


namespace NUMINAMATH_CALUDE_hamburgers_served_l846_84666

/-- Proves that the number of hamburgers served is 3, given the total made and left over. -/
theorem hamburgers_served (total : ℕ) (leftover : ℕ) (h1 : total = 9) (h2 : leftover = 6) :
  total - leftover = 3 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_served_l846_84666


namespace NUMINAMATH_CALUDE_parabola_intersection_l846_84625

theorem parabola_intersection
  (a h d : ℝ) (ha : a ≠ 0) :
  let f (x : ℝ) := a * (x - h)^2 + d
  let g (x : ℝ) := a * ((x + 3) - h)^2 + d
  ∃! x, f x = g x ∧ x = -3/2 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_l846_84625


namespace NUMINAMATH_CALUDE_arcsin_cos_arcsin_plus_arccos_sin_arccos_l846_84639

theorem arcsin_cos_arcsin_plus_arccos_sin_arccos (x : ℝ) (h : x ∈ Set.Icc (-1) 1) :
  Real.arcsin (Real.cos (Real.arcsin x)) + Real.arccos (Real.sin (Real.arccos x)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_cos_arcsin_plus_arccos_sin_arccos_l846_84639


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_is_eleven_fourths_l846_84641

/-- The repeating decimal 0.363636... as a rational number -/
def repeating_decimal : ℚ := 4 / 11

/-- The reciprocal of the repeating decimal 0.363636... -/
def reciprocal_of_repeating_decimal : ℚ := 11 / 4

/-- Theorem: The reciprocal of the common fraction form of 0.363636... is 11/4 -/
theorem reciprocal_of_repeating_decimal_is_eleven_fourths :
  (1 : ℚ) / repeating_decimal = reciprocal_of_repeating_decimal := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_is_eleven_fourths_l846_84641


namespace NUMINAMATH_CALUDE_translated_parabola_vertex_l846_84606

/-- The vertex of a translated parabola -/
theorem translated_parabola_vertex :
  let f (x : ℝ) := -(x - 3)^2 - 2
  ∃! (h k : ℝ), (∀ x, f x = -(x - h)^2 + k) ∧ h = 3 ∧ k = -2 :=
sorry

end NUMINAMATH_CALUDE_translated_parabola_vertex_l846_84606


namespace NUMINAMATH_CALUDE_milk_water_ratio_after_filling_l846_84679

/-- Represents the ratio of milk to water -/
structure Ratio where
  milk : ℕ
  water : ℕ

/-- Represents the can with its contents -/
structure Can where
  capacity : ℕ
  current_volume : ℕ
  ratio : Ratio

def initial_can : Can :=
  { capacity := 60
  , current_volume := 40
  , ratio := { milk := 5, water := 3 } }

def final_can : Can :=
  { capacity := 60
  , current_volume := 60
  , ratio := { milk := 3, water := 1 } }

theorem milk_water_ratio_after_filling (c : Can) (h : c = initial_can) :
  (final_can.ratio.milk : ℚ) / final_can.ratio.water = 3 := by
  sorry

#check milk_water_ratio_after_filling

end NUMINAMATH_CALUDE_milk_water_ratio_after_filling_l846_84679


namespace NUMINAMATH_CALUDE_certain_number_subtraction_l846_84616

theorem certain_number_subtraction (X : ℤ) (h : X - 46 = 15) : X - 29 = 32 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_subtraction_l846_84616


namespace NUMINAMATH_CALUDE_pen_discount_theorem_l846_84689

theorem pen_discount_theorem (marked_price : ℝ) :
  let purchase_quantity : ℕ := 60
  let purchase_price_in_pens : ℕ := 46
  let profit_percent : ℝ := 29.130434782608695

  let cost_price : ℝ := marked_price * purchase_price_in_pens
  let selling_price : ℝ := cost_price * (1 + profit_percent / 100)
  let selling_price_per_pen : ℝ := selling_price / purchase_quantity
  let discount : ℝ := marked_price - selling_price_per_pen
  let discount_percent : ℝ := (discount / marked_price) * 100

  discount_percent = 1 := by sorry

end NUMINAMATH_CALUDE_pen_discount_theorem_l846_84689


namespace NUMINAMATH_CALUDE_ramu_profit_percent_l846_84669

/-- Calculates the profit percent given the cost of a car, repair costs, and selling price --/
def profit_percent (car_cost repair_cost selling_price : ℚ) : ℚ :=
  let total_cost := car_cost + repair_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

/-- Theorem stating that under the given conditions, the profit percent is 18% --/
theorem ramu_profit_percent :
  profit_percent 42000 13000 64900 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ramu_profit_percent_l846_84669


namespace NUMINAMATH_CALUDE_total_weight_of_fruits_l846_84692

-- Define the weight of oranges and apples
def orange_weight : ℚ := 24 / 12
def apple_weight : ℚ := 30 / 8

-- Define the number of bags for each fruit
def orange_bags : ℕ := 5
def apple_bags : ℕ := 4

-- Theorem to prove
theorem total_weight_of_fruits :
  orange_bags * orange_weight + apple_bags * apple_weight = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_fruits_l846_84692


namespace NUMINAMATH_CALUDE_number_card_problem_l846_84682

theorem number_card_problem (A B C : ℝ) : 
  (A + B + C) / 3 = 143 →
  A + 4.5 = (B + C) / 2 →
  C = B - 3 →
  C = 143 := by sorry

end NUMINAMATH_CALUDE_number_card_problem_l846_84682


namespace NUMINAMATH_CALUDE_flagpole_break_height_l846_84661

/-- 
Given a flagpole of height 8 meters that breaks such that the upper part touches the ground 3 meters from the base, 
the height from the ground to the break point is √73/2 meters.
-/
theorem flagpole_break_height (h : ℝ) (d : ℝ) (x : ℝ) 
  (h_height : h = 8) 
  (d_distance : d = 3) 
  (x_def : x = h - (h^2 - d^2).sqrt / 2) : 
  x = Real.sqrt 73 / 2 := by
  sorry

end NUMINAMATH_CALUDE_flagpole_break_height_l846_84661


namespace NUMINAMATH_CALUDE_sum_f_odd_points_l846_84684

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_zero : f 0 = 2
axiom f_translated_odd : ∀ x, f (x - 1) = -f (-x - 1)

-- State the theorem
theorem sum_f_odd_points :
  f 1 + f 3 + f 5 + f 7 + f 9 = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_f_odd_points_l846_84684


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_6_with_digit_sum_15_l846_84634

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_6_with_digit_sum_15 :
  ∀ n : ℕ, is_three_digit n → n % 6 = 0 → digit_sum n = 15 → n ≤ 960 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_6_with_digit_sum_15_l846_84634


namespace NUMINAMATH_CALUDE_greatest_x_value_l846_84633

theorem greatest_x_value : ∃ (x : ℤ), (∀ (y : ℤ), 2.134 * (10 : ℝ) ^ y < 21000 → y ≤ x) ∧ 2.134 * (10 : ℝ) ^ x < 21000 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_greatest_x_value_l846_84633


namespace NUMINAMATH_CALUDE_equation_solution_l846_84636

theorem equation_solution (c d : ℝ) (h : d ≠ 0) :
  let x := (9 * d^2 - 4 * c^2) / (6 * d)
  x^2 + 4 * c^2 = (3 * d - x)^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l846_84636


namespace NUMINAMATH_CALUDE_pencil_theorem_l846_84604

def pencil_problem (jayden marcus dana ella : ℕ) : Prop :=
  jayden = 20 ∧
  dana = jayden + 15 ∧
  jayden = 2 * marcus ∧
  ella = 3 * marcus - 5 ∧
  dana = marcus + ella

theorem pencil_theorem :
  ∀ jayden marcus dana ella : ℕ,
  pencil_problem jayden marcus dana ella →
  dana = marcus + ella :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_theorem_l846_84604


namespace NUMINAMATH_CALUDE_stock_worth_l846_84670

/-- The total worth of a stock given specific sales conditions and overall loss -/
theorem stock_worth (stock : ℝ) : 
  (0.2 * stock * 1.1 + 0.8 * stock * 0.95 = stock - 450) → 
  stock = 22500 := by
sorry

end NUMINAMATH_CALUDE_stock_worth_l846_84670


namespace NUMINAMATH_CALUDE_parabola_properties_l846_84619

/-- Parabola C: y^2 = x with focus F -/
structure Parabola where
  focus : ℝ × ℝ
  equation : (ℝ × ℝ) → Prop

/-- Point on the parabola -/
structure PointOnParabola (C : Parabola) where
  point : ℝ × ℝ
  on_parabola : C.equation point

/-- Theorem about the slope of line AB and the length of AB when collinear with focus -/
theorem parabola_properties (C : Parabola) 
    (hC : C.focus = (1/4, 0) ∧ C.equation = fun p => p.2^2 = p.1) 
    (A B : PointOnParabola C) 
    (hAB : A.point ≠ B.point ∧ A.point ≠ (0, 0) ∧ B.point ≠ (0, 0)) :
  (∃ k : ℝ, k = (A.point.2 - B.point.2) / (A.point.1 - B.point.1) → 
    k = 1 / (A.point.2 + B.point.2)) ∧
  (∃ AB : ℝ, (∃ t : ℝ, (1 - t) • A.point + t • B.point = C.focus) → 
    AB = A.point.1 + B.point.1 + 1/2) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l846_84619


namespace NUMINAMATH_CALUDE_afternoon_rowers_l846_84649

theorem afternoon_rowers (morning evening total : ℕ) 
  (h1 : morning = 36)
  (h2 : evening = 49)
  (h3 : total = 98)
  : total - morning - evening = 13 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_rowers_l846_84649


namespace NUMINAMATH_CALUDE_invertible_function_theorem_l846_84686

noncomputable section

-- Define the function g
def g : ℝ → ℝ := sorry

-- State the theorem
theorem invertible_function_theorem (c d : ℝ) 
  (h1 : Function.Injective g) 
  (h2 : g c = d) 
  (h3 : g d = 5) : 
  c - d = -2 := by sorry

end NUMINAMATH_CALUDE_invertible_function_theorem_l846_84686


namespace NUMINAMATH_CALUDE_tileC_in_rectangleY_l846_84618

-- Define a tile with four sides
structure Tile :=
  (top : ℕ) (right : ℕ) (bottom : ℕ) (left : ℕ)

-- Define the four tiles
def tileA : Tile := ⟨5, 3, 1, 6⟩
def tileB : Tile := ⟨3, 6, 2, 5⟩
def tileC : Tile := ⟨2, 7, 0, 3⟩
def tileD : Tile := ⟨6, 2, 4, 7⟩

-- Define a function to check if a tile has unique sides
def hasUniqueSides (t : Tile) (others : List Tile) : Prop :=
  (t.right ∉ others.map (λ tile => tile.left)) ∧
  (t.bottom ∉ others.map (λ tile => tile.top))

-- Define the theorem
theorem tileC_in_rectangleY :
  hasUniqueSides tileC [tileA, tileB, tileD] ∧
  ¬hasUniqueSides tileA [tileB, tileC, tileD] ∧
  ¬hasUniqueSides tileB [tileA, tileC, tileD] ∧
  ¬hasUniqueSides tileD [tileA, tileB, tileC] :=
sorry

end NUMINAMATH_CALUDE_tileC_in_rectangleY_l846_84618


namespace NUMINAMATH_CALUDE_anya_vanya_catchup_l846_84632

/-- Represents the speeds and catch-up times in the Anya-Vanya problem -/
structure AnyaVanyaProblem where
  anya_speed : ℝ
  vanya_speed : ℝ
  original_catch_up_time : ℝ

/-- The conditions of the problem -/
def problem_conditions (p : AnyaVanyaProblem) : Prop :=
  p.anya_speed > 0 ∧ 
  p.vanya_speed > p.anya_speed ∧
  p.original_catch_up_time > 0 ∧
  (2 * p.vanya_speed - p.anya_speed) * p.original_catch_up_time = 
    3 * (p.vanya_speed - p.anya_speed) * (p.original_catch_up_time / 3)

/-- The theorem to be proved -/
theorem anya_vanya_catchup (p : AnyaVanyaProblem) 
  (h : problem_conditions p) : 
  (2 * p.vanya_speed - p.anya_speed / 2) * (p.original_catch_up_time / 7) = 
  (p.vanya_speed - p.anya_speed) * p.original_catch_up_time :=
by sorry

end NUMINAMATH_CALUDE_anya_vanya_catchup_l846_84632


namespace NUMINAMATH_CALUDE_max_value_a_l846_84626

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 2 * b) 
  (h2 : b < 3 * c) 
  (h3 : c < 2 * d) 
  (h4 : d < 50) : 
  a ≤ 579 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 579 ∧ 
    a' < 2 * b' ∧ 
    b' < 3 * c' ∧ 
    c' < 2 * d' ∧ 
    d' < 50 :=
sorry

end NUMINAMATH_CALUDE_max_value_a_l846_84626


namespace NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l846_84681

theorem largest_prime_divisor_to_test (n : ℕ) (h : 500 ≤ n ∧ n ≤ 550) :
  (∀ p : ℕ, p.Prime → p ≤ 23 → ¬(p ∣ n)) → n.Prime ∨ n = 1 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l846_84681


namespace NUMINAMATH_CALUDE_jungkook_red_balls_l846_84671

/-- The number of boxes Jungkook bought -/
def num_boxes : ℕ := 2

/-- The number of red balls in each box -/
def balls_per_box : ℕ := 3

/-- The total number of red balls Jungkook has -/
def total_balls : ℕ := num_boxes * balls_per_box

theorem jungkook_red_balls : total_balls = 6 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_red_balls_l846_84671


namespace NUMINAMATH_CALUDE_quadratic_factorization_l846_84694

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l846_84694


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l846_84691

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.tan (60 * π / 180) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l846_84691


namespace NUMINAMATH_CALUDE_age_sum_problem_l846_84685

theorem age_sum_problem (a b c : ℕ+) (h1 : a = b) (h2 : a > c) (h3 : a * b * c = 144) :
  a + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_problem_l846_84685


namespace NUMINAMATH_CALUDE_five_balls_two_boxes_l846_84630

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_balls + 1

/-- Theorem: There are 6 ways to distribute 5 indistinguishable balls into 2 distinguishable boxes -/
theorem five_balls_two_boxes : distribute_balls 5 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_two_boxes_l846_84630


namespace NUMINAMATH_CALUDE_tenth_occurrence_shift_l846_84663

/-- Represents the number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- Calculates the shift for the nth occurrence of a letter -/
def shift (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2 + 1

/-- Theorem: The 10th occurrence of a letter is replaced by the letter 13 positions to its right -/
theorem tenth_occurrence_shift :
  shift 10 % alphabet_size = 13 :=
sorry

end NUMINAMATH_CALUDE_tenth_occurrence_shift_l846_84663


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l846_84644

/-- Given vectors a, b, and c in ℝ², prove that if a is perpendicular to (b - c), then x = 4/3 -/
theorem perpendicular_vectors (a b c : ℝ × ℝ) (x : ℝ) 
  (h1 : a = (3, 1)) 
  (h2 : b = (x, -2)) 
  (h3 : c = (0, 2)) 
  (h4 : a • (b - c) = 0) : 
  x = 4/3 := by
  sorry

#check perpendicular_vectors

end NUMINAMATH_CALUDE_perpendicular_vectors_l846_84644


namespace NUMINAMATH_CALUDE_fourth_root_ten_million_l846_84608

theorem fourth_root_ten_million (x : ℝ) : x = 10 * (10 ^ (1/4)) → x^4 = 10000000 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_ten_million_l846_84608


namespace NUMINAMATH_CALUDE_train_length_calculation_l846_84615

/-- The length of a train that crosses an electric pole in a given time at a given speed. -/
def train_length (crossing_time : ℝ) (speed : ℝ) : ℝ :=
  crossing_time * speed

/-- Theorem: A train that crosses an electric pole in 40 seconds at a speed of 62.99999999999999 m/s has a length of 2520 meters. -/
theorem train_length_calculation :
  train_length 40 62.99999999999999 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l846_84615


namespace NUMINAMATH_CALUDE_sin_330_degrees_l846_84621

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l846_84621


namespace NUMINAMATH_CALUDE_rainy_days_exist_l846_84688

/-- Represents the number of rainy days given the conditions of Mo's drinking habits -/
def rainy_days (n d T H P : ℤ) : Prop :=
  ∃ (R : ℤ),
    (1 ≤ d) ∧ (d ≤ 31) ∧
    (T = 3 * (d - R)) ∧
    (H = n * R) ∧
    (T = H + P) ∧
    (R = (3 * d - P) / (n + 3)) ∧
    (0 ≤ R) ∧ (R ≤ d)

/-- Theorem stating the existence of R satisfying the conditions -/
theorem rainy_days_exist (n d T H P : ℤ) (h1 : 1 ≤ d) (h2 : d ≤ 31) 
  (h3 : T = 3 * (d - (3 * d - P) / (n + 3))) 
  (h4 : H = n * ((3 * d - P) / (n + 3))) 
  (h5 : T = H + P)
  (h6 : (3 * d - P) % (n + 3) = 0)
  (h7 : 0 ≤ (3 * d - P) / (n + 3))
  (h8 : (3 * d - P) / (n + 3) ≤ d) :
  rainy_days n d T H P :=
by
  sorry


end NUMINAMATH_CALUDE_rainy_days_exist_l846_84688


namespace NUMINAMATH_CALUDE_umbrella_arrangements_seven_l846_84623

def umbrella_arrangements (n : ℕ) : ℕ := 
  if n % 2 = 0 then 0
  else Nat.choose (n - 1) ((n - 1) / 2)

theorem umbrella_arrangements_seven :
  umbrella_arrangements 7 = 20 := by
sorry

end NUMINAMATH_CALUDE_umbrella_arrangements_seven_l846_84623


namespace NUMINAMATH_CALUDE_multiples_of_6_or_8_not_both_count_multiples_6_or_8_not_both_l846_84638

theorem multiples_of_6_or_8_not_both (n : Nat) : 
  (Finset.filter (fun x => (x % 6 = 0 ∨ x % 8 = 0) ∧ ¬(x % 6 = 0 ∧ x % 8 = 0)) (Finset.range n)).card = 
  (Finset.filter (fun x => x % 6 = 0) (Finset.range n)).card + 
  (Finset.filter (fun x => x % 8 = 0) (Finset.range n)).card - 
  (Finset.filter (fun x => x % 24 = 0) (Finset.range n)).card :=
by sorry

theorem count_multiples_6_or_8_not_both : 
  (Finset.filter (fun x => (x % 6 = 0 ∨ x % 8 = 0) ∧ ¬(x % 6 = 0 ∧ x % 8 = 0)) (Finset.range 201)).card = 42 :=
by sorry

end NUMINAMATH_CALUDE_multiples_of_6_or_8_not_both_count_multiples_6_or_8_not_both_l846_84638


namespace NUMINAMATH_CALUDE_systematic_sampling_18th_group_l846_84647

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (firstSelected : ℕ) (groupNumber : ℕ) : ℕ :=
  firstSelected + (groupNumber - 1) * (totalStudents / sampleSize)

/-- Theorem: Systematic sampling selects student number 872 in the 18th group -/
theorem systematic_sampling_18th_group :
  let totalStudents : ℕ := 1000
  let sampleSize : ℕ := 20
  let groupSize : ℕ := totalStudents / sampleSize
  let thirdGroupStart : ℕ := 2 * groupSize
  let selectedInThirdGroup : ℕ := 122
  let firstSelected : ℕ := selectedInThirdGroup - thirdGroupStart
  systematicSample totalStudents sampleSize firstSelected 18 = 872 := by
sorry

end NUMINAMATH_CALUDE_systematic_sampling_18th_group_l846_84647


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l846_84654

theorem quadratic_roots_ratio (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l846_84654


namespace NUMINAMATH_CALUDE_difference_of_decimal_and_fraction_l846_84628

theorem difference_of_decimal_and_fraction : 0.650 - (1 / 8 : ℚ) = 0.525 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_decimal_and_fraction_l846_84628


namespace NUMINAMATH_CALUDE_james_muffins_l846_84660

theorem james_muffins (arthur_muffins : Float) (ratio : Float) (james_muffins : Float)
  (h1 : arthur_muffins = 115.0)
  (h2 : ratio = 12.0)
  (h3 : arthur_muffins = ratio * james_muffins) :
  james_muffins = arthur_muffins / ratio := by
sorry

end NUMINAMATH_CALUDE_james_muffins_l846_84660


namespace NUMINAMATH_CALUDE_train_distance_proof_l846_84648

/-- Calculates the distance a train can travel given its coal efficiency and available coal. -/
def trainDistance (milesPerCoal : ℚ) (availableCoal : ℚ) : ℚ :=
  milesPerCoal * availableCoal

/-- Proves that a train with given efficiency and coal amount can travel 400 miles. -/
theorem train_distance_proof :
  let milesPerCoal : ℚ := 5 / 2
  let availableCoal : ℚ := 160
  trainDistance milesPerCoal availableCoal = 400 := by
  sorry

#eval trainDistance (5 / 2) 160

end NUMINAMATH_CALUDE_train_distance_proof_l846_84648


namespace NUMINAMATH_CALUDE_layla_goals_l846_84674

theorem layla_goals (layla kristin : ℕ) (h1 : kristin = layla - 24) (h2 : layla + kristin = 368) : layla = 196 := by
  sorry

end NUMINAMATH_CALUDE_layla_goals_l846_84674


namespace NUMINAMATH_CALUDE_coins_sum_theorem_l846_84678

theorem coins_sum_theorem (stack1 stack2 stack3 stack4 : ℕ) 
  (h1 : stack1 = 12)
  (h2 : stack2 = 17)
  (h3 : stack3 = 23)
  (h4 : stack4 = 8) :
  stack1 + stack2 + stack3 + stack4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_coins_sum_theorem_l846_84678


namespace NUMINAMATH_CALUDE_departure_sequences_count_l846_84645

/-- The number of trains --/
def num_trains : ℕ := 6

/-- The number of groups --/
def num_groups : ℕ := 2

/-- The number of trains per group --/
def trains_per_group : ℕ := 3

/-- The number of fixed trains (A and B) in the first group --/
def fixed_trains : ℕ := 2

/-- Theorem: The number of different departure sequences for the trains --/
theorem departure_sequences_count : 
  (num_trains - fixed_trains - trains_per_group) * 
  (Nat.factorial trains_per_group) * 
  (Nat.factorial trains_per_group) = 144 := by
  sorry

end NUMINAMATH_CALUDE_departure_sequences_count_l846_84645


namespace NUMINAMATH_CALUDE_base8_to_base10_conversion_l846_84650

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (8 ^ i)) 0

/-- The base-8 representation of the number --/
def base8Number : List Nat := [3, 4, 6, 2, 5]

theorem base8_to_base10_conversion :
  base8ToBase10 base8Number = 21923 := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base10_conversion_l846_84650


namespace NUMINAMATH_CALUDE_planet_coloring_theorem_specific_planet_coloring_case_l846_84602

/-- The number of colors needed for planet coloring -/
def colors_needed (num_planets : ℕ) (num_people : ℕ) : ℕ :=
  num_planets * num_people

/-- Theorem: In the planet coloring scenario, the number of colors needed
    is equal to the number of planets multiplied by the number of people coloring. -/
theorem planet_coloring_theorem (num_planets : ℕ) (num_people : ℕ) :
  colors_needed num_planets num_people = num_planets * num_people :=
by
  sorry

/-- The specific case mentioned in the problem -/
theorem specific_planet_coloring_case :
  colors_needed 8 3 = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_planet_coloring_theorem_specific_planet_coloring_case_l846_84602


namespace NUMINAMATH_CALUDE_completing_square_sum_l846_84696

theorem completing_square_sum (a b c : ℤ) : 
  (∀ x : ℝ, 49 * x^2 + 70 * x - 121 = 0 ↔ (a * x + b)^2 = c) →
  a > 0 →
  a + b + c = 158 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_sum_l846_84696


namespace NUMINAMATH_CALUDE_medical_team_combinations_l846_84600

theorem medical_team_combinations (n_male : Nat) (n_female : Nat) 
  (h1 : n_male = 6) (h2 : n_female = 5) : 
  (n_male.choose 2) * (n_female.choose 1) = 75 := by
  sorry

end NUMINAMATH_CALUDE_medical_team_combinations_l846_84600


namespace NUMINAMATH_CALUDE_tangent_sum_half_pi_l846_84656

theorem tangent_sum_half_pi (α β γ : Real) 
  (h_acute : 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 ∧ 0 < γ ∧ γ < π/2) 
  (h_sum : α + β + γ = π/2) : 
  Real.tan α * Real.tan β + Real.tan α * Real.tan γ + Real.tan β * Real.tan γ = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_half_pi_l846_84656


namespace NUMINAMATH_CALUDE_f_max_min_implies_a_range_l846_84698

/-- The function f parameterized by a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a+2)*x + 1

/-- The statement that f has both a maximum and a minimum value -/
def has_max_and_min (a : ℝ) : Prop :=
  ∃ (x_max x_min : ℝ), ∀ (x : ℝ), f a x ≤ f a x_max ∧ f a x_min ≤ f a x

/-- The main theorem -/
theorem f_max_min_implies_a_range :
  ∀ a : ℝ, has_max_and_min a → a > 2 ∨ a < -1 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_implies_a_range_l846_84698


namespace NUMINAMATH_CALUDE_a_upper_bound_l846_84637

def f (x : ℝ) := x + x^3

theorem a_upper_bound
  (h : ∀ θ : ℝ, 0 < θ → θ < π/2 → ∀ a : ℝ, f (a * Real.sin θ) + f (1 - a) > 0) :
  ∀ a : ℝ, a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_a_upper_bound_l846_84637


namespace NUMINAMATH_CALUDE_stormi_car_wash_price_l846_84627

/-- The amount Stormi charges for washing each car -/
def car_wash_price : ℝ := 10

/-- The number of cars Stormi washes -/
def num_cars : ℕ := 3

/-- The price Stormi charges for mowing a lawn -/
def lawn_mow_price : ℝ := 13

/-- The number of lawns Stormi mows -/
def num_lawns : ℕ := 2

/-- The cost of the bicycle -/
def bicycle_cost : ℝ := 80

/-- The additional amount Stormi needs to afford the bicycle -/
def additional_amount_needed : ℝ := 24

theorem stormi_car_wash_price :
  car_wash_price * num_cars + lawn_mow_price * num_lawns = bicycle_cost - additional_amount_needed :=
sorry

end NUMINAMATH_CALUDE_stormi_car_wash_price_l846_84627


namespace NUMINAMATH_CALUDE_initial_saline_concentration_l846_84652

theorem initial_saline_concentration 
  (initial_weight : ℝ) 
  (water_added : ℝ) 
  (final_concentration : ℝ) 
  (h1 : initial_weight = 100)
  (h2 : water_added = 200)
  (h3 : final_concentration = 10)
  : ∃ (initial_concentration : ℝ),
    initial_concentration = 30 ∧ 
    (initial_concentration / 100) * initial_weight = 
    (final_concentration / 100) * (initial_weight + water_added) :=
sorry

end NUMINAMATH_CALUDE_initial_saline_concentration_l846_84652


namespace NUMINAMATH_CALUDE_max_cars_quotient_l846_84676

/-- Represents the maximum number of cars that can pass a sensor in one hour -/
def N : ℕ := 4000

/-- The length of a car in meters -/
def car_length : ℝ := 5

/-- The safety rule factor: number of car lengths per 20 km/h of speed -/
def safety_factor : ℝ := 2

/-- Theorem stating that the maximum number of cars passing the sensor in one hour, 
    divided by 15, is equal to 266 -/
theorem max_cars_quotient : N / 15 = 266 := by sorry

end NUMINAMATH_CALUDE_max_cars_quotient_l846_84676


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l846_84658

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 2}

-- Define set B
def B : Set ℕ := {2, 3}

-- Theorem statement
theorem intersection_complement_equality :
  A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l846_84658


namespace NUMINAMATH_CALUDE_smallest_two_digit_integer_with_property_l846_84643

theorem smallest_two_digit_integer_with_property : ∃ n : ℕ, 
  (n ≥ 10 ∧ n < 100) ∧ 
  (let a := n / 10; let b := n % 10; 10 * b + a + 5 = 2 * n) ∧
  (∀ m : ℕ, m ≥ 10 ∧ m < 100 → 
    (let x := m / 10; let y := m % 10; 10 * y + x + 5 = 2 * m) → 
    m ≥ n) ∧
  n = 69 :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_integer_with_property_l846_84643


namespace NUMINAMATH_CALUDE_negative_seven_in_A_l846_84677

def A : Set ℤ := {1, -7}

theorem negative_seven_in_A : -7 ∈ A := by
  sorry

end NUMINAMATH_CALUDE_negative_seven_in_A_l846_84677


namespace NUMINAMATH_CALUDE_average_rope_length_l846_84610

theorem average_rope_length (piece1 piece2 : ℝ) (h1 : piece1 = 2) (h2 : piece2 = 6) :
  (piece1 + piece2) / 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_rope_length_l846_84610


namespace NUMINAMATH_CALUDE_laptop_original_price_l846_84635

/-- Proves that if a laptop's price is reduced by 15% and the new price is $680, then the original price was $800. -/
theorem laptop_original_price (discount_percent : ℝ) (discounted_price : ℝ) (original_price : ℝ) : 
  discount_percent = 15 →
  discounted_price = 680 →
  discounted_price = original_price * (1 - discount_percent / 100) →
  original_price = 800 := by
sorry

end NUMINAMATH_CALUDE_laptop_original_price_l846_84635


namespace NUMINAMATH_CALUDE_two_face_cubes_count_l846_84680

/-- Represents a 3x3x3 cube formed by cutting a larger cube painted on all faces -/
structure PaintedCube :=
  (size : Nat)
  (painted_faces : Nat)
  (h_size : size = 3)
  (h_painted : painted_faces = 6)

/-- Counts the number of smaller cubes painted on exactly two faces -/
def count_two_face_cubes (cube : PaintedCube) : Nat :=
  12

/-- Theorem: The number of smaller cubes painted on exactly two faces in a 3x3x3 PaintedCube is 12 -/
theorem two_face_cubes_count (cube : PaintedCube) : count_two_face_cubes cube = 12 := by
  sorry

end NUMINAMATH_CALUDE_two_face_cubes_count_l846_84680


namespace NUMINAMATH_CALUDE_inequalities_from_sqrt_l846_84629

theorem inequalities_from_sqrt (a b : ℝ) (h : Real.sqrt a > Real.sqrt b) :
  (a^2 > b^2) ∧ ((b + 1) / (a + 1) > b / a) ∧ (b + 1 / (b + 1) ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_from_sqrt_l846_84629


namespace NUMINAMATH_CALUDE_max_sum_of_digits_l846_84672

/-- Represents a nonzero digit (1-9) -/
def NonzeroDigit := {d : ℕ // 1 ≤ d ∧ d ≤ 9}

/-- An is an n-digit integer with all digits equal to a -/
def An (a : NonzeroDigit) (n : ℕ+) : ℕ := a.val * (10^n.val - 1) / 9

/-- Bn is an n-digit integer with all digits equal to b -/
def Bn (b : NonzeroDigit) (n : ℕ+) : ℕ := b.val * (10^n.val - 1) / 9

/-- Cn is a 3n-digit integer with all digits equal to c -/
def Cn (c : NonzeroDigit) (n : ℕ+) : ℕ := c.val * (10^(3*n.val) - 1) / 9

/-- The equation Cn - Bn = An^3 is satisfied for at least two values of n -/
def SatisfiesEquation (a b c : NonzeroDigit) : Prop :=
  ∃ n₁ n₂ : ℕ+, n₁ ≠ n₂ ∧ Cn c n₁ - Bn b n₁ = (An a n₁)^3 ∧ Cn c n₂ - Bn b n₂ = (An a n₂)^3

theorem max_sum_of_digits (a b c : NonzeroDigit) (h : SatisfiesEquation a b c) :
  a.val + b.val + c.val ≤ 19 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_l846_84672


namespace NUMINAMATH_CALUDE_unique_prime_perfect_square_l846_84613

theorem unique_prime_perfect_square :
  ∀ p : ℕ, Prime p → (∃ q : ℕ, 5^p + 4*p^4 = q^2) → p = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_perfect_square_l846_84613


namespace NUMINAMATH_CALUDE_force_for_10_inch_screwdriver_l846_84687

/-- Represents the force-length relationship for screwdrivers -/
structure ScrewdriverForce where
  force : ℝ
  length : ℝ
  constant : ℝ

/-- The force-length relationship is inverse and constant -/
axiom force_length_relation (sf : ScrewdriverForce) : sf.force * sf.length = sf.constant

/-- Given conditions for the 6-inch screwdriver -/
def initial_screwdriver : ScrewdriverForce :=
  { force := 60
    length := 6
    constant := 60 * 6 }

/-- Theorem stating the force required for a 10-inch screwdriver -/
theorem force_for_10_inch_screwdriver :
  ∃ (sf : ScrewdriverForce), sf.length = 10 ∧ sf.constant = initial_screwdriver.constant ∧ sf.force = 36 :=
by sorry

end NUMINAMATH_CALUDE_force_for_10_inch_screwdriver_l846_84687


namespace NUMINAMATH_CALUDE_correct_minus_position_l846_84607

def numbers : List ℕ := [6, 9, 12, 15, 18, 21]

def place_signs (nums : List ℕ) (minus_pos : ℕ) : ℤ :=
  (nums.take minus_pos).sum - nums[minus_pos]! + (nums.drop (minus_pos + 1)).sum

theorem correct_minus_position (nums : List ℕ) (h : nums = numbers) :
  ∃! pos : ℕ, pos < nums.length - 1 ∧ place_signs nums pos = 45 :=
by sorry

end NUMINAMATH_CALUDE_correct_minus_position_l846_84607


namespace NUMINAMATH_CALUDE_pentagon_perimeter_is_nine_l846_84611

/-- Pentagon with given side lengths -/
structure Pentagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  EA : ℝ

/-- The perimeter of a pentagon -/
def perimeter (p : Pentagon) : ℝ := p.AB + p.BC + p.CD + p.DE + p.EA

/-- Theorem: The perimeter of the given pentagon is 9 -/
theorem pentagon_perimeter_is_nine :
  ∃ (p : Pentagon), p.AB = 2 ∧ p.BC = 2 ∧ p.CD = 1 ∧ p.DE = 1 ∧ p.EA = 3 ∧ perimeter p = 9 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_perimeter_is_nine_l846_84611


namespace NUMINAMATH_CALUDE_card_draw_probability_l846_84667

/-- The number of cards in the set -/
def n : ℕ := 100

/-- The number of draws -/
def k : ℕ := 20

/-- The probability that all drawn numbers are distinct -/
noncomputable def p : ℝ := (n.factorial / (n - k).factorial) / n^k

/-- Main theorem -/
theorem card_draw_probability : p < (9/10)^19 ∧ (9/10)^19 < 1/Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_card_draw_probability_l846_84667


namespace NUMINAMATH_CALUDE_power_of_two_l846_84624

theorem power_of_two (n : ℕ) : 32 * (1/2)^2 = 2^n → 2^n = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_l846_84624


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l846_84668

/-- Given an equilateral triangle with perimeter 60 and an isosceles triangle with perimeter 70,
    where one side of the equilateral triangle is also a side of the isosceles triangle,
    prove that the base of the isosceles triangle is 30 units long. -/
theorem isosceles_triangle_base_length
  (equilateral_perimeter : ℝ)
  (isosceles_perimeter : ℝ)
  (h_equilateral_perimeter : equilateral_perimeter = 60)
  (h_isosceles_perimeter : isosceles_perimeter = 70)
  (h_shared_side : ∃ (side : ℝ), side = equilateral_perimeter / 3 ∧
                   isosceles_perimeter = 2 * side + (isosceles_perimeter - 2 * side)) :
  isosceles_perimeter - 2 * (equilateral_perimeter / 3) = 30 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l846_84668


namespace NUMINAMATH_CALUDE_means_inequality_l846_84603

theorem means_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  (a + b + c) / 3 > (a * b * c) ^ (1/3) ∧ 
  (a * b * c) ^ (1/3) > 3 / ((1/a) + (1/b) + (1/c)) := by
  sorry

#check means_inequality

end NUMINAMATH_CALUDE_means_inequality_l846_84603


namespace NUMINAMATH_CALUDE_no_prime_sided_integer_area_triangle_l846_84664

theorem no_prime_sided_integer_area_triangle : 
  ¬ ∃ (a b c : ℕ) (S : ℝ), 
    (Prime a ∧ Prime b ∧ Prime c) ∧ 
    (S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) ∧ 
    (S ≠ 0) ∧ 
    (∃ (n : ℕ), S = n) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sided_integer_area_triangle_l846_84664


namespace NUMINAMATH_CALUDE_circle_slope_bounds_l846_84651

theorem circle_slope_bounds (x y : ℝ) (h : x^2 + y^2 + 2*x - 4*y + 1 = 0) :
  ∃ (k : ℝ), y = k*(x-4) ∧ -20/21 ≤ k ∧ k ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_circle_slope_bounds_l846_84651


namespace NUMINAMATH_CALUDE_double_reflection_of_D_l846_84662

/-- Reflects a point across the x-axis -/
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflects a point across the line y = -x + 1 -/
def reflect_diagonal (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2 + 1, -p.1 + 1)

/-- The main theorem -/
theorem double_reflection_of_D (D : ℝ × ℝ) (hD : D = (5, 2)) :
  (reflect_diagonal ∘ reflect_x_axis) D = (-3, -4) := by
  sorry

end NUMINAMATH_CALUDE_double_reflection_of_D_l846_84662


namespace NUMINAMATH_CALUDE_exists_special_number_l846_84640

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number n such that the sum of its digits is 1000
    and the sum of digits of its square is 1000000 -/
theorem exists_special_number : 
  ∃ n : ℕ, sum_of_digits n = 1000 ∧ sum_of_digits (n^2) = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_exists_special_number_l846_84640


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l846_84683

/-- The radius of the largest circle inscribed in a square, given specific distances from a point on the circle to two adjacent sides of the square. -/
theorem inscribed_circle_radius (square_side : ℝ) (dist_to_side1 : ℝ) (dist_to_side2 : ℝ) :
  square_side > 20 →
  dist_to_side1 = 8 →
  dist_to_side2 = 9 →
  ∃ (radius : ℝ),
    radius > 10 ∧
    (radius - dist_to_side1)^2 + (radius - dist_to_side2)^2 = radius^2 ∧
    radius = 29 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l846_84683


namespace NUMINAMATH_CALUDE_cube_root_inequality_l846_84620

theorem cube_root_inequality (x : ℝ) : 
  x > 0 → (x^(1/3) < 3*x ↔ x > 1/(3*(3^(1/2)))) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_inequality_l846_84620


namespace NUMINAMATH_CALUDE_smallest_number_l846_84693

theorem smallest_number (jungkook yoongi yuna : ℕ) : 
  jungkook = 6 - 3 → yoongi = 4 → yuna = 5 → min jungkook (min yoongi yuna) = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_l846_84693


namespace NUMINAMATH_CALUDE_largest_term_index_l846_84622

def A (k : ℕ) : ℝ := (Nat.choose 2000 k) * (0.1 ^ k)

theorem largest_term_index : 
  ∃ (k : ℕ), k ≤ 2000 ∧ 
  (∀ (j : ℕ), j ≤ 2000 → A k ≥ A j) ∧
  k = 181 := by
  sorry

end NUMINAMATH_CALUDE_largest_term_index_l846_84622
