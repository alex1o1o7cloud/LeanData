import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_arithmetic_sequence_l686_68661

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def SumArithmeticSequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem angle_of_inclination_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_a4 : a 4 = 15) 
  (h_s5 : SumArithmeticSequence a 5 = 55) :
  let m := (a 2011 - a 2010) / (3 - 4)
  Real.pi - Real.arctan m = Real.pi - Real.arctan 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_arithmetic_sequence_l686_68661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequalities_l686_68606

open Complex

noncomputable def hyperbola_left_branch (a b : ℂ) (c : ℝ) : Set ℂ :=
  {z : ℂ | ∃ (x y : ℝ), z = ⟨x, y⟩ ∧ ((x - a.re) / c)^2 - ((y - a.im) / (Real.sqrt (c^2 - abs (b - a)^2)))^2 > 1 ∧ x < a.re}

noncomputable def ellipse (f1 f2 : ℂ) (a : ℝ) : Set ℂ :=
  {z : ℂ | abs (z - f1) + abs (z - f2) = a}

theorem complex_inequalities (z : ℂ) :
  (abs (z - (0 : ℂ) + 2*I) ≥ 4 ↔ z ∉ interior (Metric.ball ((0 : ℂ) + 2*I) 4)) ∧
  (abs (z - 1) - abs (z + 3) > 3 ↔ z.re < -1/2 ∧ z ∉ hyperbola_left_branch 1 (-3) 3) ∧
  (abs ((z - 1) / (z + 2)) ≥ 1 ↔ z.re ≤ -1/2) ∧
  (z.im ≤ 3 ↔ z.im ≤ 3) ∧
  (abs (z + 2*I) + abs (z - 2*I) = 9 ↔ z ∈ ellipse ((0 : ℂ) + 2*I) ((0 : ℂ) - 2*I) 9) := by
  sorry

-- Note: We've defined hyperbola_left_branch and ellipse as they were not available in Mathlib

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequalities_l686_68606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_same_side_l686_68632

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the outcome of flipping five coins -/
structure FiveCoinFlip where
  penny : CoinFlip
  nickel : CoinFlip
  dime : CoinFlip
  quarter : CoinFlip
  half_dollar : CoinFlip

/-- Checks if the penny, dime, and half-dollar all show the same side -/
def sameSide (flip : FiveCoinFlip) : Prop :=
  (flip.penny = flip.dime) ∧ (flip.dime = flip.half_dollar)

/-- The set of all possible outcomes when flipping five coins -/
def allOutcomes : Finset FiveCoinFlip := sorry

/-- The set of outcomes where penny, dime, and half-dollar show the same side -/
def favorableOutcomes : Finset FiveCoinFlip := sorry

theorem probability_same_side :
  (favorableOutcomes.card : ℚ) / (allOutcomes.card : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_same_side_l686_68632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_l686_68639

-- Define the line
def line (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the circle
def circleEq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2

-- Define point A (intersection of line with x-axis)
def A : ℝ × ℝ := (-2, 0)

-- Define point B (intersection of line with y-axis)
def B : ℝ × ℝ := (0, -2)

-- Define point P on the circle
noncomputable def P : ℝ → ℝ × ℝ := λ θ => (2 + Real.sqrt 2 * Real.cos θ, Real.sqrt 2 * Real.sin θ)

-- Define the area of triangle ABP
noncomputable def area (θ : ℝ) : ℝ :=
  let d := (abs (2 * Real.sin (θ + Real.pi/4) + 4)) / Real.sqrt 2
  (Real.sqrt 8 * d) / 2

-- Theorem statement
theorem triangle_area_range :
  ∀ θ : ℝ, 2 ≤ area θ ∧ area θ ≤ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_l686_68639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l686_68646

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then a * x + b else 10 - 4 * x

theorem function_composition_identity (a b : ℝ) :
  (∀ x, f a b (f a b x) = x) → a + b = 21/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l686_68646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strips_cover_circle_l686_68685

-- Define a strip as a structure with a width and a position
structure Strip where
  width : ℝ
  position : ℝ × ℝ

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define membership for a point in a circle
def inCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 ≤ c.radius^2

-- Define the theorem
theorem strips_cover_circle 
  (strips : List Strip) 
  (circle : Circle) : 
  (strips.map Strip.width).sum = 100 →
  circle.radius = 1 →
  ∃ (new_positions : List (ℝ × ℝ)), 
    (new_positions.length = strips.length) ∧ 
    (∀ (p : ℝ × ℝ), inCircle p circle → 
      ∃ (i : Fin strips.length), 
        ∃ (s : Strip), s.width = (strips.get ⟨i, by sorry⟩).width ∧ 
                       s.position = new_positions.get ⟨i, by sorry⟩ ∧
                       (p.1 - s.position.1)^2 + (p.2 - s.position.2)^2 ≤ s.width^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_strips_cover_circle_l686_68685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_sets_f_range_condition_l686_68649

/-- The function f(x) = (x + a) / (x + b) -/
noncomputable def f (a b x : ℝ) : ℝ := (x + a) / (x + b)

theorem f_inequality_solution_sets (a : ℝ) :
  (∀ x, f a 1 (x - 1) ≤ 0 ↔ 
    (a < 1 ∧ x ∈ Set.Ioc 0 (1 - a)) ∨
    (a = 1 ∧ x ∈ (∅ : Set ℝ)) ∨
    (a > 1 ∧ x ∈ Set.Ico (1 - a) 0)) :=
by sorry

theorem f_range_condition (b : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f 1 b x > -1 / (x + b)^2) ↔ b > -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_sets_f_range_condition_l686_68649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_two_same_l686_68637

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The probability of at least two dice showing the same number when rolling 6 fair 8-sided dice -/
theorem prob_at_least_two_same : 
  1 - (num_sides.factorial / (num_sides - num_dice).factorial : ℚ) / (num_sides ^ num_dice : ℚ) = 3781 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_two_same_l686_68637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l686_68676

theorem vector_properties :
  let a : Fin 3 → ℝ := λ i => if i = 0 then 1 else if i = 1 then 1 else 1
  let b : Fin 3 → ℝ := λ i => if i = 0 then -1 else if i = 1 then 0 else 2
  (∀ i, (a + b) i = (if i = 0 then 0 else if i = 1 then 1 else 3)) ∧ 
  (Real.sqrt (Finset.sum Finset.univ (λ i => (a i)^2)) = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l686_68676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sasha_is_girl_l686_68603

-- Define the set of children
inductive Child : Type
| Vanya | Dima | Egor | Inna | Lesha | Sasha | Tanya

-- Define the gender type
inductive Gender : Type
| Boy | Girl

-- Define the class type
inductive ClassType : Type
| Class1 | Class2

-- Define the answer type
inductive Answer : Type
| Two | Three

-- Function to assign gender to each child
def gender : Child → Gender
| Child.Vanya => Gender.Boy
| Child.Dima => Gender.Boy
| Child.Egor => Gender.Boy
| Child.Inna => Gender.Girl
| Child.Lesha => Gender.Boy
| Child.Tanya => Gender.Girl
| Child.Sasha => sorry -- This is what we need to prove

-- Function to assign class to each child
def classAssignment : Child → ClassType := sorry

-- Function to represent each child's answer
def answer : Child → Answer := sorry

-- Function to count classmates for a given child
def countClassmates (c : Child) : Nat := sorry

-- Function to count male classmates for a given child
def countMaleClassmates (c : Child) : Nat := sorry

-- Theorem stating that Sasha is a girl
theorem sasha_is_girl : gender Child.Sasha = Gender.Girl := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sasha_is_girl_l686_68603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_decreasing_on_interval_l686_68627

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sqrt 3 * cos (2 * x - π / 3)

-- State the theorem
theorem f_not_decreasing_on_interval :
  ¬ (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ π / 2 → f x ≥ f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_decreasing_on_interval_l686_68627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_overtime_rate_multiple_l686_68684

/-- Calculates the overtime rate multiple for Janet's work scenario -/
theorem janet_overtime_rate_multiple : 
  ∀ (normal_rate : ℝ) (total_hours : ℝ) (regular_hours : ℝ) (car_price : ℝ) (weeks_to_save : ℝ),
    normal_rate = 20 →
    total_hours = 52 →
    regular_hours = 40 →
    car_price = 4640 →
    weeks_to_save = 4 →
    let overtime_hours := total_hours - regular_hours
    let weekly_regular_pay := normal_rate * regular_hours
    let total_weekly_pay := car_price / weeks_to_save
    let weekly_overtime_pay := total_weekly_pay - weekly_regular_pay
    let overtime_rate := weekly_overtime_pay / overtime_hours
    overtime_rate / normal_rate = 1.5 := by
  intros normal_rate total_hours regular_hours car_price weeks_to_save
  intros h1 h2 h3 h4 h5
  -- Define local variables
  let overtime_hours := total_hours - regular_hours
  let weekly_regular_pay := normal_rate * regular_hours
  let total_weekly_pay := car_price / weeks_to_save
  let weekly_overtime_pay := total_weekly_pay - weekly_regular_pay
  let overtime_rate := weekly_overtime_pay / overtime_hours
  
  sorry -- Placeholder for the actual proof

#check janet_overtime_rate_multiple

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_overtime_rate_multiple_l686_68684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_order_theorem_l686_68633

/-- Represents a type of pizza -/
inductive PizzaType
| Cheese
| Pepperoni
| Vegetarian
| Vegan
deriving BEq, Repr

/-- Represents a type of crust -/
inductive CrustType
| Regular
| Thin
deriving BEq, Repr

/-- Represents a pizza order -/
structure PizzaOrder :=
  (pizzaType : PizzaType)
  (crustType : CrustType)
  (quantity : ℚ)

/-- Represents a person's pizza consumption -/
def personConsumption : List PizzaOrder :=
  [
    { pizzaType := PizzaType.Cheese, crustType := CrustType.Regular, quantity := 1/2 },
    { pizzaType := PizzaType.Vegan, crustType := CrustType.Regular, quantity := 1/3 },
    { pizzaType := PizzaType.Vegetarian, crustType := CrustType.Regular, quantity := 1/6 },
    { pizzaType := PizzaType.Pepperoni, crustType := CrustType.Regular, quantity := 1/4 },
    { pizzaType := PizzaType.Vegetarian, crustType := CrustType.Regular, quantity := 1/4 },
    { pizzaType := PizzaType.Cheese, crustType := CrustType.Thin, quantity := 1/2 },
    { pizzaType := PizzaType.Pepperoni, crustType := CrustType.Thin, quantity := 1/6 }
  ]

/-- The number of people in the group -/
def numberOfPeople : ℕ := 5

theorem pizza_order_theorem :
  let totalPizzas := personConsumption.map (λ o => Int.ceil o.quantity)
  let distinctPizzas := (personConsumption.map (λ o => (o.pizzaType, o.crustType))).eraseDups
  (totalPizzas.sum = distinctPizzas.length) ∧
  (distinctPizzas.length = 5) ∧
  (numberOfPeople = 5) := by
  sorry

#check pizza_order_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_order_theorem_l686_68633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_decreasing_on_interval_l686_68654

theorem sin_decreasing_on_interval :
  ∀ x y : ℝ, π/2 ≤ x ∧ x < y ∧ y ≤ 3*π/2 → Real.sin x > Real.sin y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_decreasing_on_interval_l686_68654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_print_time_rounded_l686_68667

noncomputable def roundToNearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

noncomputable def printTime (pages : ℝ) (rate : ℝ) : ℝ :=
  pages / rate

theorem print_time_rounded (pages rate : ℝ) :
  pages = 350 → rate = 24 → roundToNearest (printTime pages rate) = 15 := by
  intro h1 h2
  sorry

#check print_time_rounded

end NUMINAMATH_CALUDE_ERRORFEEDBACK_print_time_rounded_l686_68667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_1_min_value_f_on_open_right_1_x_at_min_value_f_l686_68690

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x + 5) / (x - 1)

-- Theorem for the solution set of f(x) ≥ 1
theorem solution_set_f_geq_1 :
  {x : ℝ | f x ≥ 1} = Set.Ioc 1 2 ∪ Set.Ioi 3 := by sorry

-- Theorem for the minimum value of f on (1, +∞)
theorem min_value_f_on_open_right_1 :
  ∃ (x : ℝ), x > 1 ∧ f x = 2*Real.sqrt 2 - 2 ∧ ∀ y > 1, f y ≥ f x := by sorry

-- Theorem for the x value at which the minimum occurs
theorem x_at_min_value_f :
  ∃ (x : ℝ), x = 1 + Real.sqrt 2 ∧ 
    (∀ y > 1, f y ≥ f x) ∧
    (∀ y > 1, f y = f x → y = x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_1_min_value_f_on_open_right_1_x_at_min_value_f_l686_68690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_l686_68611

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 3)

-- State the theorem
theorem power_function_decreasing (m : ℝ) :
  (∀ x > 0, ∃ y ≠ 0, f m x = y * x^(m^2 + m - 3)) →  -- f is a power function
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f m x₁ > f m x₂) →   -- f is decreasing for x ∈ (0, +∞)
  m = -1 :=
by
  sorry

#check power_function_decreasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_l686_68611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_max_distance_positions_correct_l686_68629

/-- The coordinates of the beetle's positions where the distance between the insects is greatest -/
noncomputable def beetle_max_distance_positions : Set (ℝ × ℝ) :=
  { ((5 / Real.sqrt 2) * (1 - Real.sqrt 7), (5 / Real.sqrt 2) * (1 + Real.sqrt 7)),
    (-(5 / Real.sqrt 2) * (1 + Real.sqrt 7), (5 / Real.sqrt 2) * (1 - Real.sqrt 7)),
    ((5 / Real.sqrt 2) * (Real.sqrt 7 - 1), -(5 / Real.sqrt 2) * (1 + Real.sqrt 7)),
    ((5 / Real.sqrt 2) * (1 + Real.sqrt 7), (5 / Real.sqrt 2) * (Real.sqrt 7 - 1)) }

/-- The initial position of the water strider -/
noncomputable def water_strider_initial : ℝ × ℝ := (2, 2 * Real.sqrt 7)

/-- The initial position of the beetle -/
noncomputable def beetle_initial : ℝ × ℝ := (5, 5 * Real.sqrt 7)

/-- The speed ratio of the water strider to the beetle -/
def speed_ratio : ℝ := 2

theorem beetle_max_distance_positions_correct :
  ∀ p ∈ beetle_max_distance_positions,
    ∃ t : ℝ,
      let water_strider_pos := (4 * Real.sqrt 2 * Real.cos (speed_ratio * t), 4 * Real.sqrt 2 * Real.sin (speed_ratio * t))
      let beetle_pos := (10 * Real.sqrt 2 * Real.cos t, 10 * Real.sqrt 2 * Real.sin t)
      p = beetle_pos ∧
      ∀ s : ℝ,
        let ws_pos_s := (4 * Real.sqrt 2 * Real.cos (speed_ratio * s), 4 * Real.sqrt 2 * Real.sin (speed_ratio * s))
        let beetle_pos_s := (10 * Real.sqrt 2 * Real.cos s, 10 * Real.sqrt 2 * Real.sin s)
        (ws_pos_s.1 - beetle_pos_s.1)^2 + (ws_pos_s.2 - beetle_pos_s.2)^2 ≤
        (water_strider_pos.1 - beetle_pos.1)^2 + (water_strider_pos.2 - beetle_pos.2)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_max_distance_positions_correct_l686_68629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l686_68624

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- State the theorem
theorem inequality_proof : f 2019 < f 2018 ∧ f 2018 < f 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l686_68624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_unique_flavors_l686_68665

/-- Represents the number of red candies -/
def red_candies : ℕ := 5

/-- Represents the number of green candies -/
def green_candies : ℕ := 4

/-- Represents a ratio of red to green candies -/
structure CandyRatio where
  red : ℕ
  green : ℕ

/-- Simplifies a CandyRatio -/
def simplify_ratio (r : CandyRatio) : CandyRatio :=
  sorry

/-- Checks if two CandyRatios are equivalent (i.e., simplify to the same ratio) -/
def ratio_equiv (r1 r2 : CandyRatio) : Prop :=
  simplify_ratio r1 = simplify_ratio r2

/-- Set of all possible CandyRatios -/
def all_ratios : Finset CandyRatio :=
  sorry

/-- Set of all unique simplified CandyRatios, excluding the case of no candies -/
def unique_ratios : Finset CandyRatio :=
  sorry

/-- The main theorem: there are 15 unique flavors -/
theorem fifteen_unique_flavors : unique_ratios.card = 15 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_unique_flavors_l686_68665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_inch_cube_value_l686_68650

/-- The value of a cube of silver -/
noncomputable def silver_value (side_length : ℝ) : ℝ :=
  let two_inch_cube_value := 200
  let two_inch_cube_volume := 2^3
  let value_per_cubic_inch := two_inch_cube_value / two_inch_cube_volume
  value_per_cubic_inch * side_length^3

/-- Theorem: A 3-inch cube of silver is worth $675 -/
theorem three_inch_cube_value :
  silver_value 3 = 675 := by
  -- Unfold the definition of silver_value
  unfold silver_value
  -- Simplify the expression
  simp
  -- Perform the numerical calculation
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_inch_cube_value_l686_68650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l686_68687

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x - 3) / Real.sqrt (x^2 - 5*x + 6)

-- Define the domain of f
def domain_f : Set ℝ := {x | x < 2 ∨ x > 3}

-- Theorem stating that the domain of f is (-∞, 2) ∪ (3, ∞)
theorem domain_of_f :
  ∀ x : ℝ, (x ∈ domain_f) ↔ (x^2 - 5*x + 6 > 0 ∧ x ≠ 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l686_68687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_g_minus_x_squared_over_2_f_geq_g_iff_k_nonpositive_l686_68662

-- Define the functions f and g
noncomputable def f (x : ℝ) := Real.exp x
noncomputable def g (k : ℝ) (x : ℝ) := (k/2) * x^2 + x + 1

-- Statement 1
theorem f_geq_g_minus_x_squared_over_2 :
  ∀ x : ℝ, f x ≥ g 1 x - x^2/2 := by sorry

-- Statement 2
theorem f_geq_g_iff_k_nonpositive :
  ∀ k : ℝ, (∀ x : ℝ, f x ≥ g k x) ↔ k ≤ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_g_minus_x_squared_over_2_f_geq_g_iff_k_nonpositive_l686_68662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equality_l686_68677

theorem cube_root_equality (n : ℕ) : (n * 27 : ℝ)^(1/3) = 54 ↔ n = 5832 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equality_l686_68677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l686_68644

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_properties (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_point : quadratic a b c 1 = 1)
  (h_min : ∀ x, quadratic a b c x ≥ 0) :
  (∃ (max_ac : ℝ), max_ac = 1/16 ∧ a * c ≤ max_ac) ∧
  (∀ lambda : ℝ, 1 - b = lambda * Real.sqrt a → lambda ≥ 2 * Real.sqrt 2 - 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l686_68644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_property_implies_logarithmic_l686_68619

/-- A function f: ℝ₊ → ℝ is logarithmic if it satisfies f(xy) = f(x) + f(y) for all x, y > 0 -/
def IsLogarithmic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f (x * y) = f x + f y

/-- Theorem: If a function satisfies the logarithmic property, then it is a logarithmic function -/
theorem logarithmic_property_implies_logarithmic (f : ℝ → ℝ) :
  IsLogarithmic f → ∃ a > 0, ∀ x > 0, f x = Real.log x / Real.log a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_property_implies_logarithmic_l686_68619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_theorem_l686_68696

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) : ℝ := 2 * n - 1

-- Define the geometric sequence b_n
noncomputable def b (n : ℕ) : ℝ := 3^(n - 1)

-- Define the sum S_n
noncomputable def S (n : ℕ) : ℝ := n * (n - 1) / 2

-- State the theorem
theorem arithmetic_geometric_sequence_theorem :
  ∀ d : ℝ,
  d ≠ 0 →
  (a 1 = 1) →
  (b 1 = 1) →
  (a 2 = b 2) →
  (2 * a 3 - b 3 = 1) →
  (∀ n : ℕ, n ≥ 1 → a n = 2 * n - 1) ∧
  (∀ n : ℕ, n ≥ 1 → b n = 3^(n - 1)) ∧
  (∀ n : ℕ, n ≥ 1 → S n = (n * (n - 1)) / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_theorem_l686_68696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_D_l686_68668

-- Define the square
def Square (A B C D : ℝ × ℝ) : Prop :=
  A = (0, 0) ∧ B = (2, 0) ∧ C = (2, 2) ∧ D = (0, 2)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem max_distance_to_D (A B C D P : ℝ × ℝ) (u v w : ℝ) :
  Square A B C D →
  distance P A = u →
  distance P B = v →
  distance P C = w →
  u^2 + v^2 = 2 * w^2 →
  ∃ (P_max : ℝ × ℝ), ∀ (P' : ℝ × ℝ),
    (distance P' A)^2 + (distance P' B)^2 = 2 * (distance P' C)^2 →
    distance P' D ≤ distance P_max D ∧
    distance P_max D = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_D_l686_68668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_two_roots_l686_68618

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if -1 < x ∧ x ≤ 0 then 1 / (x + 1) - 3
  else if 0 < x ∧ x ≤ 1 then x^2 - 3*x + 2
  else 0  -- Default value for x outside the specified ranges

-- State the theorem
theorem range_of_m_for_two_roots :
  ∀ m : ℝ, (∃ x y : ℝ, x ≠ y ∧ g x - m * x - m = 0 ∧ g y - m * y - m = 0) →
  (m ∈ Set.Ioc (-9/4) (-2) ∪ Set.Ico 0 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_two_roots_l686_68618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_field_area_l686_68699

/-- The cost of fencing per meter in Rupees -/
noncomputable def fencing_cost_per_meter : ℝ := 4.80

/-- The total cost of fencing in Rupees -/
noncomputable def total_fencing_cost : ℝ := 6334.72526658735

/-- The circumference of the circular field in meters -/
noncomputable def circumference : ℝ := total_fencing_cost / fencing_cost_per_meter

/-- The radius of the circular field in meters -/
noncomputable def radius : ℝ := circumference / (2 * Real.pi)

/-- The area of the circular field in square meters -/
noncomputable def area_sq_meters : ℝ := Real.pi * radius ^ 2

/-- The area of the circular field in hectares -/
noncomputable def area_hectares : ℝ := area_sq_meters / 10000

/-- Theorem stating that the area of the circular field is approximately 13.8545 hectares -/
theorem circular_field_area : ‖area_hectares - 13.8545‖ < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_field_area_l686_68699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_circle_l686_68605

/-- The area of a triangle with side lengths in the ratio 5:12:13, inscribed in a circle of radius 5 -/
noncomputable def triangleArea : ℝ := 6000 / 169

/-- The radius of the circle in which the triangle is inscribed -/
def circleRadius : ℝ := 5

/-- The ratio of the triangle's side lengths -/
def sideRatio : Fin 3 → ℝ
  | 0 => 5
  | 1 => 12
  | 2 => 13

theorem triangle_area_in_circle :
  ∃ (s : ℝ), s > 0 ∧
  (∀ i : Fin 3, s * sideRatio i ≤ 2 * circleRadius) ∧
  (∃ i : Fin 3, s * sideRatio i = 2 * circleRadius) ∧
  triangleArea = (s^2 * sideRatio 0 * sideRatio 1) / 4 := by
  sorry

#check triangle_area_in_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_circle_l686_68605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_root_of_complex_quadratic_l686_68664

theorem real_root_of_complex_quadratic (b : ℝ) : 
  (∃ (x : ℂ), x^2 + (4 + I) * x + (4 + 2 * I) = 0) →
  (b^2 + (4 + I) * b + (4 + 2 * I) = 0) →
  b = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_root_of_complex_quadratic_l686_68664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_relation_l686_68631

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

def arithmetic_mean_property (a : ℕ → ℝ) : Prop :=
  (Finset.sum (Finset.range 20) (fun i => a (i + 41))) / 20 = 
  (Finset.sum (Finset.range 100) (fun i => a (i + 1))) / 100

noncomputable def geometric_mean_property (b : ℕ → ℝ) : Prop :=
  (Finset.prod (Finset.range 20) (fun i => b (i + 41))) ^ (1/20) = 
  (Finset.prod (Finset.range 100) (fun i => b (i + 1))) ^ (1/100)

theorem arithmetic_geometric_mean_relation (a : ℕ → ℝ) (b : ℕ → ℝ) :
  is_arithmetic_sequence a →
  is_geometric_sequence b →
  (∀ n : ℕ, b n > 0) →
  arithmetic_mean_property a →
  geometric_mean_property b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_relation_l686_68631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_is_six_l686_68610

def next_term (n : ℕ) : ℕ :=
  if n < 10 then n * 7
  else if n % 2 = 0 then n / 3
  else n - 3

def harry_sequence (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => next_term (harry_sequence start n)

theorem hundredth_term_is_six :
  harry_sequence 120 99 = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_is_six_l686_68610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_initial_amount_l686_68642

/-- Represents the initial amount of money Jean has in euros -/
noncomputable def jean_initial : ℝ := sorry

/-- Represents the initial amount of money Jane has in US dollars -/
noncomputable def jane_initial : ℝ := sorry

/-- The initial exchange rate from euros to US dollars -/
def initial_exchange_rate : ℝ := 1.2

/-- The final exchange rate from euros to US dollars -/
def final_exchange_rate : ℝ := 1.1

/-- Jack's constant amount in US dollars -/
def jack_amount : ℝ := 120

/-- The total balance in US dollars at the end of six months -/
def total_balance : ℝ := 3000

/-- Jean's monthly contribution rate -/
def jean_contribution_rate : ℝ := 0.2

/-- Jane's monthly contribution rate -/
def jane_contribution_rate : ℝ := 0.25

/-- Number of months -/
def months : ℕ := 6

/-- Jean initially has three times as much money as Jane when converted to the same currency -/
axiom initial_ratio : jean_initial * initial_exchange_rate = 3 * jane_initial

/-- The total balance equation after six months -/
axiom balance_equation : 
  jean_initial * (1 - jean_contribution_rate)^months * final_exchange_rate + 
  jane_initial * (1 - jane_contribution_rate)^months + 
  jack_amount = total_balance

/-- Theorem stating that Jean's initial amount is approximately 9845.58 euros -/
theorem jean_initial_amount : ∃ ε > 0, |jean_initial - 9845.58| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_initial_amount_l686_68642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l686_68622

/-- The ellipse C₁ defined by x²/3 + y² = 1 -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 = 1}

/-- The line C₂ defined by x + y = 4 -/
def C₂ : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | q.1 + q.2 = 4}

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The minimum distance between C₁ and C₂ is √2 -/
theorem min_distance_C₁_C₂ :
  ∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ C₁ ∧ q ∈ C₂ ∧
    (∀ (p' : ℝ × ℝ) (q' : ℝ × ℝ), p' ∈ C₁ → q' ∈ C₂ →
      distance p q ≤ distance p' q') ∧
    distance p q = Real.sqrt 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l686_68622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_observation_value_l686_68641

theorem incorrect_observation_value 
  (n : ℕ) 
  (initial_mean : ℝ) 
  (correct_value : ℝ) 
  (new_mean : ℝ) 
  (h1 : n = 50) 
  (h2 : initial_mean = 36) 
  (h3 : correct_value = 44) 
  (h4 : new_mean = 36.5) : 
  ∃ (incorrect_value : ℝ), 
    n * new_mean = (n * initial_mean - incorrect_value + correct_value) ∧ 
    incorrect_value = 19 := by
  sorry

#check incorrect_observation_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_observation_value_l686_68641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_points_with_properties_l686_68675

/-- A type representing a point in a plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear --/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Calculate the distance between two points --/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- Main theorem statement --/
theorem exists_n_points_with_properties (N : ℕ) : 
  ∃ (points : Finset Point), 
    points.card = N ∧ 
    (∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points → p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬collinear p1 p2 p3) ∧
    (∀ p1 p2, p1 ∈ points → p2 ∈ points → p1 ≠ p2 → ∃ k : ℕ, distance p1 p2 = k) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_points_with_properties_l686_68675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l686_68656

noncomputable def f (m k : ℤ) (x : ℝ) : ℝ := (3 * m^2 - 2 * m + 1) * x^(3 * k - k^2 + 4)

theorem f_properties (m k : ℤ) :
  (∀ x, f m k x = f m k (-x)) →  -- f is an even function
  (∀ x y, 0 < x ∧ x < y → f m k x < f m k y) →  -- f is monotonically increasing on (0, +∞)
  ((f m k = fun x => x^4) ∨ (f m k = fun x => x^6)) ∧
  {x : ℝ | f m k (3 * x + 2) > f m k (1 - 2 * x)} = {x : ℝ | x < -3 ∨ x > -1/5} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l686_68656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soldiers_joined_eq_528_l686_68681

/- Define the initial conditions -/
def initial_soldiers : ℕ := 1200
def initial_consumption : ℚ := 3
def initial_duration : ℕ := 30

/- Define the new conditions -/
def new_consumption : ℚ := 5/2
def new_duration : ℕ := 25

/- Define the total provisions -/
def total_provisions : ℚ := initial_soldiers * initial_consumption * initial_duration

/- Theorem to prove -/
theorem soldiers_joined_eq_528 : 
  ∃ (x : ℕ), x = 528 ∧ 
  total_provisions = (initial_soldiers + x) * new_consumption * new_duration := by
  -- The proof goes here
  sorry

/- Function to compute the number of soldiers joined -/
def soldiers_joined : ℕ := 
  528  -- We're using the known answer here

#eval soldiers_joined

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soldiers_joined_eq_528_l686_68681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_all_reals_l686_68697

/-- The function f(x) with parameter p -/
noncomputable def f (p : ℝ) (x : ℝ) : ℝ := (p * x^2 + 3 * x - 4) / (-3 * x^2 + 3 * x + p)

/-- The domain of f is all real numbers iff p < -3/4 -/
theorem domain_all_reals (p : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, f p x = y) ↔ p < -3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_all_reals_l686_68697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_partition_perfect_square_triples_l686_68680

/-- A partition of ℤ into triples -/
def IntegerPartition := ℕ → (ℤ × ℤ × ℤ)

/-- Predicate to check if a number is a perfect square -/
def IsPerfectSquare (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

/-- The main theorem stating the existence of a partition satisfying the condition -/
theorem exists_partition_perfect_square_triples :
  ∃ (p : IntegerPartition), ∀ (i : ℕ),
    let (a, b, c) := p i
    IsPerfectSquare (|a^3 * b + b^3 * c + c^3 * a|) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_partition_perfect_square_triples_l686_68680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l686_68640

-- Define the line l
noncomputable def line_l (x : ℝ) : ℝ := (Real.sqrt 3 / 3) * x + 1

-- Define the parabola
noncomputable def parabola (x : ℝ) : ℝ := x^2 / 4

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ x, p.1 = x ∧ p.2 = line_l x ∧ p.2 = parabola x}

-- State the theorem
theorem length_AB : 
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16/3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l686_68640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_sine_sum_l686_68615

theorem zero_points_sine_sum (x₁ x₂ m : Real) : 
  x₁ ∈ Set.Icc 0 (π/2) →
  x₂ ∈ Set.Icc 0 (π/2) →
  x₁ ≠ x₂ →
  2 * Real.sin (2 * x₁) + Real.cos (2 * x₁) = m →
  2 * Real.sin (2 * x₂) + Real.cos (2 * x₂) = m →
  Real.sin (x₁ + x₂) = 2 * Real.sqrt 5 / 5 := by
  intros h1 h2 h3 h4 h5
  sorry

#check zero_points_sine_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_sine_sum_l686_68615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_is_three_l686_68674

/-- The radius of a sphere given specific conditions -/
noncomputable def sphere_radius (h d : ℝ) : ℝ :=
  let cylinder_radius := d / 2
  let sphere_area := fun r => 4 * Real.pi * r^2
  let cylinder_curved_area := 2 * Real.pi * cylinder_radius * h
  Real.sqrt (cylinder_curved_area / (4 * Real.pi))

/-- Theorem: The radius of the sphere is 3 cm under given conditions -/
theorem sphere_radius_is_three (h d : ℝ) (h_pos : h > 0) (d_pos : d > 0) 
    (h_eq : h = 6) (d_eq : d = 6) : sphere_radius h d = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_is_three_l686_68674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_tire_production_l686_68628

/-- Represents the number of tires John can produce per day -/
def tires_produced : ℕ := sorry

/-- The cost to produce each tire in dollars -/
def production_cost : ℕ := 250

/-- The selling price multiplier -/
def selling_price_multiplier : ℚ := 3/2

/-- The maximum number of tires that could be sold per day -/
def max_tires_sold : ℕ := 1200

/-- The weekly loss in dollars due to production limitations -/
def weekly_loss : ℕ := 175000

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem stating that John can produce 1134 tires per day -/
theorem johns_tire_production :
  tires_produced = 1134 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_tire_production_l686_68628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_strip_length_correct_l686_68672

/-- Represents a straight cone -/
structure Cone where
  base_diameter : ℝ
  slant_height : ℝ

/-- Represents an adhesive strip -/
structure Strip where
  width : ℝ

/-- Calculates the maximum length of an adhesive strip on a cone's surface -/
noncomputable def max_strip_length (cone : Cone) (strip : Strip) : ℝ :=
  2 * Real.sqrt (cone.slant_height ^ 2 - strip.width ^ 2)

theorem max_strip_length_correct (cone : Cone) (strip : Strip) 
  (h_base : cone.base_diameter = 20)
  (h_slant : cone.slant_height = 20)
  (h_width : strip.width = 2) :
  max_strip_length cone strip = 2 * Real.sqrt 396 := by
  sorry

#eval Float.sqrt 396 * 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_strip_length_correct_l686_68672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_alpha_l686_68648

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.sin (Real.pi/4 + x) * Real.sin (Real.pi/4 - x)

-- State the theorem
theorem f_value_at_alpha (α : ℝ) (h1 : 0 < α ∧ α < Real.pi/2) (h2 : Real.sin (α - Real.pi/4) = 1/2) :
  f α = -Real.sqrt 3 / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_alpha_l686_68648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l686_68621

/-- Non-collinear vectors in a real vector space -/
structure NonCollinearVectors (V : Type*) [AddCommGroup V] [Module ℝ V] where
  a : V
  b : V
  noncollinear : ∃ (x y : ℝ), x • a + y • b ≠ 0

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- The main theorem -/
theorem vector_problem (vecs : NonCollinearVectors V) 
  (k : ℝ) 
  (hAB : ∃ (A B : V), B - A = 2 • vecs.a + k • vecs.b)
  (hCB : ∃ (C B : V), B - C = vecs.a + 3 • vecs.b)
  (hCD : ∃ (C D : V), D - C = 2 • vecs.a - vecs.b)
  (hCollinear : ∃ (A B D : V) (t : ℝ), B - A = t • (D - A)) :
  k = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l686_68621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_length_bound_l686_68635

/-- A convex broken line in a plane -/
structure ConvexBrokenLine where
  points : List (ℝ × ℝ)
  is_convex : Bool -- Changed to Bool for simplicity

/-- External angle at a point in a broken line -/
noncomputable def external_angle (l : ConvexBrokenLine) (p : ℕ) : ℝ := sorry

/-- The length of a broken line -/
noncomputable def length (l : ConvexBrokenLine) : ℝ := sorry

/-- The sum of external angles of a broken line -/
noncomputable def sum_external_angles (l : ConvexBrokenLine) : ℝ := sorry

theorem broken_line_length_bound 
  (A B : ℝ × ℝ) 
  (l : ConvexBrokenLine) 
  (h_distance : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 1) 
  (h_endpoints : l.points.head? = some A ∧ l.points.getLast? = some B)
  (h_angle_sum : sum_external_angles l < Real.pi) : 
  length l ≤ 1 / Real.cos (sum_external_angles l / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_length_bound_l686_68635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_age_difference_l686_68678

theorem cricket_team_age_difference 
  (total_members : ℕ) 
  (avg_age : ℚ) 
  (wicket_keeper_age_diff : ℚ) 
  (remaining_avg_age : ℚ) 
  (h1 : total_members = 11)
  (h2 : avg_age = 27)
  (h3 : wicket_keeper_age_diff = 3)
  (h4 : remaining_avg_age = 24)
  : avg_age - (((total_members : ℚ) * avg_age - avg_age - (avg_age + wicket_keeper_age_diff)) / 9) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_age_difference_l686_68678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_for_specific_case_l686_68614

/-- The height of a cone formed by rolling one sector of a circular sheet. -/
noncomputable def cone_height (r : ℝ) (n : ℕ) : ℝ :=
  Real.sqrt (r^2 - (2 * r / n)^2)

/-- Theorem: The height of a cone formed by rolling one sector of a circular
    sheet with radius 8 cm, when cut into four congruent sectors, is 2√15 cm. -/
theorem cone_height_for_specific_case :
  cone_height 8 4 = 2 * Real.sqrt 15 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_for_specific_case_l686_68614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closer_region_probability_l686_68601

/-- Triangle with side lengths a, b, and c -/
structure Triangle (a b c : ℝ) where
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The area of a triangle given its side lengths using Heron's formula -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- The region closer to one vertex than the other two in a triangle -/
noncomputable def closerRegionArea (t : Triangle a b c) : ℝ := sorry

theorem closer_region_probability (t : Triangle 7 6 5) :
  closerRegionArea t / triangleArea 7 6 5 = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closer_region_probability_l686_68601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_intersecting_circle_radius_correct_l686_68626

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    this function returns the largest radius of a circle centered at 
    one endpoint of the minor axis that still intersects the ellipse. -/
noncomputable def largest_intersecting_circle_radius (a b : ℝ) : ℝ :=
  if b ≤ Real.sqrt (a^2 - b^2)
  then a^2 / Real.sqrt (a^2 - b^2)
  else 2 * b

/-- Theorem stating that the largest_intersecting_circle_radius function 
    correctly computes the largest radius of a circle centered at one endpoint 
    of the minor axis of an ellipse that still intersects the ellipse. -/
theorem largest_intersecting_circle_radius_correct (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a ≥ b) :
  let r := largest_intersecting_circle_radius a b
  ∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 → 
  ∃ (x' y' : ℝ), x'^2 + (y' - b)^2 = r^2 ∧
  (∀ (r' : ℝ), r' > r → 
    ¬∃ (x'' y'' : ℝ), x''^2/a^2 + y''^2/b^2 = 1 ∧ 
    x''^2 + (y'' - b)^2 = r'^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_intersecting_circle_radius_correct_l686_68626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_is_real_iff_k_in_range_l686_68607

/-- The function f(x) parameterized by k -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x + 7) / (k * x^2 + 4 * k * x + 3)

/-- The domain of f(x) is ℝ if and only if k is in [0, 3/4) -/
theorem domain_is_real_iff_k_in_range :
  ∀ k : ℝ, (∀ x : ℝ, ∃ y : ℝ, f k x = y) ↔ (0 ≤ k ∧ k < 3/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_is_real_iff_k_in_range_l686_68607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_l686_68652

noncomputable def a (n : ℕ) : ℝ := (2 * (n : ℝ)^3) / ((n : ℝ)^3 - 2)

theorem limit_of_sequence : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 2| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_l686_68652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_neg_seven_equals_one_l686_68683

-- Define the functions f and g
def f (x : ℝ) : ℝ := 5 * x - 12

noncomputable def g (x : ℝ) : ℝ := 3 * ((f⁻¹ x) ^ 2) + 4 * (f⁻¹ x) - 6

-- State the theorem
theorem g_of_neg_seven_equals_one : g (-7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_neg_seven_equals_one_l686_68683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l686_68669

/-- Given a hyperbola with equation x^2 - 3y^2 = 1, its asymptotes are y = ± (√3/3)x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  x^2 - 3 * y^2 = 1 →
  ∃ (k : ℝ), k = Real.sqrt 3 / 3 ∧ (y = k * x ∨ y = -k * x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l686_68669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_coloring_theorem_l686_68688

/-- Represents a face of a unit cube -/
structure CubeFace :=
  (marked : Bool)

/-- Represents an edge of a unit cube -/
structure CubeEdge :=
  (color : Bool)

/-- Represents a cube composed of unit cubes -/
structure Cube (n : ℕ) :=
  (faces : List CubeFace)
  (edges : List CubeEdge)

/-- Represents a path through the cube -/
structure CubePath :=
  (segments : List (ℕ × ℕ × ℕ))

/-- Checks if a face has the correct coloring based on whether it's marked -/
def validFaceColoring (f : CubeFace) (adjacentEdges : List CubeEdge) : Prop :=
  if f.marked
  then (adjacentEdges.filter (λ e => e.color)).length % 2 = 1 ∧
       (adjacentEdges.filter (λ e => ¬e.color)).length % 2 = 1
  else (adjacentEdges.filter (λ e => e.color)).length % 2 = 0 ∧
       (adjacentEdges.filter (λ e => ¬e.color)).length % 2 = 0

/-- The main theorem -/
theorem cube_edge_coloring_theorem (n : ℕ) (c : Cube n) (p : CubePath) :
  ∃ (coloring : List CubeEdge),
    ∀ (f : CubeFace) (adjEdges : List CubeEdge),
      f ∈ c.faces → adjEdges ⊆ coloring → validFaceColoring f adjEdges :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_coloring_theorem_l686_68688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_l686_68686

/-- The speed of a goods train passing a man in an opposite-moving train --/
theorem goods_train_speed 
  (man_speed : ℝ) 
  (goods_length : ℝ) 
  (passing_time : ℝ) 
  (h1 : man_speed = 60) 
  (h2 : goods_length = 280) 
  (h3 : passing_time = 9) : 
  ∃ (goods_speed : ℝ), 
    goods_speed > 0 ∧ 
    goods_speed < man_speed ∧ 
    (goods_length / passing_time) * 3.6 = man_speed + goods_speed ∧ 
    abs (goods_speed - 52) < 1 := by
  sorry

#check goods_train_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_l686_68686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_and_rounding_l686_68657

-- Define the dividend and divisor
def dividend : ℚ := 2.2
def divisor : ℕ := 6

-- Define the repeating decimal representation
def repeating_decimal : ℚ := 0.366666666666

-- Define the rounded value
def rounded_value : ℚ := 0.37

-- Helper function for rounding (not implemented)
noncomputable def round_to_decimal_places (n : ℕ) (q : ℚ) : ℚ := 
  sorry

-- Theorem statement
theorem division_and_rounding :
  (dividend / (divisor : ℚ) = repeating_decimal) ∧
  (round_to_decimal_places 2 (dividend / (divisor : ℚ)) = rounded_value) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_and_rounding_l686_68657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_theory_problems_l686_68620

theorem number_theory_problems :
  (∀ k : ℤ, ¬ (∃ m : ℤ, k^2 ≡ m [ZMOD 10] ∧ (m = 2 ∨ m = 3 ∨ m = 7 ∨ m = 8))) ∧
  (∀ n : ℕ, ¬ (∃ a : ℕ, 5*n + 2 = a^2) ∧ ¬ (∃ b : ℕ, 5*n + 3 = b^2)) ∧
  (∀ n : ℤ, ¬ (5 ∣ (n^2 + 3))) ∧
  (∀ n : ℕ, (∃ m : ℕ, n! + 97 = m^2) ↔ n = 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_theory_problems_l686_68620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_straight_line_divides_plane_into_two_regions_hyperbola_divides_plane_into_three_regions_three_parallel_lines_divide_plane_into_four_regions_l686_68698

-- Define a plane
def Plane := ℝ × ℝ

-- Define a straight line
def StraightLine (m a : ℝ) : Set Plane :=
  {p : Plane | p.2 = m * p.1 + a}

-- Define a hyperbola
def Hyperbola (a b h k : ℝ) : Set Plane :=
  {p : Plane | (p.1 - h)^2 / a^2 - (p.2 - k)^2 / b^2 = 1}

-- Define three parallel lines
def ThreeParallelLines (m a b c : ℝ) : Set (Set Plane) :=
  {StraightLine m a, StraightLine m b, StraightLine m c}

-- Theorem statements
theorem straight_line_divides_plane_into_two_regions (m a : ℝ) :
  ∃ (R₁ R₂ : Set Plane), R₁ ∪ R₂ = Set.univ ∧ R₁ ∩ R₂ = ∅ ∧
  R₁ ∪ R₂ ∪ (StraightLine m a) = Set.univ := by sorry

theorem hyperbola_divides_plane_into_three_regions (a b h k : ℝ) :
  ∃ (R₁ R₂ R₃ : Set Plane), R₁ ∪ R₂ ∪ R₃ = Set.univ ∧
  R₁ ∩ R₂ = ∅ ∧ R₁ ∩ R₃ = ∅ ∧ R₂ ∩ R₃ = ∅ ∧
  R₁ ∪ R₂ ∪ R₃ ∪ (Hyperbola a b h k) = Set.univ := by sorry

theorem three_parallel_lines_divide_plane_into_four_regions (m a b c : ℝ) :
  ∃ (R₁ R₂ R₃ R₄ : Set Plane), R₁ ∪ R₂ ∪ R₃ ∪ R₄ = Set.univ ∧
  R₁ ∩ R₂ = ∅ ∧ R₁ ∩ R₃ = ∅ ∧ R₁ ∩ R₄ = ∅ ∧
  R₂ ∩ R₃ = ∅ ∧ R₂ ∩ R₄ = ∅ ∧ R₃ ∩ R₄ = ∅ ∧
  R₁ ∪ R₂ ∪ R₃ ∪ R₄ ∪ (⋃₀ (ThreeParallelLines m a b c)) = Set.univ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_straight_line_divides_plane_into_two_regions_hyperbola_divides_plane_into_three_regions_three_parallel_lines_divide_plane_into_four_regions_l686_68698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l686_68602

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the foci
def F₁ : ℝ × ℝ := (0, -1)
def F₂ : ℝ × ℝ := (0, 1)

-- Define a line passing through F₁
def Line (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * p.1 - 1}

-- Define the perimeter of a triangle
noncomputable def TrianglePerimeter (p q r : ℝ × ℝ) : ℝ :=
  ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt +
  ((q.1 - r.1)^2 + (q.2 - r.2)^2).sqrt +
  ((r.1 - p.1)^2 + (r.2 - p.2)^2).sqrt

theorem ellipse_equation (M N : ℝ × ℝ) (m : ℝ)
  (h1 : M ∈ Ellipse 2 (Real.sqrt 3))
  (h2 : N ∈ Ellipse 2 (Real.sqrt 3))
  (h3 : M ∈ Line m)
  (h4 : N ∈ Line m)
  (h5 : TrianglePerimeter M F₂ N = 8) :
  Ellipse 2 (Real.sqrt 3) = Ellipse (Real.sqrt 4) (Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l686_68602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_max_length_MN_l686_68682

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 * t, t)

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := Real.sqrt (Real.cos θ ^ 2 + Real.sin θ)

-- Theorem for the length of AB
theorem length_AB :
  ∃ A B : ℝ × ℝ,
    (∃ t₁ t₂ : ℝ, A = line_l t₁ ∧ B = line_l t₂) ∧
    (∃ θ₁ θ₂ : ℝ, 
      Real.sqrt ((A.1 ^ 2 + A.2 ^ 2) : ℝ) = curve_C θ₁ ∧
      Real.sqrt ((B.1 ^ 2 + B.2 ^ 2) : ℝ) = curve_C θ₂) ∧
    Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) = (Real.sqrt 5 + 1) / 2 := by
  sorry

-- Theorem for the maximum length of MN
theorem max_length_MN :
  ∃ M N : ℝ × ℝ,
    (∃ θ₁ θ₂ : ℝ, 
      Real.sqrt ((M.1 ^ 2 + M.2 ^ 2) : ℝ) = curve_C θ₁ ∧
      Real.sqrt ((N.1 ^ 2 + N.2 ^ 2) : ℝ) = curve_C θ₂) ∧
    M.1 * N.1 + M.2 * N.2 = 0 ∧
    ∀ P Q : ℝ × ℝ,
      (∃ θ₃ θ₄ : ℝ, 
        Real.sqrt ((P.1 ^ 2 + P.2 ^ 2) : ℝ) = curve_C θ₃ ∧
        Real.sqrt ((Q.1 ^ 2 + Q.2 ^ 2) : ℝ) = curve_C θ₄) ∧
      P.1 * Q.1 + P.2 * Q.2 = 0 →
      Real.sqrt ((M.1 - N.1) ^ 2 + (M.2 - N.2) ^ 2) ≥ Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2) ∧
    Real.sqrt ((M.1 - N.1) ^ 2 + (M.2 - N.2) ^ 2) = Real.sqrt (1 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_max_length_MN_l686_68682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minus_b_equals_one_l686_68689

/-- A rectangle in a 2D plane --/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The specific rectangle from the problem --/
def specificRectangle : Rectangle where
  A := (5, 5)
  B := (9, 2)
  C := (11, 13)
  D := (15, 10)

/-- Theorem stating that a - b = 1 for the given rectangle --/
theorem a_minus_b_equals_one (rect : Rectangle) 
  (h1 : rect.A = (5, 5))
  (h2 : rect.B = (9, 2))
  (h3 : rect.C = (11, 13))
  (h4 : rect.D = (15, 10)) :
  rect.C.1 - rect.D.2 = 1 := by
  sorry

#check a_minus_b_equals_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minus_b_equals_one_l686_68689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yard_completion_time_l686_68671

noncomputable def carpenter_time (n : ℕ) : ℝ := n

noncomputable def combined_time (t₁ t₂ t₃ t₄ : ℝ) : ℝ :=
  1 / (1/t₁ + 1/t₂ + 1/t₃ + 1/t₄)

theorem yard_completion_time :
  let t₁ := carpenter_time 1
  let t₂ := carpenter_time 2
  let t₃ := carpenter_time 3
  let t₄ := carpenter_time 4
  combined_time t₁ t₂ t₃ t₄ = 12/25 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval combined_time 1 2 3 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yard_completion_time_l686_68671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extreme_prime_factors_1155_l686_68655

theorem sum_of_extreme_prime_factors_1155 :
  ∃ (factors : List Nat),
    (factors.all Nat.Prime) ∧
    (factors.prod = 1155) ∧
    (factors.minimum? ≠ none) ∧
    (factors.maximum? ≠ none) ∧
    ((factors.minimum?.getD 0 + factors.maximum?.getD 0) = 14) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extreme_prime_factors_1155_l686_68655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l686_68692

/-- Given vectors a and b in ℝ², prove that if a + λb is perpendicular to a, then λ = 13. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (lambda : ℝ) :
  a = (3, -2) →
  b = (1, 2) →
  (a.1 + lambda * b.1, a.2 + lambda * b.2) • a = 0 →
  lambda = 13 := by
  sorry

#check perpendicular_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l686_68692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_a_1_l686_68638

def sequence_a : ℕ → ℚ
  | 0 => 6/7  -- Add a case for 0
  | 1 => 6/7
  | (n+1) => 
      if 0 ≤ sequence_a n ∧ sequence_a n ≤ 1/2 then 
        2 * sequence_a n
      else if 1/2 < sequence_a n ∧ sequence_a n < 1 then 
        2 * sequence_a n - 1
      else 
        sequence_a n

theorem a_2017_equals_a_1 : sequence_a 2017 = sequence_a 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_a_1_l686_68638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_a_eq_two_max_value_of_m_is_one_a_leq_one_for_f_geq_zero_zeros_of_h_l686_68625

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2*x + Real.log x - a*(x^2 + x)

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2 + 1/x - a*(2*x + 1)

theorem tangent_line_parallel_implies_a_eq_two (a : ℝ) :
  f_derivative a 1 = -3 → a = 2 := by sorry

/-- The function m(x) as defined in the solution -/
noncomputable def m (x : ℝ) : ℝ := (2*x + Real.log x) / (x^2 + x)

theorem max_value_of_m_is_one :
  ∀ x > 0, m x ≤ 1 := by sorry

theorem a_leq_one_for_f_geq_zero (a : ℝ) :
  (∃ x > 0, f a x ≥ 0) → a ≤ 1 := by sorry

/-- The functions p(x) and q(x) as defined in the problem -/
noncomputable def p (x : ℝ) : ℝ := 1 - Real.log x
noncomputable def q (m : ℝ) (x : ℝ) : ℝ := x^3 - m*x + Real.exp 1

/-- The function h(x) as defined in the problem -/
noncomputable def h (m : ℝ) (x : ℝ) : ℝ :=
  (p x + q m x + |p x - q m x|) / 2

theorem zeros_of_h (m : ℝ) :
  (m < Real.exp 2 + 1 → ∀ x, h m x ≠ 0) ∧
  (m = Real.exp 2 + 1 → ∃! x, h m x = 0) ∧
  (m > Real.exp 2 + 1 → ∃ x y, x ≠ y ∧ h m x = 0 ∧ h m y = 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_a_eq_two_max_value_of_m_is_one_a_leq_one_for_f_geq_zero_zeros_of_h_l686_68625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_along_line_segment_l686_68617

noncomputable def line_integral (f : ℝ × ℝ → ℝ) (a b : ℝ × ℝ) : ℝ :=
  ∫ t in (0 : ℝ)..1, f (a.1 + t * (b.1 - a.1), a.2 + t * (b.2 - a.2)) * 
    Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

noncomputable def f (p : ℝ × ℝ) : ℝ := 1 / Real.sqrt (|p.1| + |p.2|)

theorem integral_along_line_segment :
  line_integral f (0, -2) (4, 0) = 2 * (Real.sqrt 20 - Real.sqrt 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_along_line_segment_l686_68617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_magnitude_bounds_l686_68609

/-- The maximum and minimum values of |z| for a complex number z defined in terms of θ. -/
theorem complex_number_magnitude_bounds (θ : Real) (h : θ ∈ Set.Icc 0 (Real.pi / 2)) :
  let z : ℂ := 2 * (Real.cos θ)^2 + Complex.I + Complex.I * (Real.sin θ + Real.cos θ)^2
  (∃ (θ_max : Real), θ_max ∈ Set.Icc 0 (Real.pi / 2) ∧ Complex.abs z ≤ Complex.abs (2 * (Real.cos θ_max)^2 + Complex.I + Complex.I * (Real.sin θ_max + Real.cos θ_max)^2)) ∧
  (Complex.abs z ≥ 2) ∧
  (∃ (θ_min θ_max : Real), 
    θ_min ∈ Set.Icc 0 (Real.pi / 2) ∧ 
    θ_max ∈ Set.Icc 0 (Real.pi / 2) ∧ 
    Complex.abs (2 * (Real.cos θ_min)^2 + Complex.I + Complex.I * (Real.sin θ_min + Real.cos θ_min)^2) = 2 ∧
    Complex.abs (2 * (Real.cos θ_max)^2 + Complex.I + Complex.I * (Real.sin θ_max + Real.cos θ_max)^2) = Real.sqrt 5 + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_magnitude_bounds_l686_68609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l686_68608

-- Define the ellipse M
def ellipse_M (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ (y^2 / a^2 + x^2 / b^2 = 1)

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the circle (rename to avoid conflict with built-in circle)
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the eccentricity relation
def eccentricity_relation (e_ellipse e_hyperbola : ℝ) : Prop :=
  e_ellipse * e_hyperbola = 1

-- Define the internal tangency condition
def internal_tangency (M : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), M x y ∧ circle_eq x y

-- Define point A
noncomputable def point_A : ℝ × ℝ := (-2, Real.sqrt 2)

-- Define the lower focus F
def lower_focus (F : ℝ × ℝ) (M : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), F = (x, y) ∧ M x y ∧ y < 0

-- Helper function definitions (not proven)
noncomputable def perimeter_triangle (A B C : ℝ × ℝ) : ℝ := sorry
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem ellipse_properties :
  ∃ (M : ℝ → ℝ → Prop) (F : ℝ × ℝ),
    (∀ x y, M x y ↔ y^2 / 4 + x^2 / 2 = 1) ∧
    ellipse_M = M ∧
    internal_tangency M ∧
    lower_focus F M ∧
    (∃ (P : ℝ × ℝ),
      M P.1 P.2 ∧
      (∀ (Q : ℝ × ℝ), M Q.1 Q.2 →
        perimeter_triangle point_A F Q ≤ perimeter_triangle point_A F P) ∧
      perimeter_triangle point_A F P = 6 + 2 * Real.sqrt 3 ∧
      area_triangle point_A F P = 3 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l686_68608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_folding_cutting_area_l686_68663

/-- Represents the area of a square paper after folding and cutting -/
noncomputable def remaining_area (side_length : ℝ) : ℝ :=
  let initial_area := side_length ^ 2
  let folded_area := initial_area / 4
  let cut_area := folded_area
  initial_area - cut_area

/-- Theorem stating that for a square paper with side length 10 cm, 
    the remaining area after folding twice and cutting is 75 sq cm -/
theorem paper_folding_cutting_area :
  remaining_area 10 = 75 := by
  -- Unfold the definition of remaining_area
  unfold remaining_area
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_folding_cutting_area_l686_68663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l686_68623

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x + a

noncomputable def g (x : ℝ) : ℝ := Real.log x - 2 * x

-- Define the interval [1/2, 2]
def I : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem function_inequality (a : ℝ) : 
  (∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → f a x₁ ≤ g x₂) → a ≤ Real.log 2 - 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l686_68623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_onto_b_l686_68679

def a : Fin 2 → ℚ := ![1, 2]
def b : Fin 2 → ℚ := ![-2, 4]

def dotProduct (u v : Fin 2 → ℚ) : ℚ :=
  (u 0) * (v 0) + (u 1) * (v 1)

def projection (u v : Fin 2 → ℚ) : Fin 2 → ℚ :=
  let scalar := (dotProduct u v) / (dotProduct v v)
  fun i => scalar * (v i)

theorem projection_a_onto_b :
  projection a b = ![-(3/5), 6/5] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_onto_b_l686_68679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_exists_non_triangle_obtuse_triangle_root_l686_68693

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ := a^x + b^x - c^x

-- Define the set M
def M : Set (ℝ × ℝ × ℝ) := {(a, b, c) | a = b ∧ a + b ≤ c ∧ c > a ∧ a > 0 ∧ c > b ∧ b > 0}

-- Theorem 1
theorem root_in_interval (a b c : ℝ) (h : (a, b, c) ∈ M) :
  ∃ x, x ∈ Set.Ioo 0 1 ∧ f a b c x = 0 := by
  sorry

-- Theorem 2
theorem exists_non_triangle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ x, ¬(a^x + b^x > c^x ∧ a^x + c^x > b^x ∧ b^x + c^x > a^x) := by
  sorry

-- Theorem 3
theorem obtuse_triangle_root (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_obtuse : c^2 > a^2 + b^2) :
  ∃ x ∈ Set.Ioo 1 2, f a b c x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_exists_non_triangle_obtuse_triangle_root_l686_68693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_passes_through_mean_error_variance_measures_forecast_accuracy_l686_68691

/-- Represents a linear regression model -/
structure LinearRegression where
  x : ℝ → ℝ  -- Independent variable
  y : ℝ → ℝ  -- Dependent variable
  a : ℝ      -- Intercept
  b : ℝ      -- Slope
  e : ℝ → ℝ  -- Random error term

/-- The mean of a real-valued function over a finite interval -/
noncomputable def mean (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (∫ x in a..b, f x) / (b - a)

/-- The variance of a real-valued function over a finite interval -/
noncomputable def variance (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  mean (fun x => (f x - mean f a b)^2) a b

/-- The regression line passes through the mean point -/
theorem regression_line_passes_through_mean (model : LinearRegression) (a b : ℝ) :
  model.y (mean model.x a b) = model.b * (mean model.x a b) + model.a := by
  sorry

/-- The error variance measures forecast accuracy -/
theorem error_variance_measures_forecast_accuracy (model : LinearRegression) (a b : ℝ) :
  ∃ (accuracy : ℝ → ℝ), accuracy (variance model.e a b) = accuracy (variance (fun x => model.y x - (model.b * model.x x + model.a)) a b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_passes_through_mean_error_variance_measures_forecast_accuracy_l686_68691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l686_68651

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin (3 * x) + 5 * Real.sqrt 3 * Real.cos (3 * x)

-- State the theorem
theorem f_monotone_increasing : 
  MonotoneOn f (Set.Icc 0 (Real.pi / 20)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l686_68651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_capital_growth_theorem_l686_68670

/-- Represents the remaining capital (in thousands of yuan) after n years -/
noncomputable def remaining_capital (n : ℕ) : ℝ :=
  4500 * (3/2)^(n-1) + 2000

/-- The initial capital in thousands of yuan -/
def initial_capital : ℝ := 50000

/-- The annual growth rate -/
def growth_rate : ℝ := 0.5

/-- The annual submission in thousands of yuan -/
def annual_submission : ℝ := 10000

/-- The theorem stating the correctness of the remaining capital formula and the minimum year to exceed 300 million yuan -/
theorem capital_growth_theorem :
  (∀ n : ℕ, remaining_capital n = 
    (initial_capital * (1 + growth_rate)^n - annual_submission * (((1 + growth_rate)^n - 1) / growth_rate))) ∧
  (∀ m : ℕ, m ≥ 6 ↔ remaining_capital m > 30000) := by
  sorry

#check capital_growth_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_capital_growth_theorem_l686_68670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_l686_68600

/-- Proves that given a train of length 500 m, moving at 174.98560115190784 km/h,
    taking 10 seconds to cross a man walking in the opposite direction,
    the speed of the man is 5.014396850094164 km/h. -/
theorem man_speed (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) :
  train_length = 500 →
  train_speed = 174.98560115190784 →
  crossing_time = 10 →
  let relative_speed := train_length / crossing_time
  let train_speed_ms := train_speed * 1000 / 3600
  let man_speed_ms := relative_speed - train_speed_ms
  let man_speed_kmh := man_speed_ms * 3600 / 1000
  man_speed_kmh = 5.014396850094164 := by
  sorry

#check man_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_l686_68600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_primes_ending_in_three_l686_68636

theorem two_digit_primes_ending_in_three : 
  (Finset.filter (λ x : ℕ => 10 ≤ x ∧ x < 100 ∧ Nat.Prime x ∧ x % 10 = 3) (Finset.range 100)).card = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_primes_ending_in_three_l686_68636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_l686_68666

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * sin x + a * x

theorem a_value (a : ℝ) : 
  (∀ x, deriv (f a) x = sin x + x * cos x + a) →
  deriv (f a) (π/2) = 1 →
  a = 0 := by
  intros h1 h2
  have h3 : deriv (f a) (π/2) = sin (π/2) + (π/2) * cos (π/2) + a := h1 (π/2)
  rw [h2, sin_pi_div_two, cos_pi_div_two, mul_zero] at h3
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_l686_68666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rate_of_change_f_on_1_2_l686_68659

-- Define the function f(x) = 1 + 1/x
noncomputable def f (x : ℝ) : ℝ := 1 + 1/x

-- Define the average rate of change function
noncomputable def averageRateOfChange (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (f b - f a) / (b - a)

-- Theorem statement
theorem average_rate_of_change_f_on_1_2 :
  averageRateOfChange f 1 2 = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rate_of_change_f_on_1_2_l686_68659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_A_B_l686_68653

def A : ℤ := 2 * 3 + 4 * 5 + 6 * 7 + 8 * 9 + 10 * 11 + 12 * 13 + 14 * 15 + 16 * 17 + 18

def B : ℤ := 2 + 3 * 4 + 5 * 6 + 7 * 8 + 9 * 10 + 11 * 12 + 13 * 14 + 15 * 16 + 17 * 18

theorem difference_A_B : |A - B| = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_A_B_l686_68653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l686_68634

/-- A parabola in the cartesian coordinate plane -/
structure Parabola where
  /-- The equation of the parabola in the form y^2 = 4x -/
  equation : ℝ → ℝ → Prop

/-- A point on a parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.equation x y

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := (1, 0)

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_point_x_coordinate 
  (p : Parabola)
  (point : ParabolaPoint p)
  (h : p.equation = fun x y ↦ y^2 = 4*x)
  (dist_to_focus : distance (point.x, point.y) (focus p) = 5) :
  point.x = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l686_68634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_circle_relation_l686_68630

/-- A circle that always bisects another circle -/
structure BisectingCircle (a b : ℝ) where
  -- Circle 1 equation: (x-a)^2 + (y-b)^2 = b^2 + 1
  eq_circle1 : ∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = b^2 + 1 → True
  -- Circle 2 equation: (x+1)^2 + (y+1)^2 = 4
  eq_circle2 : ∀ (x y : ℝ), (x + 1)^2 + (y + 1)^2 = 4 → True
  -- Circle 1 always bisects the circumference of Circle 2
  bisects : ∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = b^2 + 1 → 
    (x + 1)^2 + (y + 1)^2 = 4 → ∃ (z w : ℝ), (z - a)^2 + (w - b)^2 = b^2 + 1 ∧ 
    (z + 1)^2 + (w + 1)^2 = 4 ∧ (x, y) ≠ (z, w)

/-- The relationship between a and b for a bisecting circle -/
theorem bisecting_circle_relation (a b : ℝ) (h : BisectingCircle a b) : 
  a^2 + 2*a + 2*b + 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_circle_relation_l686_68630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_angle_l686_68694

/-- Given an isosceles triangle with angle α satisfying tan(α/2) = 2,
    θ is the angle between the altitude and the angle bisector drawn from
    the vertex angle to the base of the triangle. -/
theorem isosceles_triangle_angle (α θ : ℝ) : 
  Real.tan (α / 2) = 2 → Real.tan θ = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_angle_l686_68694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_l686_68658

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 3^x + a / (3^x + 1)

-- State the theorem
theorem min_value_implies_a (a : ℝ) :
  a > 0 → (∀ x : ℝ, f a x ≥ 5) → (∃ x : ℝ, f a x = 5) → a = 9 :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_l686_68658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_in_possible_areas_l686_68643

/-- Represents a point on the side of a square --/
structure SquarePoint where
  side : Fin 4
  position : Fin 3

/-- Represents a quadrilateral formed by four points on a square --/
structure Quadrilateral where
  p1 : SquarePoint
  p2 : SquarePoint
  p3 : SquarePoint
  p4 : SquarePoint
  distinct_sides : p1.side ≠ p2.side ∧ p1.side ≠ p3.side ∧ p1.side ≠ p4.side ∧
                   p2.side ≠ p3.side ∧ p2.side ≠ p4.side ∧ p3.side ≠ p4.side

/-- The set of possible areas for the quadrilateral --/
def PossibleAreas : Set ℝ := {16, 18, 20}

/-- Calculates the area of a quadrilateral formed on a square with side length 6 --/
noncomputable def quadrilateralArea (q : Quadrilateral) : ℝ := sorry

/-- Theorem stating that the area of any valid quadrilateral is in the set of possible areas --/
theorem quadrilateral_area_in_possible_areas (q : Quadrilateral) :
  quadrilateralArea q ∈ PossibleAreas := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_in_possible_areas_l686_68643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l686_68673

noncomputable section

-- Define the function f
def f (ω : ℝ) (a : ℝ) (x : ℝ) : ℝ := 4 * Real.cos (ω * x) * Real.sin (ω * x + Real.pi / 6) + a

-- State the theorem
theorem function_properties (ω : ℝ) (a : ℝ) (h_ω_pos : ω > 0) 
  (h_max : ∀ x, f ω a x ≤ 2)
  (h_period : ∀ x, f ω a (x + Real.pi) = f ω a x) :
  (a = -1 ∧ ω = 1) ∧
  (∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 
    ∀ y ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3),
    x < y → f ω a y < f ω a x) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l686_68673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_count_l686_68645

theorem integer_solutions_count : 
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ s ↔ x^4 + y^2 = 4*y) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_count_l686_68645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_ratio_sum_n_k_l686_68660

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k+1) : ℚ) = 1/3 ∧
  (Nat.choose n (k+1) : ℚ) / (Nat.choose n (k+2) : ℚ) = 1/2 →
  n = 11 ∧ k = 2 :=
by
  sorry

theorem sum_n_k (n k : ℕ) :
  (Nat.choose n k : ℚ) / (Nat.choose n (k+1) : ℚ) = 1/3 ∧
  (Nat.choose n (k+1) : ℚ) / (Nat.choose n (k+2) : ℚ) = 1/2 →
  n + k = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_ratio_sum_n_k_l686_68660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l686_68647

/-- Calculates the length of a platform given train parameters --/
theorem platform_length
  (train_length : ℝ)
  (train_speed_kmph : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 150)
  (h2 : train_speed_kmph = 75)
  (h3 : crossing_time = 20) :
  train_speed_kmph * 1000 / 3600 * crossing_time - train_length = 1350 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l686_68647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_different_dance_counts_l686_68616

/-- Represents a dance pairing between a boy and a girl -/
structure DancePair where
  boy : Fin 29
  girl : Fin 15

/-- Represents the dance configuration for the ball -/
def DanceConfiguration := List DancePair

/-- Returns the number of dances for a given person -/
def danceCount (config : DanceConfiguration) (person : Fin 29 ⊕ Fin 15) : Nat :=
  match person with
  | Sum.inl boy => (config.filter (fun pair => pair.boy = boy)).length
  | Sum.inr girl => (config.filter (fun pair => pair.girl = girl)).length

/-- The theorem stating the maximum number of different dance counts -/
theorem max_different_dance_counts (config : DanceConfiguration) :
  (Finset.image (danceCount config) (Finset.univ : Finset (Fin 29 ⊕ Fin 15))).card ≤ 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_different_dance_counts_l686_68616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_natural_numbers_reachable_l686_68613

def board_operation (x : ℕ) : ℕ → ℕ
  | 0 => 3 * x + 1
  | _ => x / 2

def reachable (n : ℕ) : Prop :=
  ∃ (seq : List ℕ), seq.head? = some 1 ∧ seq.getLast? = some n ∧
    ∀ (i : ℕ), i < seq.length - 1 →
      (board_operation (seq[i]!) 0 = seq[i+1]!) ∨
      (board_operation (seq[i]!) 1 = seq[i+1]!)

theorem all_natural_numbers_reachable :
  ∀ (n : ℕ), reachable n := by
  sorry

#check all_natural_numbers_reachable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_natural_numbers_reachable_l686_68613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_side_b_length_l686_68604

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi ∧
  0 < t.B ∧ t.B < Real.pi ∧
  0 < t.C ∧ t.C < Real.pi ∧
  t.A + t.B + t.C = Real.pi ∧
  Real.tan t.B = 2 ∧
  Real.tan t.C = 3

-- Theorem for part 1
theorem angle_A_measure (t : Triangle) (h : triangle_conditions t) : t.A = Real.pi / 4 := by
  sorry

-- Theorem for part 2
theorem side_b_length (t : Triangle) (h : triangle_conditions t) (hc : t.c = 3) : t.b = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_side_b_length_l686_68604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l686_68612

theorem trigonometric_problem (α : ℝ) 
  (h1 : Real.sin (α + Real.pi/4) = Real.sqrt 2/10)
  (h2 : Real.pi/2 < α ∧ α < Real.pi) :
  (Real.cos α = -3/5) ∧ 
  (Real.sin (2*α - Real.pi/4) = -17*Real.sqrt 2/50) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l686_68612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sequences_count_l686_68695

def word : String := "EQUALS"

def valid_sequence (s : List Char) : Prop :=
  s.length = 5 ∧
  s.head? = some 'E' ∧
  s.get? 1 = some 'Q' ∧
  s.getLast? = some 'S' ∧
  s.toFinset.card = 5

def remaining_letters : Finset Char :=
  (word.toList.toFinset).filter (λ c => c ∉ ['E', 'Q', 'S'])

theorem distinct_sequences_count :
  (remaining_letters.toList.permutations).length = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sequences_count_l686_68695
