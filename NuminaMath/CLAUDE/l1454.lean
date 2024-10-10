import Mathlib

namespace bianca_candy_count_l1454_145488

def candy_problem (eaten : ℕ) (pieces_per_pile : ℕ) (num_piles : ℕ) : ℕ :=
  eaten + pieces_per_pile * num_piles

theorem bianca_candy_count : candy_problem 12 5 4 = 32 := by
  sorry

end bianca_candy_count_l1454_145488


namespace rectangle_equation_l1454_145469

theorem rectangle_equation (x : ℝ) : 
  (∀ L W : ℝ, L * W = 864 ∧ L + W = 60 ∧ L = W + x) →
  (60 - x) / 2 * (60 + x) / 2 = 864 := by
sorry

end rectangle_equation_l1454_145469


namespace fifteenth_replacement_in_april_l1454_145457

def months : List String := ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

def monthsAfterFebruary (n : Nat) : Nat :=
  (months.indexOf "February" + n) % months.length

theorem fifteenth_replacement_in_april :
  months[monthsAfterFebruary 98] = "April" := by
  sorry

end fifteenth_replacement_in_april_l1454_145457


namespace set_representability_l1454_145465

-- Define the items
def item1 : Type := Unit  -- Placeholder for vague concept
def item2 : Set ℝ := {x : ℝ | x^2 + 3 = 0}
def item3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = p.2}

-- Define a predicate for set representability
def is_set_representable (α : Type) : Prop := Nonempty (Set α)

-- State the theorem
theorem set_representability :
  ¬ is_set_representable item1 ∧ 
  is_set_representable item2 ∧ 
  is_set_representable item3 :=
sorry

end set_representability_l1454_145465


namespace lipstick_ratio_l1454_145435

def lipstick_problem (total_students : ℕ) (blue_lipstick : ℕ) : Prop :=
  let red_lipstick := blue_lipstick * 5
  let colored_lipstick := red_lipstick * 4
  colored_lipstick * 2 = total_students

theorem lipstick_ratio :
  lipstick_problem 200 5 :=
sorry

end lipstick_ratio_l1454_145435


namespace x_range_for_inequality_l1454_145410

theorem x_range_for_inequality (x : ℝ) : 
  (0 ≤ x ∧ x < (1 + Real.sqrt 13) / 3) ↔ 
  (∀ y : ℝ, y > 0 → (2 * (x * y^2 + x^2 * y + 2 * y^2 + 2 * x * y)) / (x + y) > 3 * x^2 * y) :=
by sorry

end x_range_for_inequality_l1454_145410


namespace carol_final_gold_tokens_l1454_145402

/-- Represents the state of Carol's tokens -/
structure TokenState where
  purple : ℕ
  green : ℕ
  gold : ℕ

/-- Defines the exchange rules -/
def exchange1 (state : TokenState) : TokenState :=
  { purple := state.purple - 3, green := state.green + 2, gold := state.gold + 1 }

def exchange2 (state : TokenState) : TokenState :=
  { purple := state.purple + 1, green := state.green - 4, gold := state.gold + 1 }

/-- Checks if an exchange is possible -/
def canExchange (state : TokenState) : Bool :=
  state.purple ≥ 3 ∨ state.green ≥ 4

/-- The initial state of Carol's tokens -/
def initialState : TokenState :=
  { purple := 100, green := 85, gold := 0 }

/-- The theorem to prove -/
theorem carol_final_gold_tokens :
  ∃ (finalState : TokenState),
    (¬canExchange finalState) ∧
    (finalState.gold = 90) ∧
    (∃ (n m : ℕ),
      finalState = (exchange2^[m] ∘ exchange1^[n]) initialState) :=
sorry

end carol_final_gold_tokens_l1454_145402


namespace functional_equation_solution_l1454_145407

/-- A function satisfying the given functional equation. -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y - y * f x) = f (x * y) - x * y

/-- The main theorem stating that functions satisfying the equation are either the identity or absolute value function. -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : SatisfiesEquation f) :
    (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = |x|) := by
  sorry


end functional_equation_solution_l1454_145407


namespace five_balls_four_boxes_l1454_145460

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 4 ways to distribute 5 indistinguishable balls into 4 indistinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 4 := by sorry

end five_balls_four_boxes_l1454_145460


namespace new_person_weight_l1454_145461

/-- Given a group of 8 people, if replacing one person weighing 65 kg
    with a new person increases the average weight by 3.5 kg,
    then the weight of the new person is 93 kg. -/
theorem new_person_weight
  (initial_count : Nat)
  (weight_increase : ℝ)
  (replaced_weight : ℝ)
  (h1 : initial_count = 8)
  (h2 : weight_increase = 3.5)
  (h3 : replaced_weight = 65)
  : ℝ :=
by
  sorry

end new_person_weight_l1454_145461


namespace max_y_coordinate_l1454_145490

theorem max_y_coordinate (x y : ℝ) : 
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 := by
  sorry

end max_y_coordinate_l1454_145490


namespace min_sum_of_product_3920_l1454_145451

theorem min_sum_of_product_3920 (x y z : ℕ+) (h : x * y * z = 3920) :
  ∃ (a b c : ℕ+), a * b * c = 3920 ∧ (∀ x' y' z' : ℕ+, x' * y' * z' = 3920 → a + b + c ≤ x' + y' + z') ∧ a + b + c = 70 := by
  sorry

end min_sum_of_product_3920_l1454_145451


namespace parabola_point_y_coord_l1454_145485

/-- A point on a parabola with a specific distance to the focus -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : x^2 = 4*y
  distance_to_focus : (x - 0)^2 + (y - 1)^2 = 2^2

/-- Theorem: The y-coordinate of a point on the parabola x^2 = 4y that is 2 units away from the focus (0, 1) is 1 -/
theorem parabola_point_y_coord (P : ParabolaPoint) : P.y = 1 := by
  sorry

end parabola_point_y_coord_l1454_145485


namespace f_less_than_g_l1454_145413

/-- Represents a board arrangement -/
def Board (m n : ℕ+) := Fin m → Fin n → Bool

/-- Number of arrangements with at least one row or column of noughts -/
def f (m n : ℕ+) : ℕ := sorry

/-- Number of arrangements with at least one row of noughts or column of crosses -/
def g (m n : ℕ+) : ℕ := sorry

/-- The theorem stating that f(m,n) < g(m,n) for all positive m and n -/
theorem f_less_than_g (m n : ℕ+) : f m n < g m n := by sorry

end f_less_than_g_l1454_145413


namespace english_only_enrollment_l1454_145493

theorem english_only_enrollment (total : ℕ) (both : ℕ) (german : ℕ) 
  (h1 : total = 45)
  (h2 : both = 12)
  (h3 : german = 22)
  (h4 : german ≥ both) :
  total - german + both = 23 := by
  sorry

end english_only_enrollment_l1454_145493


namespace min_value_trig_expression_l1454_145406

theorem min_value_trig_expression (x : ℝ) :
  (Real.sin x)^8 + (Real.cos x)^8 + 2 ≥ 5/4 * ((Real.sin x)^6 + (Real.cos x)^6 + 2) := by
  sorry

end min_value_trig_expression_l1454_145406


namespace pet_store_birds_l1454_145489

theorem pet_store_birds (num_cages : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ)
  (h1 : num_cages = 9)
  (h2 : parrots_per_cage = 2)
  (h3 : parakeets_per_cage = 2) :
  num_cages * (parrots_per_cage + parakeets_per_cage) = 36 := by
  sorry

end pet_store_birds_l1454_145489


namespace seven_eighths_of_48_l1454_145446

theorem seven_eighths_of_48 : (7 / 8 : ℚ) * 48 = 42 := by
  sorry

end seven_eighths_of_48_l1454_145446


namespace max_fences_for_100_houses_prove_max_fences_199_l1454_145481

/-- Represents a village with houses and fences. -/
structure Village where
  num_houses : ℕ
  num_fences : ℕ

/-- Represents the process of combining houses within a fence. -/
def combine_houses (v : Village) : Village :=
  { num_houses := v.num_houses - 1
  , num_fences := v.num_fences - 2 }

/-- The maximum number of fences for a given number of houses. -/
def max_fences (n : ℕ) : ℕ :=
  2 * n - 1

/-- Theorem stating the maximum number of fences for 100 houses. -/
theorem max_fences_for_100_houses :
  ∃ (v : Village), v.num_houses = 100 ∧ v.num_fences = max_fences v.num_houses :=
by
  sorry

/-- Theorem proving that 199 is the maximum number of fences for 100 houses. -/
theorem prove_max_fences_199 :
  max_fences 100 = 199 :=
by
  sorry

end max_fences_for_100_houses_prove_max_fences_199_l1454_145481


namespace quadratic_equation_roots_l1454_145421

theorem quadratic_equation_roots (c : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 16 * x + c = 0 ↔ x = (-16 + Real.sqrt 24) / 4 ∨ x = (-16 - Real.sqrt 24) / 4) →
  c = 29 := by
sorry

end quadratic_equation_roots_l1454_145421


namespace greatest_sum_consecutive_integers_l1454_145422

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500 ∧ (n + 1) * (n + 2) ≥ 500) → n + (n + 1) = 43 := by
  sorry

end greatest_sum_consecutive_integers_l1454_145422


namespace puzzle_solution_l1454_145433

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- The problem statement -/
theorem puzzle_solution 
  (EH OY AY OH : TwoDigitNumber)
  (h1 : EH.val = 4 * OY.val)
  (h2 : AY.val = 4 * OH.val) :
  EH.val + OY.val + AY.val + OH.val = 150 :=
sorry

end puzzle_solution_l1454_145433


namespace geometric_sequence_problem_l1454_145439

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  a 1 + a 2 + a 3 + a 4 + a 5 = 3 →
  a 1^2 + a 2^2 + a 3^2 + a 4^2 + a 5^2 = 15 →
  a 1 - a 2 + a 3 - a 4 + a 5 = 5 :=
by sorry

end geometric_sequence_problem_l1454_145439


namespace factor_expression_l1454_145436

theorem factor_expression (y : ℝ) : 16 * y^2 + 8 * y = 8 * y * (2 * y + 1) := by
  sorry

end factor_expression_l1454_145436


namespace homework_problem_l1454_145486

theorem homework_problem (p t : ℕ) (h1 : p > 12) (h2 : t > 0) 
  (h3 : p * t = (p + 6) * (t - 3)) : p * t = 140 := by
  sorry

end homework_problem_l1454_145486


namespace equation_solutions_l1454_145464

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 = x ↔ x = 0 ∨ x = 1/4) ∧
  (∀ x : ℝ, x^2 - 18*x + 1 = 0 ↔ x = 9 + 4*Real.sqrt 5 ∨ x = 9 - 4*Real.sqrt 5) := by
  sorry

end equation_solutions_l1454_145464


namespace tourism_revenue_scientific_notation_l1454_145445

/-- Represents the value of 1 billion in scientific notation -/
def billion : ℝ := 10^9

/-- The tourism revenue in billions of yuan -/
def tourism_revenue : ℝ := 12.41

theorem tourism_revenue_scientific_notation : 
  tourism_revenue * billion = 1.241 * (10 : ℝ)^9 := by sorry

end tourism_revenue_scientific_notation_l1454_145445


namespace polynomial_coefficient_bound_l1454_145447

/-- A real polynomial of degree 3 -/
structure Polynomial3 where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Evaluation of the polynomial at a point x -/
def Polynomial3.eval (p : Polynomial3) (x : ℝ) : ℝ :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- The condition that |p(x)| ≤ 1 for all x such that |x| ≤ 1 -/
def BoundedOnUnitInterval (p : Polynomial3) : Prop :=
  ∀ x : ℝ, |x| ≤ 1 → |p.eval x| ≤ 1

/-- The theorem statement -/
theorem polynomial_coefficient_bound (p : Polynomial3) 
  (h : BoundedOnUnitInterval p) : 
  |p.a| + |p.b| + |p.c| + |p.d| ≤ 7 := by
  sorry

end polynomial_coefficient_bound_l1454_145447


namespace cos_555_degrees_l1454_145474

theorem cos_555_degrees : Real.cos (555 * Real.pi / 180) = -(Real.sqrt 6 / 4 + Real.sqrt 2 / 4) := by
  sorry

end cos_555_degrees_l1454_145474


namespace sqrt_2_times_sqrt_8_l1454_145453

theorem sqrt_2_times_sqrt_8 : Real.sqrt 2 * Real.sqrt 8 = 4 := by
  sorry

end sqrt_2_times_sqrt_8_l1454_145453


namespace hyperbola_eccentricity_l1454_145456

/-- Given an ellipse and a hyperbola with related equations, prove that the hyperbola's eccentricity is √6/2 -/
theorem hyperbola_eccentricity
  (m n : ℝ)
  (h_pos : 0 < m ∧ m < n)
  (h_ellipse : ∀ x y : ℝ, m * x^2 + n * y^2 = 1)
  (h_ellipse_ecc : Real.sqrt 2 / 2 = Real.sqrt (1 - (1/n) / (1/m)))
  (h_hyperbola : ∀ x y : ℝ, m * x^2 - n * y^2 = 1) :
  Real.sqrt 6 / 2 = Real.sqrt (1 + (1/n) / (1/m)) :=
sorry

end hyperbola_eccentricity_l1454_145456


namespace age_ratio_sandy_molly_l1454_145408

/-- Given that Sandy is 70 years old and Molly is 20 years older than Sandy,
    prove that the ratio of their ages is 7:9. -/
theorem age_ratio_sandy_molly :
  let sandy_age : ℕ := 70
  let age_difference : ℕ := 20
  let molly_age : ℕ := sandy_age + age_difference
  (sandy_age : ℚ) / (molly_age : ℚ) = 7 / 9 := by sorry

end age_ratio_sandy_molly_l1454_145408


namespace distance_XY_is_16_l1454_145462

-- Define the travel parameters
def travel_time_A : ℕ → Prop := λ t => t * t = 16

def travel_time_B : ℕ → Prop := λ t => 
  ∃ (rest : ℕ), t = 11 ∧ 2 * (t - rest) = 16 ∧ 4 * rest < 16 ∧ 4 * rest + 4 ≥ 16

-- Theorem statement
theorem distance_XY_is_16 : 
  (∃ t : ℕ, travel_time_A t ∧ travel_time_B t) → 
  (∃ d : ℕ, d = 16 ∧ ∀ t : ℕ, travel_time_A t → t * t = d) :=
by
  sorry

end distance_XY_is_16_l1454_145462


namespace smaller_number_proof_l1454_145434

theorem smaller_number_proof (x y : ℝ) : 
  x - y = 9 → x + y = 46 → min x y = 18.5 := by
sorry

end smaller_number_proof_l1454_145434


namespace CD_length_approx_l1454_145472

/-- A quadrilateral with intersecting diagonals -/
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)
  (BO : ℝ)
  (OD : ℝ)
  (AO : ℝ)
  (OC : ℝ)
  (AB : ℝ)

/-- The length of CD in the quadrilateral -/
def CD_length (q : Quadrilateral) : ℝ :=
  sorry

/-- Theorem stating the length of CD in the given quadrilateral -/
theorem CD_length_approx (q : Quadrilateral) 
  (h1 : q.BO = 3)
  (h2 : q.OD = 5)
  (h3 : q.AO = 7)
  (h4 : q.OC = 4)
  (h5 : q.AB = 5) :
  ∃ ε > 0, |CD_length q - 8.51| < ε :=
sorry

end CD_length_approx_l1454_145472


namespace park_outer_diameter_l1454_145479

/-- Given a circular park with a central fountain, surrounded by a garden ring and a walking path,
    this theorem proves the diameter of the outer boundary of the walking path. -/
theorem park_outer_diameter
  (fountain_diameter : ℝ)
  (garden_width : ℝ)
  (path_width : ℝ)
  (h1 : fountain_diameter = 20)
  (h2 : garden_width = 10)
  (h3 : path_width = 6) :
  2 * (fountain_diameter / 2 + garden_width + path_width) = 52 :=
by sorry

end park_outer_diameter_l1454_145479


namespace min_value_product_l1454_145492

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a/b + b/c + c/a + b/a + c/b + a/c = 9) :
  (a/b + b/c + c/a) * (b/a + c/b + a/c) ≥ 57 ∧
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    a/b + b/c + c/a + b/a + c/b + a/c = 9 ∧
    (a/b + b/c + c/a) * (b/a + c/b + a/c) = 57 := by
  sorry

end min_value_product_l1454_145492


namespace power_three_250_mod_13_l1454_145423

theorem power_three_250_mod_13 : 3^250 % 13 = 3 := by
  sorry

end power_three_250_mod_13_l1454_145423


namespace decision_box_has_two_exits_l1454_145463

/-- Represents a decision box in a program flowchart -/
structure DecisionBox where
  entrance : Nat
  exits : Nat

/-- Represents a flowchart -/
structure Flowchart where
  endpoints : Nat

/-- Theorem: A decision box in a program flowchart has exactly 2 exits -/
theorem decision_box_has_two_exits (d : DecisionBox) (f : Flowchart) : 
  d.entrance = 1 ∧ f.endpoints ≥ 1 → d.exits = 2 := by
  sorry

end decision_box_has_two_exits_l1454_145463


namespace faster_train_distance_and_time_l1454_145459

/-- Represents the speed and distance of a train -/
structure Train where
  speed : ℝ
  distance : ℝ

/-- Proves the distance covered by a faster train and the time taken -/
theorem faster_train_distance_and_time 
  (old_train : Train)
  (new_train : Train)
  (speed_increase_percent : ℝ)
  (h1 : old_train.distance = 300)
  (h2 : new_train.speed = old_train.speed * (1 + speed_increase_percent))
  (h3 : speed_increase_percent = 0.3)
  (h4 : new_train.speed = 120) : 
  new_train.distance = 390 ∧ (new_train.distance / new_train.speed) = 3.25 := by
  sorry

#check faster_train_distance_and_time

end faster_train_distance_and_time_l1454_145459


namespace hyperbola_eccentricity_l1454_145444

/-- Given a hyperbola with equation y²/2 - x²/8 = 1, its eccentricity is √5 -/
theorem hyperbola_eccentricity :
  ∀ (x y : ℝ), y^2/2 - x^2/8 = 1 → 
  ∃ (e : ℝ), e = Real.sqrt 5 ∧ e = Real.sqrt ((2 + 8) / 2) := by
  sorry

end hyperbola_eccentricity_l1454_145444


namespace total_points_sum_l1454_145426

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def gina_rolls : List ℕ := [6, 5, 2, 3, 4]
def helen_rolls : List ℕ := [1, 2, 4, 6, 3]

theorem total_points_sum : (gina_rolls.map g).sum + (helen_rolls.map g).sum = 44 := by
  sorry

end total_points_sum_l1454_145426


namespace inequality_and_optimization_l1454_145430

theorem inequality_and_optimization (m : ℝ) :
  (∀ x : ℝ, |x + 3| + |x + m| ≥ 2*m) →
  m ≤ 1 ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    a^2 + 2*b^2 + 3*c^2 ≥ 6/11 ∧
    (a^2 + 2*b^2 + 3*c^2 = 6/11 ↔ a = 6/11 ∧ b = 3/11 ∧ c = 2/11)) :=
by sorry

end inequality_and_optimization_l1454_145430


namespace square_cut_corners_l1454_145440

theorem square_cut_corners (s : ℝ) (h : (2 / 9) * s^2 = 288) :
  s - 2 * (1 / 3 * s) = 24 := by
  sorry

end square_cut_corners_l1454_145440


namespace sum_due_calculation_l1454_145411

/-- Represents the relationship between banker's discount, true discount, and face value -/
def banker_discount_relation (banker_discount true_discount face_value : ℚ) : Prop :=
  banker_discount = true_discount + (true_discount * banker_discount) / face_value

/-- Proves that given a banker's discount of 576 and a true discount of 480, the sum due (face value) is 2880 -/
theorem sum_due_calculation (banker_discount true_discount : ℚ) 
  (h1 : banker_discount = 576)
  (h2 : true_discount = 480) :
  ∃ face_value : ℚ, face_value = 2880 ∧ banker_discount_relation banker_discount true_discount face_value :=
by
  sorry

end sum_due_calculation_l1454_145411


namespace turnip_zhuchka_weight_ratio_l1454_145419

/-- The weight ratio between Zhuchka and a cat -/
def zhuchka_cat_ratio : ℚ := 3

/-- The weight ratio between a cat and a mouse -/
def cat_mouse_ratio : ℚ := 10

/-- The weight ratio between a turnip and a mouse -/
def turnip_mouse_ratio : ℚ := 60

/-- The weight ratio between a turnip and Zhuchka -/
def turnip_zhuchka_ratio : ℚ := 2

theorem turnip_zhuchka_weight_ratio :
  turnip_mouse_ratio / (cat_mouse_ratio * zhuchka_cat_ratio) = turnip_zhuchka_ratio :=
by sorry

end turnip_zhuchka_weight_ratio_l1454_145419


namespace least_common_denominator_of_fractions_l1454_145425

theorem least_common_denominator_of_fractions : Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7)))) = 420 := by
  sorry

end least_common_denominator_of_fractions_l1454_145425


namespace succeeding_number_in_base_3_l1454_145480

def base_3_to_decimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (3^i)) 0

def decimal_to_base_3 (n : Nat) : List Nat :=
  sorry  -- Implementation not provided as it's not needed for the statement

def M : List Nat := [0, 2, 0, 1]  -- Representing 1020 in base 3

theorem succeeding_number_in_base_3 :
  decimal_to_base_3 (base_3_to_decimal M + 1) = [1, 2, 0, 1] :=
sorry

end succeeding_number_in_base_3_l1454_145480


namespace degrees_to_minutes_03_negative_comparison_l1454_145491

-- Define the conversion factor from degrees to minutes
def degrees_to_minutes (d : ℝ) : ℝ := d * 60

-- Theorem 1: 0.3 degrees is equal to 18 minutes
theorem degrees_to_minutes_03 : degrees_to_minutes 0.3 = 18 := by sorry

-- Theorem 2: -2 is greater than -3
theorem negative_comparison : -2 > -3 := by sorry

end degrees_to_minutes_03_negative_comparison_l1454_145491


namespace triangle_ratio_equality_l1454_145449

/-- Given a triangle ABC with sides a, b, c, height ha corresponding to side a,
    and inscribed circle radius r, prove that (a + b + c) / a = ha / r -/
theorem triangle_ratio_equality (a b c ha r : ℝ) (ha_pos : ha > 0) (r_pos : r > 0) 
  (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) : (a + b + c) / a = ha / r :=
sorry

end triangle_ratio_equality_l1454_145449


namespace expenditure_ratio_l1454_145420

/-- Given two persons P1 and P2 with the following conditions:
    - The ratio of their incomes is 5:4
    - Each saves Rs. 1800
    - The income of P1 is Rs. 4500
    Prove that the ratio of their expenditures is 3:2 -/
theorem expenditure_ratio (income_p1 income_p2 expenditure_p1 expenditure_p2 savings : ℕ) :
  income_p1 = 4500 ∧
  5 * income_p2 = 4 * income_p1 ∧
  savings = 1800 ∧
  income_p1 - expenditure_p1 = savings ∧
  income_p2 - expenditure_p2 = savings →
  3 * expenditure_p2 = 2 * expenditure_p1 := by
  sorry

end expenditure_ratio_l1454_145420


namespace troy_needs_ten_dollars_l1454_145414

/-- The amount of additional money Troy needs to buy a new computer -/
def additional_money_needed (new_computer_cost initial_savings old_computer_price : ℕ) : ℕ :=
  new_computer_cost - (initial_savings + old_computer_price)

/-- Theorem: Troy needs $10 more to buy the new computer -/
theorem troy_needs_ten_dollars : 
  additional_money_needed 80 50 20 = 10 := by
  sorry

end troy_needs_ten_dollars_l1454_145414


namespace min_disks_required_l1454_145409

def total_files : ℕ := 40
def disk_capacity : ℚ := 2
def files_1mb : ℕ := 4
def files_0_9mb : ℕ := 16
def file_size_1mb : ℚ := 1
def file_size_0_9mb : ℚ := 9/10
def file_size_0_5mb : ℚ := 1/2

theorem min_disks_required :
  let remaining_files := total_files - files_1mb - files_0_9mb
  let total_size := files_1mb * file_size_1mb + 
                    files_0_9mb * file_size_0_9mb + 
                    remaining_files * file_size_0_5mb
  let min_disks := Int.ceil (total_size / disk_capacity)
  min_disks = 16 := by sorry

end min_disks_required_l1454_145409


namespace cookies_and_game_cost_l1454_145424

-- Define the quantities of each item
def bracelets : ℕ := 12
def necklaces : ℕ := 8
def rings : ℕ := 20

-- Define the costs to make each item
def bracelet_cost : ℚ := 1
def necklace_cost : ℚ := 2
def ring_cost : ℚ := 1/2

-- Define the selling prices of each item
def bracelet_price : ℚ := 3/2
def necklace_price : ℚ := 3
def ring_price : ℚ := 1

-- Define the target profit margin
def target_margin : ℚ := 1/2

-- Define the remaining money after purchases
def remaining_money : ℚ := 5

-- Theorem to prove
theorem cookies_and_game_cost :
  let total_cost := bracelets * bracelet_cost + necklaces * necklace_cost + rings * ring_cost
  let total_revenue := bracelets * bracelet_price + necklaces * necklace_price + rings * ring_price
  let profit := total_revenue - total_cost
  let target_profit := total_cost * target_margin
  let cost_of_purchases := profit - remaining_money
  cost_of_purchases = 43 := by sorry

end cookies_and_game_cost_l1454_145424


namespace fish_left_in_tank_l1454_145483

def fish_tank_problem (initial_fish : ℕ) (fish_taken_out : ℕ) : Prop :=
  initial_fish ≥ fish_taken_out ∧ 
  initial_fish - fish_taken_out = 3

theorem fish_left_in_tank : fish_tank_problem 19 16 := by
  sorry

end fish_left_in_tank_l1454_145483


namespace det_E_l1454_145438

/-- A 2x2 matrix representing a dilation centered at the origin with scale factor 5 -/
def E : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![5, 0],
    ![0, 5]]

/-- Theorem stating that the determinant of E is 25 -/
theorem det_E : Matrix.det E = 25 := by
  sorry

end det_E_l1454_145438


namespace molar_mass_X1_l1454_145417

-- Define the substances
def X1 : String := "CuO"
def X2 : String := "Cu"
def X3 : String := "CuSO4"
def X4 : String := "Cu(OH)2"

-- Define the molar masses
def molar_mass_Cu : Float := 63.5
def molar_mass_O : Float := 16.0

-- Define the chemical reactions
def reaction1 : String := "X1 + H2 → X2 + H2O"
def reaction2 : String := "X2 + H2SO4 → X3 + H2"
def reaction3 : String := "X3 + 2KOH → X4 + K2SO4"
def reaction4 : String := "X4 → X1 + H2O"

-- Define the properties of the substances
def X1_properties : String := "black powder"
def X2_properties : String := "red-colored substance"
def X3_properties : String := "blue-colored solution"
def X4_properties : String := "blue precipitate"

-- Theorem to prove
theorem molar_mass_X1 : 
  molar_mass_Cu + molar_mass_O = 79.5 := by sorry

end molar_mass_X1_l1454_145417


namespace system_of_equations_l1454_145458

theorem system_of_equations (x y a b : ℝ) (h1 : 4 * x - 3 * y = a) (h2 : 6 * y - 8 * x = b) (h3 : b ≠ 0) :
  a / b = -1 / 2 := by
  sorry

end system_of_equations_l1454_145458


namespace optimal_system_is_best_l1454_145429

/-- Represents a monetary system with three coin denominations -/
structure MonetarySystem where
  d1 : ℕ
  d2 : ℕ
  d3 : ℕ
  h1 : 0 < d1 ∧ d1 < d2 ∧ d2 < d3
  h2 : d3 ≤ 100

/-- Calculates the minimum number of coins required for a given monetary system -/
def minCoinsRequired (system : MonetarySystem) : ℕ := sorry

/-- The optimal monetary system -/
def optimalSystem : MonetarySystem :=
  { d1 := 1, d2 := 7, d3 := 14,
    h1 := by simp,
    h2 := by simp }

theorem optimal_system_is_best :
  (∀ system : MonetarySystem, minCoinsRequired system ≥ minCoinsRequired optimalSystem) ∧
  minCoinsRequired optimalSystem = 14 := by sorry

end optimal_system_is_best_l1454_145429


namespace min_value_fraction_l1454_145497

theorem min_value_fraction (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (hsum : p + q + r = 2) : 
  (p + q) / (p * q * r) ≥ 9 ∧ ∃ p q r, p > 0 ∧ q > 0 ∧ r > 0 ∧ p + q + r = 2 ∧ (p + q) / (p * q * r) = 9 :=
sorry

end min_value_fraction_l1454_145497


namespace seven_people_seven_rooms_l1454_145418

/-- The number of ways to assign n people to m rooms with at most k people per room -/
def assignmentCount (n m k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 131460 ways to assign 7 people to 7 rooms with at most 2 people per room -/
theorem seven_people_seven_rooms : assignmentCount 7 7 2 = 131460 := by sorry

end seven_people_seven_rooms_l1454_145418


namespace credit_limit_problem_l1454_145466

/-- The credit limit problem -/
theorem credit_limit_problem (payments_made : ℕ) (remaining_payment : ℕ) 
  (h1 : payments_made = 38)
  (h2 : remaining_payment = 62) :
  payments_made + remaining_payment = 100 := by
  sorry

end credit_limit_problem_l1454_145466


namespace retailer_profit_percentage_l1454_145404

theorem retailer_profit_percentage 
  (cost_price : ℝ) 
  (markup_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : markup_percentage = 60) 
  (h2 : discount_percentage = 25) 
  (h3 : cost_price > 0) : 
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 20 := by
sorry

end retailer_profit_percentage_l1454_145404


namespace sequence_existence_l1454_145416

def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → f n < f m

theorem sequence_existence
  (f : ℕ → ℕ) (h_inc : StrictlyIncreasing f) :
  (∃ y : ℕ → ℝ, (∀ n, y n > 0) ∧
    (∀ n m, n < m → y m < y n) ∧
    (∀ ε > 0, ∃ N, ∀ n ≥ N, y n < ε) ∧
    (∀ n, y n ≤ 2 * y (f n))) ∧
  (∀ x : ℕ → ℝ,
    (∀ n m, n < m → x m < x n) →
    (∀ ε > 0, ∃ N, ∀ n ≥ N, x n < ε) →
    ∃ y : ℕ → ℝ,
      (∀ n m, n < m → y m < y n) ∧
      (∀ ε > 0, ∃ N, ∀ n ≥ N, y n < ε) ∧
      (∀ n, x n ≤ y n ∧ y n ≤ 2 * y (f n))) :=
by
  sorry

end sequence_existence_l1454_145416


namespace jackson_decorations_to_friend_l1454_145401

/-- Represents the number of Christmas decorations Mrs. Jackson gives to her friend. -/
def decorations_to_friend (total_boxes : ℕ) (decorations_per_box : ℕ) (used_decorations : ℕ) (given_to_neighbor : ℕ) : ℕ :=
  total_boxes * decorations_per_box - used_decorations - given_to_neighbor

/-- Proves that Mrs. Jackson gives 17 decorations to her friend under the given conditions. -/
theorem jackson_decorations_to_friend :
  decorations_to_friend 6 25 58 75 = 17 := by
  sorry

end jackson_decorations_to_friend_l1454_145401


namespace quadratic_inequality_implies_m_range_l1454_145405

theorem quadratic_inequality_implies_m_range :
  (∀ x : ℝ, x^2 + m*x + 2*m - 3 ≥ 0) → m ∈ Set.Icc 2 6 :=
by sorry

end quadratic_inequality_implies_m_range_l1454_145405


namespace unit_digit_of_8_power_1533_l1454_145403

theorem unit_digit_of_8_power_1533 : (8^1533 : ℕ) % 10 = 8 := by sorry

end unit_digit_of_8_power_1533_l1454_145403


namespace calculation_proof_l1454_145428

theorem calculation_proof : 4 * Real.sqrt 24 * (Real.sqrt 6 / 8) / Real.sqrt 3 - 3 * Real.sqrt 3 = - Real.sqrt 3 := by
  sorry

end calculation_proof_l1454_145428


namespace candy_bar_sales_theorem_l1454_145471

/-- Calculates the total money earned from candy bar sales given the number of members,
    average number of candy bars sold per member, and the cost per candy bar. -/
def total_money_earned (num_members : ℕ) (avg_bars_per_member : ℕ) (cost_per_bar : ℚ) : ℚ :=
  (num_members * avg_bars_per_member : ℚ) * cost_per_bar

/-- Proves that a group of 20 members selling an average of 8 candy bars at $0.50 each
    earns a total of $80 from their sales. -/
theorem candy_bar_sales_theorem :
  total_money_earned 20 8 (1/2) = 80 := by
  sorry

end candy_bar_sales_theorem_l1454_145471


namespace decimal_shift_problem_l1454_145467

theorem decimal_shift_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 * (1 / x)) : 
  x = Real.sqrt 30 / 100 := by
  sorry

end decimal_shift_problem_l1454_145467


namespace min_value_theorem_l1454_145470

theorem min_value_theorem (n : ℕ+) : 
  (n : ℝ) / 3 + 27 / (n : ℝ) ≥ 6 ∧ 
  ((n : ℝ) / 3 + 27 / (n : ℝ) = 6 ↔ n = 9) := by
sorry

end min_value_theorem_l1454_145470


namespace circle_radius_l1454_145496

/-- The radius of a circle given by the equation x^2 + 10x + y^2 - 8y + 25 = 0 is 4 -/
theorem circle_radius (x y : ℝ) : x^2 + 10*x + y^2 - 8*y + 25 = 0 → ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 4^2 := by
  sorry

end circle_radius_l1454_145496


namespace cosine_identity_problem_l1454_145427

theorem cosine_identity_problem (α : Real) 
  (h : Real.cos (π / 4 + α) = -1 / 3) : 
  (Real.sin (2 * α) - 2 * Real.sin α ^ 2) / Real.sqrt (1 - Real.cos (2 * α)) = 2 / 3 ∨ 
  (Real.sin (2 * α) - 2 * Real.sin α ^ 2) / Real.sqrt (1 - Real.cos (2 * α)) = -2 / 3 :=
sorry

end cosine_identity_problem_l1454_145427


namespace arithmetic_sequence_property_l1454_145484

/-- An arithmetic sequence is a sequence where the difference between 
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem stating that for an arithmetic sequence satisfying 
    the given condition, 2a_9 - a_10 = 24 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_sum : a 1 + 3 * a 8 + a 15 = 120) : 
  2 * a 9 - a 10 = 24 := by
  sorry

end arithmetic_sequence_property_l1454_145484


namespace new_pages_read_per_week_jim_new_pages_read_l1454_145455

/-- Calculates the new number of pages read per week after changes in reading speed and time --/
theorem new_pages_read_per_week
  (initial_rate : ℝ)
  (initial_pages : ℝ)
  (speed_increase : ℝ)
  (time_decrease : ℝ)
  (h1 : initial_rate = 40)
  (h2 : initial_pages = 600)
  (h3 : speed_increase = 1.5)
  (h4 : time_decrease = 4)
  : ℝ :=
  by
  -- Proof goes here
  sorry

/-- The main theorem stating that Jim now reads 660 pages per week --/
theorem jim_new_pages_read :
  new_pages_read_per_week 40 600 1.5 4 rfl rfl rfl rfl = 660 :=
by
  -- Proof goes here
  sorry

end new_pages_read_per_week_jim_new_pages_read_l1454_145455


namespace sum_and_difference_of_numbers_l1454_145476

theorem sum_and_difference_of_numbers : ∃ (a b : ℕ), 
  b = 100 * a ∧ 
  a + b = 36400 ∧ 
  b - a = 35640 := by
sorry

end sum_and_difference_of_numbers_l1454_145476


namespace forty_five_candies_cost_candies_for_fifty_l1454_145450

-- Define the cost of one candy in rubles
def cost_per_candy : ℝ := 1

-- Define the relationship between 45 candies and their cost
theorem forty_five_candies_cost (c : ℝ) : c * 45 = 45 := by sorry

-- Define the number of candies that can be bought for 20 rubles
def candies_for_twenty : ℝ := 20

-- Theorem to prove
theorem candies_for_fifty : ℝ := by
  -- The number of candies that can be bought for 50 rubles is 50
  exact 50

/- Proof
sorry
-/

end forty_five_candies_cost_candies_for_fifty_l1454_145450


namespace sum_of_x_coordinates_l1454_145482

-- Define the points
def O : ℝ × ℝ := (0, 0)
def P : ℝ → ℝ × ℝ := λ t ↦ (5*t, 12*t)
def Q : ℝ → ℝ × ℝ := λ t ↦ (8*t, 6*t)

-- State the theorem
theorem sum_of_x_coordinates (t : ℝ) : 
  (P t).1 + (Q t).1 = 13*t := by
  sorry

end sum_of_x_coordinates_l1454_145482


namespace distinct_roots_find_m_l1454_145454

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 - (2*m + 1)*x + m^2 + m

-- Define the discriminant
def discriminant (m : ℝ) : ℝ := (-(2*m + 1))^2 - 4*(m^2 + m)

-- Define the condition for the roots
def root_condition (a b : ℝ) : Prop := (2*a + b) * (a + 2*b) = 20

-- Theorem 1: The equation always has two distinct real roots
theorem distinct_roots (m : ℝ) : discriminant m > 0 :=
sorry

-- Theorem 2: When the root condition is satisfied, m = -2 or m = 1
theorem find_m (m : ℝ) :
  (∃ a b : ℝ, quadratic m a = 0 ∧ quadratic m b = 0 ∧ a ≠ b ∧ root_condition a b) →
  m = -2 ∨ m = 1 :=
sorry

end distinct_roots_find_m_l1454_145454


namespace probability_A_timeliness_at_least_75_l1454_145437

/-- Represents the survey data for a company -/
structure SurveyData where
  total_questionnaires : ℕ
  excellent_timeliness : ℕ
  good_timeliness : ℕ
  fair_timeliness : ℕ

/-- Calculates the probability of timeliness rating at least 75 points -/
def probabilityAtLeast75 (data : SurveyData) : ℚ :=
  (data.excellent_timeliness + data.good_timeliness : ℚ) / data.total_questionnaires

/-- The survey data for company A -/
def companyA : SurveyData := {
  total_questionnaires := 120,
  excellent_timeliness := 29,
  good_timeliness := 47,
  fair_timeliness := 44
}

/-- Theorem stating the probability of company A's delivery timeliness being at least 75 points -/
theorem probability_A_timeliness_at_least_75 :
  probabilityAtLeast75 companyA = 19 / 30 := by sorry

end probability_A_timeliness_at_least_75_l1454_145437


namespace parallel_lines_a_perpendicular_lines_a_l1454_145494

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 4 * y + 6 = 0
def l₂ (a x y : ℝ) : Prop := ((3/4) * a + 1) * x + a * y - 3/2 = 0

-- Parallel lines condition
def parallel (a : ℝ) : Prop := a^2 - 4 * ((3/4) * a + 1) = 0 ∧ 4 * (-3/2) - 6 * a ≠ 0

-- Perpendicular lines condition
def perpendicular (a : ℝ) : Prop := a * ((3/4) * a + 1) + 4 * a = 0

-- Theorem for parallel lines
theorem parallel_lines_a (a : ℝ) :
  parallel a → a = 4 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines_a (a : ℝ) :
  perpendicular a → a = 0 ∨ a = -20/3 :=
sorry

end parallel_lines_a_perpendicular_lines_a_l1454_145494


namespace abs_negative_two_l1454_145431

theorem abs_negative_two : |(-2 : ℤ)| = 2 := by
  sorry

end abs_negative_two_l1454_145431


namespace common_factor_proof_l1454_145478

theorem common_factor_proof (n : ℤ) : ∃ (k₁ k₂ : ℤ), 
  n^2 - 1 = (n + 1) * k₁ ∧ n^2 + n = (n + 1) * k₂ := by
  sorry

end common_factor_proof_l1454_145478


namespace profit_ratio_of_partners_l1454_145415

theorem profit_ratio_of_partners (p q : ℕ) (investment_ratio : Rat) (time_p time_q : ℕ) 
  (h1 : investment_ratio = 7 / 5)
  (h2 : time_p = 7)
  (h3 : time_q = 14) :
  (p : Rat) / q = 7 / 10 := by
sorry

end profit_ratio_of_partners_l1454_145415


namespace left_handed_sci_fi_fans_count_l1454_145442

/-- Represents the book club with its member properties -/
structure BookClub where
  total_members : ℕ
  left_handed : ℕ
  sci_fi_fans : ℕ
  right_handed_non_sci_fi : ℕ

/-- The number of left-handed members who like sci-fi books in the book club -/
def left_handed_sci_fi_fans (club : BookClub) : ℕ :=
  club.total_members - (club.left_handed + club.sci_fi_fans + club.right_handed_non_sci_fi) + club.left_handed + club.sci_fi_fans - club.total_members

/-- Theorem stating that the number of left-handed sci-fi fans is 4 for the given book club -/
theorem left_handed_sci_fi_fans_count (club : BookClub) 
  (h1 : club.total_members = 30)
  (h2 : club.left_handed = 12)
  (h3 : club.sci_fi_fans = 18)
  (h4 : club.right_handed_non_sci_fi = 4) :
  left_handed_sci_fi_fans club = 4 := by
  sorry

end left_handed_sci_fi_fans_count_l1454_145442


namespace bus_cost_relationship_l1454_145400

/-- The functional relationship between the number of large buses purchased and the total cost -/
theorem bus_cost_relationship (x : ℝ) (y : ℝ) : y = 22 * x + 800 ↔ 
  y = 62 * x + 40 * (20 - x) := by sorry

end bus_cost_relationship_l1454_145400


namespace cosine_equation_roots_l1454_145432

theorem cosine_equation_roots (θ : Real) :
  (0 ≤ θ) ∧ (θ < 360) →
  (3 * Real.cos θ + 1 / Real.cos θ = 4) →
  ∃ p : Nat, p = 3 := by sorry

end cosine_equation_roots_l1454_145432


namespace container_volume_ratio_l1454_145475

theorem container_volume_ratio :
  ∀ (V₁ V₂ : ℝ), V₁ > 0 → V₂ > 0 →
  (3/4 : ℝ) * V₁ = (2/3 : ℝ) * V₂ →
  V₁ / V₂ = 8/9 := by
sorry

end container_volume_ratio_l1454_145475


namespace line_through_intersection_and_parallel_l1454_145468

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := x + 3 * y - 3 = 0
def l2 (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 2 * x + y - 3 = 0

-- Theorem statement
theorem line_through_intersection_and_parallel :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a * x + b * y + c = 0 ↔ 
      (∃ x0 y0 : ℝ, l1 x0 y0 ∧ l2 x0 y0 ∧ 
        (y - y0 = -(a/b) * (x - x0))) ∧
      (∃ k : ℝ, a/b = -2)) :=
sorry

end line_through_intersection_and_parallel_l1454_145468


namespace bijection_iteration_fixed_point_l1454_145452

theorem bijection_iteration_fixed_point {n : ℕ} (f : Fin n → Fin n) (h : Function.Bijective f) :
  ∃ M : ℕ+, ∀ i : Fin n, (f^[M.val] i) = f i := by
  sorry

end bijection_iteration_fixed_point_l1454_145452


namespace pet_store_animals_l1454_145443

/-- Calculates the total number of animals in a pet store given the number of dogs and ratios for other animals. -/
def total_animals (num_dogs : ℕ) : ℕ :=
  let num_cats := num_dogs / 2
  let num_birds := num_dogs * 2
  let num_fish := num_dogs * 3
  num_dogs + num_cats + num_birds + num_fish

/-- Theorem stating that a pet store with 6 dogs and specified ratios of other animals has 39 animals in total. -/
theorem pet_store_animals : total_animals 6 = 39 := by
  sorry

end pet_store_animals_l1454_145443


namespace algebra_test_average_l1454_145477

theorem algebra_test_average (total_average : ℝ) (male_average : ℝ) (male_count : ℕ) (female_count : ℕ) 
  (h1 : total_average = 90)
  (h2 : male_average = 87)
  (h3 : male_count = 8)
  (h4 : female_count = 12) :
  let total_count := male_count + female_count
  let total_score := total_average * total_count
  let male_score := male_average * male_count
  let female_score := total_score - male_score
  female_score / female_count = 92 := by
sorry

end algebra_test_average_l1454_145477


namespace michaels_fish_count_l1454_145495

theorem michaels_fish_count (original_count added_count total_count : ℕ) : 
  added_count = 18 →
  total_count = 49 →
  original_count + added_count = total_count :=
by
  sorry

end michaels_fish_count_l1454_145495


namespace other_root_of_complex_quadratic_l1454_145487

theorem other_root_of_complex_quadratic (z : ℂ) :
  z = 4 + 7*I ∧ z^2 = -73 + 24*I → (-z)^2 = -73 + 24*I := by
  sorry

end other_root_of_complex_quadratic_l1454_145487


namespace sum_of_products_even_l1454_145412

/-- Represents a regular hexagon with natural numbers assigned to its vertices -/
structure Hexagon where
  vertices : Fin 6 → ℕ

/-- The sum of products of adjacent vertex pairs in a hexagon -/
def sum_of_products (h : Hexagon) : ℕ :=
  (h.vertices 0 * h.vertices 1) + (h.vertices 1 * h.vertices 2) +
  (h.vertices 2 * h.vertices 3) + (h.vertices 3 * h.vertices 4) +
  (h.vertices 4 * h.vertices 5) + (h.vertices 5 * h.vertices 0)

/-- A hexagon with opposite vertices having the same value -/
def opposite_same_hexagon (a b c : ℕ) : Hexagon :=
  { vertices := fun i => match i with
    | 0 | 3 => a
    | 1 | 4 => b
    | 2 | 5 => c }

theorem sum_of_products_even (a b c : ℕ) :
  Even (sum_of_products (opposite_same_hexagon a b c)) := by
  sorry

#check sum_of_products_even

end sum_of_products_even_l1454_145412


namespace set_operation_result_l1454_145441

open Set

def U : Set Int := univ
def A : Set Int := {-2, -1, 0, 1, 2}
def B : Set Int := {-1, 0, 1, 2, 3}

theorem set_operation_result : A ∩ (U \ B) = {-2} := by
  sorry

end set_operation_result_l1454_145441


namespace sum_of_coefficients_is_zero_l1454_145473

def polynomial (x : ℝ) : ℝ := 4 * (2 * x^8 + 3 * x^5 - 5) + 6 * (x^6 - 5 * x^3 + 4)

theorem sum_of_coefficients_is_zero : 
  polynomial 1 = 0 := by sorry

end sum_of_coefficients_is_zero_l1454_145473


namespace floor_of_e_l1454_145499

-- Define e as the base of the natural logarithm
noncomputable def e : ℝ := Real.exp 1

-- State the theorem
theorem floor_of_e : ⌊e⌋ = 2 := by sorry

end floor_of_e_l1454_145499


namespace largest_common_value_proof_l1454_145448

/-- First arithmetic progression with initial term 4 and common difference 5 -/
def seq1 (n : ℕ) : ℕ := 4 + 5 * n

/-- Second arithmetic progression with initial term 5 and common difference 8 -/
def seq2 (m : ℕ) : ℕ := 5 + 8 * m

/-- The largest common value less than 1000 in both sequences -/
def largest_common_value : ℕ := 989

theorem largest_common_value_proof :
  (∃ n m : ℕ, seq1 n = largest_common_value ∧ seq2 m = largest_common_value) ∧
  (∀ k : ℕ, k < 1000 → (∃ n m : ℕ, seq1 n = k ∧ seq2 m = k) → k ≤ largest_common_value) :=
sorry

end largest_common_value_proof_l1454_145448


namespace monotone_increasing_range_l1454_145498

/-- A function g(x) = ax³ + ax² + x is monotonically increasing on ℝ -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (a * x^3 + a * x^2 + x) < (a * y^3 + a * y^2 + y)

/-- The range of a for which g(x) = ax³ + ax² + x is monotonically increasing on ℝ -/
theorem monotone_increasing_range :
  ∀ a : ℝ, is_monotone_increasing a ↔ (0 ≤ a ∧ a ≤ 3) :=
by sorry

end monotone_increasing_range_l1454_145498
